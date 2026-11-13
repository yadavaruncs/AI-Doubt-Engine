"""Graph and API behaviour with the LLM scripted, so failures are reproducible."""

import json
from types import SimpleNamespace as NS

import httpx
import openai
import pytest
import yaml
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import app.graph as graph
import app.main as main
from app.cache import SemanticCache
from app.config import EXAMPLE_API_KEY, ROOT
from app.llm import Completion


def _tool_msg(name: str, arguments: str) -> NS:
    call = NS(id="call_1", function=NS(name=name, arguments=arguments))
    return NS(content=None, tool_calls=[call])


class ScriptedLLM:
    """Stands in for client(): replays messages and records each request."""

    def __init__(self, *messages):
        self.script = list(messages)
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        msg = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(msg, Exception):
            raise msg
        return NS(choices=[NS(message=msg)], usage=None)


def _run_numeric(monkeypatch, *messages):
    llm = ScriptedLLM(*messages)
    monkeypatch.setattr(graph, "client", lambda: llm)
    return graph.numeric_node({"question": "q", "context": ""}), llm


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("calculate", '{"expression": "1 +'),  # truncated by max_tokens
        ("calculate", '"just a string"'),
        ("no_such_tool", "{}"),
    ],
)
def test_bad_tool_call_is_reported_to_the_model(monkeypatch, name, arguments):
    state, llm = _run_numeric(
        monkeypatch, _tool_msg(name, arguments), NS(content="done", tool_calls=None)
    )
    assert state["answer"] == "done"
    assert state["tool_calls"][0]["result"]["ok"] is False
    assert json.loads(llm.requests[1]["messages"][-1]["content"])["ok"] is False


def _provider_400(code: str) -> openai.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "http://llm"))
    return openai.BadRequestError("rejected", response=response, body={"code": code})


def test_provider_rejected_tool_call_is_reported_to_the_model(monkeypatch):
    state, llm = _run_numeric(
        monkeypatch, _provider_400("tool_use_failed"), NS(content="done", tool_calls=None)
    )
    assert state["answer"] == "done"
    assert "calculate" in llm.requests[1]["messages"][-1]["content"]


def test_other_provider_errors_still_raise(monkeypatch):
    with pytest.raises(openai.BadRequestError):
        _run_numeric(monkeypatch, _provider_400("context_length_exceeded"))


def test_missing_tool_argument_is_reported_to_the_model(monkeypatch):
    monkeypatch.setitem(graph.TOOL_IMPLS, "calculate", lambda expression: {"ok": True})
    state, _ = _run_numeric(
        monkeypatch, _tool_msg("calculate", "{}"), NS(content="done", tool_calls=None)
    )
    assert state["tool_calls"][0]["result"]["ok"] is False


def test_last_round_must_answer_instead_of_calling_tools(monkeypatch):
    monkeypatch.setitem(graph.TOOL_IMPLS, "calculate", lambda expression: {"ok": True})
    _, llm = _run_numeric(monkeypatch, _tool_msg("calculate", '{"expression": "1"}'))
    assert [r["tool_choice"] for r in llm.requests] == ["auto"] * 3 + ["none"]


@pytest.mark.parametrize(
    "output,question,label",
    [
        ("numeric", "q", "numeric"),
        ('"Numeric".', "q", "numeric"),
        ("**conceptual**", "q", "conceptual"),
        ("Category: out of scope", "q", "out_of_scope"),
        ("out-of-scope", "q", "out_of_scope"),
        ("no idea", "a 20 m tower", "numeric"),
        ("no idea", "what is inertia", "conceptual"),
    ],
)
def test_router_parsing(monkeypatch, output, question, label):
    monkeypatch.setattr(graph, "complete", lambda *a, **k: Completion(output, "m"))
    assert graph.route_node({"question": question})["route"] == label


def test_retrieve_node_applies_ablation_config(monkeypatch):
    seen = {}

    def fake_retrieve(query, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr(graph, "retrieve", fake_retrieve)
    cfg = {"use_hybrid": False, "use_rerank": False}
    graph.retrieve_node({"question": "q", "retrieval": cfg})
    assert seen == cfg


def _fake_complete(prompts):
    def complete(prompt, **kwargs):
        prompts.append(prompt)
        if "standalone question" in prompt:
            return Completion("What if the mass of the ball doubles?", "m", 5, 5)
        if "Classify" in prompt:
            return Completion("conceptual", "m", 5, 5)
        return Completion("answer", "m", 5, 5)

    return complete


def test_followup_is_condensed_before_retrieval(monkeypatch, seeded_store):
    prompts = []
    monkeypatch.setattr(graph, "complete", _fake_complete(prompts))
    state = graph.GRAPH.invoke(
        {
            "question": "what if it doubles?",
            "history": [{"role": "user", "content": "momentum of a ball"}],
        }
    )
    assert "momentum of a ball" in prompts[0]
    assert "What if the mass of the ball doubles?" in prompts[1]
    assert state["prompt_tokens"] == 15


def test_no_history_skips_the_condense_call(monkeypatch, seeded_store):
    prompts = []
    monkeypatch.setattr(graph, "complete", _fake_complete(prompts))
    state = graph.GRAPH.invoke({"question": "what is inertia"})
    assert state["answer"] == "answer"
    assert len(prompts) == 2


# ------------------------------------------------------------------- API
@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main, "CACHE", SemanticCache(max_size=8, threshold=0.99))
    monkeypatch.setattr(main, "_LATENCIES", main.deque(maxlen=500))
    monkeypatch.setattr(main, "_rate_hits", {})
    return TestClient(main.app)


def _stub_graph(monkeypatch, route="conceptual"):
    calls = []

    def invoke(state):
        calls.append(state)
        return {"answer": "42", "route": route}

    monkeypatch.setattr(main, "GRAPH", NS(invoke=invoke))
    return calls


def test_backend_failure_is_a_clean_503(api, monkeypatch):
    def boom(_):
        raise ConnectionError("qdrant at 10.0.0.5 refused")

    monkeypatch.setattr(main, "GRAPH", NS(invoke=boom))
    resp = api.post("/ask", json={"question": "q"})
    assert resp.status_code == 503
    assert "10.0.0.5" not in resp.text


def test_numeric_answers_are_not_cached(api, monkeypatch):
    calls = _stub_graph(monkeypatch, route="numeric")
    for _ in range(2):
        assert api.post("/ask", json={"question": "a 20 m tower"}).json()["cached"] is False
    assert len(calls) == 2


def test_cache_hits_stay_out_of_latency_metrics(api, monkeypatch):
    calls = _stub_graph(monkeypatch)
    api.post("/ask", json={"question": "what is inertia"})
    assert api.post("/ask", json={"question": "what is inertia"}).json()["cached"] is True
    assert len(calls) == 1
    assert api.get("/metrics").json()["requests"] == 1


def test_followups_bypass_cache_and_reach_the_graph(api, monkeypatch):
    calls = _stub_graph(monkeypatch)
    history = [{"role": "user", "content": "what is momentum"}]
    for _ in range(2):
        api.post("/ask", json={"question": "why?", "history": history})
    assert len(calls) == 2
    assert calls[0]["history"] == history


def test_history_is_length_limited(api):
    history = [{"role": "user", "content": "x"}] * 7
    assert api.post("/ask", json={"question": "q", "history": history}).status_code == 422


def test_api_key_required_when_configured(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "API_KEY", "secret")
    assert api.post("/ask", json={"question": "q"}).status_code == 401
    ok = api.post("/ask", json={"question": "q"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_missing_api_key_header_is_401_and_never_reaches_the_graph(api, monkeypatch):
    calls = _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "API_KEY", "secret")
    assert api.post("/ask", json={"question": "q"}).status_code == 401
    assert calls == []


def test_non_ascii_api_key_is_a_401_not_a_crash(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "API_KEY", "secret")
    headers = {"X-API-Key": "é".encode("latin-1")}
    response = api.post("/ask", json={"question": "q"}, headers=headers)
    assert response.status_code == 401


def test_rate_limit(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 2)
    codes = [api.post("/ask", json={"question": "q"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_rate_limit_slides_instead_of_resetting(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 2)
    clock = [1000.0]
    monkeypatch.setattr(main.time, "monotonic", lambda: clock[0])

    def post() -> int:
        return api.post("/ask", json={"question": "q"}).status_code

    assert post() == 200
    clock[0] = 1059.0
    assert post() == 200
    clock[0] = 1059.5  # would be a new clock-minute, but both hits are within 60s
    assert post() == 429
    clock[0] = 1060.5  # the first hit has aged out
    assert post() == 200


def test_rate_limit_holds_across_a_minute_boundary(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 5)
    clock = [1075.0]  # 1080 is a clock-minute boundary, where the old counter reset

    def now() -> float:
        return clock[0]

    monkeypatch.setattr(main, "time", NS(time=now, monotonic=now, perf_counter=now))

    codes = []
    for _ in range(10):
        codes.append(api.post("/ask", json={"question": "q"}).status_code)
        clock[0] += 1
    assert codes == [200] * 5 + [429] * 5


def _post_from(client_ip: str, trusted_proxy: str) -> int:
    proxied = TestClient(ProxyHeadersMiddleware(main.app, trusted_hosts=trusted_proxy))
    headers = {"X-Forwarded-For": client_ip}
    return proxied.post("/ask", json={"question": "q"}, headers=headers).status_code


def test_rate_limit_is_per_client_behind_a_trusted_proxy(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 1)
    codes = [_post_from(ip, "testclient") for ip in ("1.1.1.1", "2.2.2.2", "1.1.1.1")]
    assert codes == [200, 200, 429]


def test_rate_limit_ignores_forwarded_ip_from_an_untrusted_peer(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "RATE_LIMIT_PER_MIN", 1)
    codes = [_post_from(ip, "10.0.0.1") for ip in ("1.1.1.1", "2.2.2.2")]
    assert codes == [200, 429]


def test_rate_limit_forgets_quiet_ips(api, monkeypatch):
    _stub_graph(monkeypatch)
    monkeypatch.setattr(main, "_RATE_MAX_IPS", 2)
    monkeypatch.setattr(main.time, "monotonic", lambda: 1000.0)
    main._rate_hits.update({"a": main.deque([1.0]), "b": main.deque([2.0])})
    assert api.post("/ask", json={"question": "q"}).status_code == 200
    assert "a" not in main._rate_hits and "b" not in main._rate_hits


def test_refuses_to_boot_without_api_key_unless_allowed(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", "")
    monkeypatch.setattr(main, "ALLOW_UNAUTHENTICATED", False)
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(main.app):
            pass
    monkeypatch.setattr(main, "ensure_collection", lambda: None)
    monkeypatch.setattr(main, "ALLOW_UNAUTHENTICATED", True)
    with TestClient(main.app):
        pass


def test_refuses_to_boot_with_the_example_key_even_if_unauthenticated_is_allowed(monkeypatch):
    monkeypatch.setattr(main, "API_KEY", EXAMPLE_API_KEY)
    monkeypatch.setattr(main, "ALLOW_UNAUTHENTICATED", True)
    with pytest.raises(RuntimeError, match="example"):
        with TestClient(main.app):
            pass


def test_cache_finds_best_match_and_drops_expired(monkeypatch):
    vectors = {"a": [1.0, 0.0], "b": [0.0, 2.0], "near b": [0.0, 1.0], "zero": [0.0, 0.0]}
    monkeypatch.setattr("app.cache.embed_dense_one", lambda q: vectors[q])
    cache = SemanticCache(max_size=2, threshold=0.9, ttl=100)
    assert cache.get("a") is None  # empty cache

    cache.put("a", {"answer": "A"})
    cache.put("b", {"answer": "B"})
    assert cache.get("near b")["answer"] == "B"
    assert cache.get("zero") is None

    vectors["c"] = [-1.0, 0.0]
    cache.put("c", {"answer": "C"})  # evicts "a", the least recently used
    assert cache.get("a") is None

    monkeypatch.setattr("app.cache.time.monotonic", lambda: 1e12)
    assert cache.get("b") is None
    assert cache.stats()["entries"] == 0


def _env_value(name: str) -> str:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    return next(line.split("=", 1)[1] for line in lines if line.startswith(f"{name}="))


def test_env_example_starts_with_a_key_and_says_to_replace_it():
    assert len(_env_value("API_KEY")) >= 32
    assert _env_value("API_KEY") == EXAMPLE_API_KEY  # or the startup check stops matching
    assert "replace API_KEY" in (ROOT / "README.md").read_text(encoding="utf-8")


def test_qdrant_is_not_published_to_the_host():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert "ports" not in compose["services"]["qdrant"]
    assert compose["services"]["api"]["environment"]["QDRANT_URL"] == "http://qdrant:6333"
