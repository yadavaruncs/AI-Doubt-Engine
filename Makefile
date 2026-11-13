.PHONY: up down ingest eval dev logs

up:          ## start qdrant + api + ui
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f api

ingest:      ## index everything under ./data
	docker compose exec api python -m app.ingest --path data/ --recreate

eval:        ## run the ablation harness (retrieval only, no API cost)
	docker compose exec api python -m eval.run_eval --dataset eval/dataset.jsonl --no-ragas

dev: export ALLOW_UNAUTHENTICATED := 1
dev:         ## run the api locally without docker
	uvicorn app.main:app --reload --port 8000
