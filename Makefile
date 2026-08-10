install:
	python -m pip install -r requirements.txt

verify:
	python verify.py
	pytest -q

train:
	python scripts/train_deploy_model.py --model tft --epochs 5

serve:
	uvicorn api.app:app --host 0.0.0.0 --port 8000

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f
