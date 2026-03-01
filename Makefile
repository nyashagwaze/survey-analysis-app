.PHONY: setup dev run-app screenshots lint format test

setup:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -r requirements-dev.txt

run-app:
	streamlit run streamlit_app.py

screenshots:
	python -m playwright install
	python scripts/capture_streamlit_screenshots.py --use-sample

lint:
	ruff check .

format:
	black .

test:
	pytest -q
