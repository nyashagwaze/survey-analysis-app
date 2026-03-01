.PHONY: setup dev run-app lint format test

setup:
	python -m pip install -r requirements.txt

dev:
	python -m pip install -r requirements-dev.txt

run-app:
	streamlit run streamlit_app.py

lint:
	ruff check .

format:
	black .

test:
	pytest -q
