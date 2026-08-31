.PHONY: venv install train predict eda notebook

venv:
	python -m venv .venv

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python src/train.py

predict:
	python src/predict.py

eda:
	python src/eda.py

notebook:
	python -m jupyter notebook
