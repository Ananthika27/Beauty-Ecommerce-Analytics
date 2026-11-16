.PHONY: all pipeline model basket charts app test
all: pipeline model basket charts

pipeline:  ## clean raw data, build analysis tables
	python src/pipeline.py
model:     ## train + evaluate next-month purchase model
	python src/churn_model.py
basket:    ## brand co-purchase analysis
	python src/basket.py
charts:
	python src/charts.py
app:
	streamlit run app/dashboard.py
test:
	python -m pytest -q tests
