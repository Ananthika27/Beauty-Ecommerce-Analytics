.PHONY: all pipeline model conversion basket charts powerbi app test
all: pipeline model conversion basket charts powerbi

pipeline:    ## clean raw data, build analysis tables
	python src/pipeline.py
model:       ## next-month purchase model for buyers
	python src/churn_model.py
conversion:  ## first-purchase model for browsers
	python src/conversion_model.py
basket:      ## brand co-purchase analysis
	python src/basket.py
charts:
	python src/charts.py
powerbi:     ## star-schema CSVs for Power BI
	python src/powerbi_export.py
app:
	streamlit run app/dashboard.py
test:
	python -m pytest -q tests
