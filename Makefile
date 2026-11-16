.PHONY: all pipeline model conversion uplift compare clv explain basket charts powerbi demo app test docker
all: pipeline model conversion clv explain basket charts powerbi demo

pipeline:    ## clean raw data, build analysis tables
	python src/pipeline.py
model:       ## next-month purchase model for buyers
	python src/churn_model.py
conversion:  ## first-purchase model for browsers
	python src/conversion_model.py
uplift:      ## measured campaign uplift (needs data_external/dunnhumby)
	python -I src/campaign_uplift.py
compare:     ## LightGBM / XGBoost / sklearn comparison
	python src/model_comparison.py
clv:         ## customer lifetime value (BG/NBD)
	python src/clv.py
explain:     ## SHAP explanations
	python src/explain.py
basket:      ## brand co-purchase analysis
	python src/basket.py
charts:
	python src/charts.py
powerbi:     ## star-schema CSVs for Power BI
	python src/powerbi_export.py
demo:        ## small sample for deploying the dashboard
	python src/make_demo_data.py
docker:
	docker build -t beauty-analytics . && docker run -p 8501:8501 beauty-analytics
app:
	streamlit run app/dashboard.py
test:
	python -m pytest -q tests
