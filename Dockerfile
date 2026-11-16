FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY app ./app
COPY demo_data ./demo_data
COPY tests ./tests

# Dashboard runs on the bundled demo sample. To use the full results, mount them:
#   docker run -p 8501:8501 -v "$PWD/outputs:/app/outputs" beauty-analytics
EXPOSE 8501
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
