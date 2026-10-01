FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data/sample ./data/sample
RUN pip install --no-cache-dir .
ENV COMMERCE_DATA_DIR=/app/data/sample
EXPOSE 8008
CMD ["uvicorn", "commerce_ai.api:app", "--host", "0.0.0.0", "--port", "8008"]

