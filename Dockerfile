FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY data ./data
ARG INSTALL_EXTRAS=""
RUN pip install --no-cache-dir ".${INSTALL_EXTRAS}"
ENV COMMERCE_DATA_DIR=/app/data/sample
EXPOSE 8008
CMD ["uvicorn", "commerce_ai.api:app", "--host", "0.0.0.0", "--port", "8008"]

