# Container image for the edge inference module (for deployment as an Azure IoT Edge module).
FROM python:3.12-slim
WORKDIR /app
COPY requirements-edge.txt .
RUN pip install --no-cache-dir -r requirements-edge.txt
COPY config.py storage.py features.py edge_module.py energy_model.pkl ./
ENV EDGE_HOST=0.0.0.0 EDGE_PORT=5000 DB_PATH=/app/data/predictions.db
EXPOSE 5000
CMD ["python", "edge_module.py"]
