FROM python:3.9-slim

WORKDIR /app

# Install libgomp1 for implicit
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Move recommendation_service.pkl to /data if it exists, then remove from /app
RUN mkdir -p /data && \
    [ -f recommendation_service.pkl ] && mv recommendation_service.pkl /data/recommendation_service.pkl || true && \
    rm -f /app/recommendation_service.pkl

CMD ["python", "app.py"]