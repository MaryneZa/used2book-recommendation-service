# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose a port if using a web service (e.g., Flask or FastAPI)
EXPOSE 5000

# Command to run your application
CMD ["python", "main.py"]
