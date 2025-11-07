# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
# Note: --trusted-host flags are used for CI/CD environments with SSL certificate issues
# For production, use proper SSL certificates or configure pip with trusted certificate authority
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY model/ ./model/

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "src/app.py"]
