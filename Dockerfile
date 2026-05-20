# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.11-slim

# Prevents Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensures logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Upgrade build tools
RUN pip install --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install -r requirements.txt

# Install spaCy model directly (more reliable than "spacy download")
RUN pip install \
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application code
COPY . .

# Create non-root user
RUN adduser --uid 5678 --disabled-password --gecos "" appuser \
    && chown -R appuser /app

USER appuser

EXPOSE 8000

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8000", "--server.address=0.0.0.0"]