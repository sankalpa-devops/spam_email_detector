FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-serving.txt .
RUN python -m pip install --no-cache-dir -r requirements-serving.txt

COPY app.py .
COPY spam.model.joblib .
COPY templates/ ./templates/

EXPOSE 5000

CMD ["python", "app.py"]
