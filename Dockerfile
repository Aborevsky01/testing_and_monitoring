FROM python:3.12-slim

RUN apt-get update && apt-get install -y git

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY . /app

EXPOSE 8890

CMD uvicorn main:app --host 0.0.0.0 --port 8890
