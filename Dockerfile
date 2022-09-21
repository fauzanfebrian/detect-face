FROM python:3.10.4-slim-buster

WORKDIR /

COPY requirements.txt .
COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "index.py"]