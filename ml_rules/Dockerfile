FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY generator.py /app/generator.py

RUN pip3 install -r requirements.txt

CMD ["python", "generator.py"]