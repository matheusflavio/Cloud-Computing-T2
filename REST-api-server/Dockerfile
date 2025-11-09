FROM python:3.9-slim-bullseye

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

RUN pip3 install -r requirements.txt

EXPOSE 50024

CMD ["python", "app.py","--host=0.0.0.0" ,"--port=50024"]