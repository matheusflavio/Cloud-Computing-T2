FROM python:3.9-slim-bullseye

WORKDIR /server

COPY requirements.txt server/requirements.txt
RUN pip3 install -r server/requirements.txt

COPY app.py server/app.py
EXPOSE 50024

CMD ["python", "./server/app.py","--host=0.0.0.0" ,"--port=50024"]