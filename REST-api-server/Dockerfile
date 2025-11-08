FROM python:3.9-slim-bullseye

WORKDIR /rest-api-server

COPY requirements.txt rest-api-server/requirements.txt
RUN pip3 install -r rest-api-server/requirements.txt

COPY app.py api.py
EXPOSE 50024

CMD ["python", "app.py","--host=0.0.0.0" ,"--port=50024"]