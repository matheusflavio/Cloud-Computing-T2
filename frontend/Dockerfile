FROM python:3.9-slim-bullseye

WORKDIR /frontend

COPY requirements.txt frontend/requirements.txt
RUN pip3 install -r frontend/requirements.txt

COPY api.py frontend/api.py
EXPOSE 50024

CMD ["python", "./frontend/api.py","--host=0.0.0.0" ,"--port=50024"]