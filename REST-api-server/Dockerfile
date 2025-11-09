FROM python:3.9-slim-bullseye

WORKDIR /server

COPY requirements.txt requirements2.txt
RUN pip3 install -r requirements2.txt

COPY app.py /
EXPOSE 50024

CMD ["python", "app.py","--host=0.0.0.0" ,"--port=50024"]