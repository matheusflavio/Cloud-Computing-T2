FROM python:3.9-slim-bullseye

#WORKDIR /app

COPY requirements.txt /
RUN pip3 install -r requirements.txt

COPY ml_code_music.py /

CMD ["python", "ml_code_music.py"]