FROM python:3.9-slim-bullseye

#WORKDIR /recommend-rules

COPY requirements.txt /
RUN pip3 install -r requirements.txt

COPY generator.py /

RUN ls -la 

CMD ["python", "generator.py"]