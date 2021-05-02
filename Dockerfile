# a Dockerfile is a build spec for a Docker image
FROM continuumio/anaconda3:2020.11

ADD . /code 
WORKDIR /code

ENTRYPOINT ["python", "result_app.py"]