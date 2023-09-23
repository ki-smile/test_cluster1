FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN apt-get update -y
RUN apt-get install unzip
WORKDIR /opt

RUN mkdir test
RUN mkdir /opt/app/

RUN chmod -R 777 test


## install monai
RUN pip install monai

RUN adduser user
USER user


WORKDIR /opt/test

## Copy SegRap Code
COPY main.py .

CMD ["python","main.py"]
