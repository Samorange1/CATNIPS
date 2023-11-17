FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip nano sudo
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US/Michigan
RUN apt-get install -y python3-tk
WORKDIR /home/madhavr/ 
COPY requirements.txt /tmp/requirements.txt

# COPY catnips_implementation /home/madhavr/catnips
RUN python3 -m pip install -r /tmp/requirements.txt
