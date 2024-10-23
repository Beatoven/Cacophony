FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
WORKDIR /home/ubuntu

COPY requirements.txt .

RUN set -eux &&\
    apt-get update &&\
    apt-get install wget -y &&\
    # ubuntu 20.04 defaults to python 3.8.9
    apt-get install python3 -y &&\
    apt-get install python3-pip -y &&\
    apt-get install libsndfile1 -y &&\
    wget https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.7+cuda11.cudnn82-cp38-cp38-manylinux2014_x86_64.whl&&\
    sed -i 's/jaxlib.*/jaxlib-0.4.7+cuda11.cudnn82-cp38-cp38-manylinux2014_x86_64.whl/' requirements.txt
RUN pip install -r requirements.txt

#CMD ["python3", "cenascoiso.py"]