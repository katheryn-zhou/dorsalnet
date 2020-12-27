FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN git clone https://github.com/facebookresearch/slowfast; cd SlowFast; python setup.py build develop

RUN apt-get update && apt-get install -y wget unzip