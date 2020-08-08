ARG platform=cpu

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.5.1-cpu-py36-ubuntu16.04
# FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.5.1-gpu-py36-cu101-ubuntu16.04

# RUN if [ "x$arg" = "x" ] ; then echo Argument not provided ; else echo Argument is $arg ; fi
RUN ["mkdir", "-p", "/opt/ml/output"]
RUN ["mkdir", "-p", "/opt/ml/input"]


RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /opt/ml/code
RUN cd $WORKDIR
COPY . .

ENV SAGEMAKER_PROGRAM sagemaker.py

ENTRYPOINT python -m sagemaker