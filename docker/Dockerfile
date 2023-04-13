FROM supervisely/base-py-sdk:6.71.11
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git

RUN python3 -m pip install supervisely==6.71.11
