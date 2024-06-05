FROM supervisely/base-py-sdk:6.72.111
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN pip3 install cachetools

RUN python3 -m pip install supervisely==6.73.96

RUN mkdir -p weights
COPY ViT-B_SAM.pth /weights/
COPY ViT-L_SAM.pth /weights/
COPY ViT-H_SAM.pth /weights/
