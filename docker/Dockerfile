FROM supervisely/base-py-sdk:6.72.111
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install git+https://github.com/facebookresearch/segment-anything.git
RUN pip3 install cachetools

RUN python3 -m pip install supervisely==6.73.104

RUN mkdir -p weights

COPY --from=supervisely/segment-anything:1.0.1 /weights/ViT-B_SAM.pth /weights/ViT-B_SAM.pth
COPY --from=supervisely/segment-anything:1.0.1 /weights/ViT-L_SAM.pth /weights/ViT-L_SAM.pth
COPY --from=supervisely/segment-anything:1.0.1 /weights/ViT-H_SAM.pth /weights/ViT-H_SAM.pth
