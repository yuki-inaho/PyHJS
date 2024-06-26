FROM python:3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential unzip cmake pkg-config libopencv-dev \
    python3-dev python3-numpy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install cython

RUN ln -sf /usr/lib/x86_64-linux-gnu/pkgconfig/opencv4.pc /usr/lib/x86_64-linux-gnu/pkgconfig/opencv.pc
ENV PKG_CONFIG_PATH=/usr/share/x86_64-linux-gnu/pkgconfig:/usr/share/pkgconfig:$PKG_CONFIG_PATH
ENV PKG_CONFIG_LIBDIR=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_LIBDIR

WORKDIR /home/pyhjs
COPY . /home/pyhjs

RUN pip install -r requirements.txt
RUN python setup.py install
