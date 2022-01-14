FROM python:3.6
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENCV_VERSION 3.4.16
ARG N_CORES_FOR_OPENCV_BUILD 8

RUN apt-get update && apt-get install -y \
    build-essential unzip cmake pkg-config libopencv-dev \
    curl git libatlas-base-dev gfortran \
    libgtk2.0-dev libavcodec-dev libavformat-dev \
    libswscale-dev libjpeg-dev libpng-dev libtiff-dev libv4l-dev \
    python3-dev python3-numpy python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /home/pyhjs

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install cmake numpy

# Avoiding below error on setup.py installing process
# """
# Beginning with Matplotlib 3.4, Python 3.7 or above is required.
# You are using Python 3.6.15.
# """
RUN pip install 'matplotlib<3.0'

RUN curl -fsSL https://github.com/opencv/opencv/archive/refs/tags/$OPENCV_VERSION.tar.gz | tar xz && mkdir opencv-$OPENCV_VERSION/build
WORKDIR /home/pyhjs/opencv-$OPENCV_VERSION/build
RUN cmake \
        -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
        -D CMAKE_BUILD_TYPE=Release \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_python3=OFF \
        .. && \
    make -j${N_CORES_FOR_OPENCV_BUILD} && make install && ldconfig && \
    cd ../.. && rm -rf opencv-$OPENCV_VERSION

WORKDIR /home/pyhjs
COPY . /home/pyhjs

RUN python setup.py install