FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# Old: cuda:7.5-cudnn5-devel-ubuntu14.04
MAINTAINER caffe-maint@googlegroups.com

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential nano \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
	doxygen doxygen-doc doxygen-gui graphviz  && \
    rm -rf /var/lib/apt/lists/*

ENV CAFFE_ROOT=/home/mscnn/
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

RUN ls
# Install python basics
RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/kingjin94/mscnn.git . && \
    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    pip install scikit-image

# Install caffe
RUN make -j"$(nproc)" && make pycaffe -j"$(nproc)"

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

