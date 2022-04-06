# Python version 3.7.3
FROM jupyter/datascience-notebook:5ed91e8e3249
USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential dvipng

RUN pip install \
  dask==2.11.0 \
  distributed==2.11.0 \
  isort==4.3.21 \
  jax==0.1.76 \
  jaxlib==0.1.60 \
  matplotlib==3.1.2 \
  numpy==1.19 \
  pandas==1.1.4 \
  pgmpy==0.1.18 \
  pytest==5.3.2 \
  tornado==6.0.3 \
  yapf==0.20.1

WORKDIR /hela
RUN mkdir /hela/notebooks

# Install a copy of the library (so we can use the image without mounting)
COPY . /tmp/hela
RUN pip install /tmp/hela && rm -rf /tmp/hela

RUN usermod -l hela jovyan
USER hela

ENV PYTHONPATH /hela
CMD jupyter notebook --notebook-dir=/hela/notebooks
