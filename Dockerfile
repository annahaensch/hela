# Python version 3.7.3
FROM jupyter/datascience-notebook:5ed91e8e3249
USER root

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential dvipng

RUN pip install \
  altair==3.2.0 \
  altair_saver==0.1.0 \
  argparse==1.4.0 \
  blosc==1.8.3 \
  bokeh==1.1.0 \
  boto3==1.10.50 \
  cloudpickle==1.3.0 \
  CMake==3.18.4 \
  coverage==5.0 \
  cython==0.29.6 \
  dask-kubernetes==0.10.1 \
  dask==2.11.0 \
  distributed==2.11.0 \
  feather-format==0.4.0 \
  gast==0.3.3 \
  gcsfs==0.6.1 \
  hdbscan==0.8.24 \
  hiplot==0.1.7.post2 \
  isort==4.3.21 \
  jax==0.1.76 \
  jaxlib==0.1.55 \
  jupyter_contrib_nbextensions==0.5.1 \
  lz4==3.0.2 \
  matplotlib==3.1.2 \
  memory_profiler==0.58.0 \
  msgpack==1.0.0 \
  numpy==1.18.1 \
  openpyxl==3.0.1 \
  pandas==1.1.4 \
  prospector==1.3.1 \
  pysurvival==0.1.2 \
  pytest==5.3.2 \
  python-forecastio==1.4.0 \
  PyWavelets==1.1.1 \
  s3fs==0.4.2 \
  scikit-learn==0.23.0 \
  scikit-survival==0.14.0 \
  scipy==1.4.1 \
  sklearn-pandas==1.8.0 \
  tensorflow-probability==0.11.1 \
  tensorflow==2.3.0 \
  tornado==6.0.3 \
  tqdm==4.40.2 \
  tsfresh==0.16.0 \
  urllib3==1.25.10 \
  vega==2.6.0 \
  xarray==0.16.1 \
  yapf==0.20.1

WORKDIR /hela
RUN mkdir /hela/notebooks

# Install a copy of the library (so we can use the image without mounting)
COPY . /tmp/hela
RUN pip install /tmp/hela && rm -rf /tmp/hela

RUN usermod -l tagup jovyan
USER tagup

ENV PYTHONPATH /hela
CMD jupyter notebook --notebook-dir=/hela/notebooks
