# Hela HMM Toolkit

[![DOI](https://zenodo.org/badge/331356967.svg)](https://zenodo.org/badge/latestdoi/331356967)

This repository contains various code related to hidden Markov models (HMM).  For a technical overview of HMM, see this note on [Inference and Imputation for Hidden Markov Models with Hybrid State Outputs](https://annahaensch.com/papers/hmm_hybrid_em_and_inf.pdf).   

## Getting Started 

There are two ways to access the Hela codebase.  The first is using a Conda virtual environemnt, the other is by mounting a Docker image.  Instructions for both are below.

### Using Hela with Conda (_recommended_)

Before you get started, you'll need to create a new environment using `conda` (in case you need it, [installation guide here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). If you use `conda` you can create a new environment (we'll call it `hela_env`) and it's important (for backwards compatibility) that we create it specifically with Python version 3.7.3 as follows.

```
conda create --name hela_env python=3.7.3
```

and activate your new environment, with

```
conda activate hela_env
```
To run the tools in the libarary will need to install the necessary dependencies. First you'll need to conda install `pip` and then install the remaining required Python libraries as follows.

```
conda install pip
pip install -U -r requirements.txt
```
Now, to access your Jupyter server, run the following.
```
$ jupyter notebook
```
This will print a link which you can cut/paste into a web browser to access the Hela notebook directory.  You will see one folder called `tracked` which contains several tracked notebooks that will walk you through the data generation and modeling tools in hela.

### Using Hela with Docker

This code is packaged as a Docker container, so before you can interact with Hela, you'll need to do a few things: 

1. Install Docker ([available here](https://docs.docker.com/get-docker/)), or if you think you might already have Docker installed, run `docker --version` from you terminal command line.
2. Install and configure Git and ([instructions here](https://www.atlassian.com/git/tutorials/install-git)), if you think you already have Git installed, run `git --version` from your terminal command line.
3. _Recommended:_ Make sure you are prepared to connect to Github with SSH ([instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)). 

Once your working environment is prepared, navigate to the directory where you'd like to install Hela.  From there, run the following.

```
$ git clone <copy_the_link_from_Code_Clone_SSH>
```
In case it's your first time, when we write "<something_here>" in a code snippet, that means replace the angle brackets and everything between them with whatever is being suggested.  Once you've cloned the repository, navigate into the top level Hela folder.  From here you'll want to launch a Jupyter server in the Docker container (this step will also mount a Docker image in the background and might take a few moments).

```
$ cd hela
$ make jupyter
```

Now, to access your Jupyter server, run the following.
```
$ make jupyter_url
```
This will print a link which you can cut/paste into a web browser to access the Hela notebook directory.  You will see one folder called `tracked` which contains several tracked notebooks that will walk you through the data generation and modeling tools in hela.


## Working with Hela

There are several tracked notebooks to help get you started in `notebooks\tracked`.

## Unit Tests

Before pushing any code you should run the unit tests.  You can do this from the top level directory with

```
$ pytest
```
If you get any errors that means your code has broken Hela and you should figure out why.  Don't worry if you get warnings, that's totally fine.

## Contact

If you have questions or comments not suited for the Github workflow, please reach out to anna.haensch@tufts.edu.
