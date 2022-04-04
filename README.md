# Hela HMM Toolkit

This repository contains various code related to hidden Markov models (HMM).  For a technical overview of HMM, see this note on [Inference and Imputation for Hidden Markov Models with Hybrid State Outputs](This repository contains code for hidden Markov modeling).   

## Getting Started 

This code is packaged as a docker container, so before you can interact with Hela, you'll need to do a few things: 

1. Install Docker ([available here](https://docs.docker.com/get-docker/)), or if you think you might already have in installed, run `docker --version` from you terminal command line.
2. Install and configure Git and ([instructions here](https://www.atlassian.com/git/tutorials/install-git)), if if you thin you already have in installed, run `git --version` from your terminal command line.
3. _Recommended:_ Make sure you are prepared to connect to Github with SSH ([instructions here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)). 

Once your working environment is prepared, navigate to the directory where you'd like to install hela.  From there, run the following.

```
$ git clone <copy_the_link_from_Code_Clone_SSH>
```
In case it's your first time, when we write "<something_here>" in a code snippet, that means replace the angle brackets and everything between them with whatever is being suggested.  Once you've cloned the repository, navigate into the top level hela folder and launch a jupyter server in the Docker container (this step will also mount a docker image in the background and might take a few moments).

```
$ cd hela
$ make jupyter
``

Now, to access your Jupyter server, run the following.
```
$ make jupyter_url
```
This will print a link which you can cut/paste into a web browser to access the hela notebook drive.


## Working with Hela

### Generative Modeling with Hela

### Inference and Learning with Hela

## Contact

If you have questions or comments not suited for the Github workflow, please reach out to anna.haensch@tufts.edu.
