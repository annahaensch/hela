export PYTHONPATH := $(CURDIR)

RUN=$(CURDIR)/scripts/run.sh 
PYTHON_FILES = $(shell find . -not -path '*/\.*' -name '*.py')
GIT_VERSION=$(shell (git describe --tags --dirty 2>/dev/null || echo "0.0.0"))
PY_VERSION=$(shell (sed -ne "s/^__version__\s*=\s*['\"]\(.*\)['\"]/\1/p" hela/__init__.py))

.PHONY: version extensions format lint test python bash

#################################################################################
# General purpose development commands
#################################################################################

format: .docker_images/$(GIT_VERSION).stamp
	@$(RUN) yapf --style=google -pir $(PYTHON_FILES)

lint: .docker_images/$(GIT_VERSION).stamp
	@mkdir -p logs
	@$(RUN) prospector | tee logs/prospector.out

test: .docker_images/$(GIT_VERSION).stamp
	@$(RUN) python -m pytest test

isort:
	@$(RUN) isort $(PYTHON_FILES)

python: .docker_images/$(GIT_VERSION).stamp
	docker run --rm -it -v $(CURDIR):/hela hela:$(GIT_VERSION) python3

bash: .docker_images/$(GIT_VERSION).stamp
	docker run --rm -it -v $(CURDIR):/hela hela:$(GIT_VERSION) bash

#################################################################################
# Docker and build commands
#################################################################################

# Copy git tag to python __version__
python_version:
	@sed -i "s/.*__version__.*/__version__ = '$(GIT_VERSION)'/" hela/__init__.py

# Copy python_version to git tag
git_version:
ifeq (,$(shell (echo $(PY_VERSION) | grep -E '^[0-9]+.[0-9]+.[0-9]+$$')))
	@echo "Refusing to tag python version $(PY_VERSION): must match '#.#.#'"
else
	@git tag -a $(PY_VERSION)
	@git push --tags
endif

release: version .docker_images/$(GIT_VERSION).stamp
	@$(RUN) python3 setup.py sdist

extensions: version .docker_images/$(GIT_VERSION).stamp
	@$(RUN) python3 setup.py build_ext --inplace

.docker_images:
	@mkdir -p .docker_images

.docker_images/$(GIT_VERSION).stamp: Dockerfile .docker_images
	@docker build \
		-t hela:$(GIT_VERSION) \
		-t hela \
		--build-arg \
		VERSION=$(GIT_VERSION) .
	@touch $@

#################################################################################
# Commands for managing the local jupyter notebook
#################################################################################

notebooks:
	mkdir -p notebooks

## Start a jupyter server in the docker container
jupyter: .docker_images/$(GIT_VERSION).stamp notebooks
	@docker run -d --rm   \
		--name hela         \
		-p 8889:8888        \
		-v /tmp:/tmp        \
		-v $(CURDIR):/hela  \
		hela:$(GIT_VERSION)
	@docker logs hela 2>&1 | sed -n -e 's/^.*\(?token=\)/http:\/\/localhost:8889\/\1/p' | head -n 1

## Get the current local jupyter URL.
jupyter_url: .docker_images/$(GIT_VERSION).stamp
	@docker logs hela 2>&1 | sed -n -e 's/^.*\(?token=\)/http:\/\/localhost:8889\/\1/p' | head -n 1
