FROM ubuntu:20.04

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.8 python3-distutils python3-pip python3-apt

RUN apt-get -y install git

## Setup Java environment
# From: https://hub.docker.com/r/picoded/ubuntu-openjdk-8-jdk/dockerfile/

# This is in accordance to : https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04
RUN apt-get update && \
	apt-get install -y openjdk-17-jdk && \
	apt-get install -y ant && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/* && \
	rm -rf /var/cache/oracle-jdk17-installer;

# Fix certificate issues, found as of
# https://bugs.launchpad.net/ubuntu/+source/ca-certificates-java/+bug/983302
RUN apt-get update && \
	apt-get install -y ca-certificates-java && \
	apt-get clean && \
	update-ca-certificates -f && \
	rm -rf /var/lib/apt/lists/* && \
	rm -rf /var/cache/oracle-jdk17-installer;

# Setup JAVA_HOME, this is useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64/
RUN export JAVA_HOME


# Setup Python environment

COPY ./pyproject.toml ./CodeGraphClassification/
WORKDIR ./CodeGraphClassification


ENV POETRY_VIRTUALENVS_CREATE=false

ENV PYTHONPATH=${PYTHONPATH}:/usr/local/CodeGraphClassification/src

RUN pip3 install poetry
RUN poetry install --no-interaction --no-ansi

COPY . ./CodeGraphClassification