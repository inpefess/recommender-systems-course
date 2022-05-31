FROM jupyter/base-notebook:python-3.9.7
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
USER root
RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y build-essential
USER ${NB_USER}
ENV HOME /home/${NB_USER}
ENV PACKAGE_NAME rs_course
COPY ./${PACKAGE_NAME} ${HOME}/${PACKAGE_NAME}
COPY pyproject.toml ${HOME}
COPY poetry.lock ${HOME}
COPY poetry.toml ${HOME}
COPY README.rst ${HOME}
RUN pip install --no-cache-dir poetry
RUN poetry install
