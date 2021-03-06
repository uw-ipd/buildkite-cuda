FROM nvidia/cuda:9.1-runtime-ubuntu16.04

# Installation steps from https://hub.docker.com/r/continuumio/miniconda3/~/dockerfile/
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Updated to miniconda 4.5.1
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Inlined tini version in fetch url
ENV TINI_VERSION v0.16.1
RUN wget --quiet -O /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.16.1/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

ADD environment.yml /environment.yml
RUN conda env update -n base -f /environment.yml && conda clean -tipsy
