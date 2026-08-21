FROM ubuntu:20.04

# Install base utilities
RUN apt-get update \
    && apt install -y software-properties-common \
    && apt update \
    && add-apt-repository ppa:linuxuprising/libpng12 \
    && apt-get install -y --no-install-recommends \
    	build-essential wget git cmake cmake-curses-gui python python3-pip libtbb-dev \
    	libboost-all-dev libeigen3-dev zlib1g-dev libncurses5-dev libgdbm-dev \
    	libnss3-dev libssl-dev libreadline-dev libffi-dev zram-config ca-certificates \
    	libglu1 libcurl4-openssl-dev libsm6 libxt6 libfreetype6 libxrender1 libfontconfig1 \
    	libglib2.0-0 python3-pyqt5 libgtk2.0-dev libpng12-0 libspdlog-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
   
# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py311_24.3.0-0-Linux-x86_64.sh -O ~/miniconda.sh \ 
    && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Add conda to PATH
ENV PATH=$CONDA_DIR/bin:$PATH

# Install MRtrix 3.0.8
RUN conda install mrtrix3::mrtrix3==3.0.8

# Install ITK-SNAP 3.8.0
RUN wget -O itksnap.tar.gz 'https://sourceforge.net/projects/itk-snap/files/itk-snap/3.8.0/itksnap-3.8.0-20190612-Linux-x86_64-qt4.tar.gz/download' \
    && tar -zxf itksnap.tar.gz -C /opt/ \
    && mv /opt/itksnap-*/ /opt/itksnap/ \
    && rm itksnap.tar.gz
    
# Add ITK-SNAP to PATH
ENV PATH="/opt/itksnap/bin/:${PATH}" 
ENV LD_LIBRARY_PATH=/opt/itksnap/lib/:${LD_LIBRARY_PATH} 

# Install SVR-lite 1.1
COPY ./src/svr-lite/ /home/svr-lite/
RUN tar -xzvf /home/svr-lite/svr-lite_1.1.0-ubuntu20.04-x86_64.tar.gz -C /home/svr-lite

# Add SVR-lite to PATH
ENV PATH="/home/svr-lite/bin:${PATH}"
ENV LD_LIBRARY_PATH="/home/svr-lite/lib:${LD_LIBRARY_PATH}"

# Set a directory for the app
WORKDIR /usr/src/app

# Make the image runnable as an arbitrary non-root UID
RUN mkdir -p /mnt && chmod -R 777 /mnt
ENV HOME=/home/user
RUN mkdir -p /home/user && chmod -R 777 /home/user
ENV TMPDIR=/home/user/tmp
RUN mkdir -p /home/user/tmp && chmod 777 /home/user/tmp
ENV MPLCONFIGDIR=/home/user/tmp/matplotlib
RUN mkdir -p /home/user/.itksnap.org/ITK-SNAP && chmod -R 777 /home/user/.itksnap.org/ITK-SNAP
COPY ./src/sparc/cfg/UserPreferences.xml /home/user/.itksnap.org/ITK-SNAP/UserPreferences.xml

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy SPARC python package tree
COPY ./src ./src

# Download DL models from Hugging Face
ARG MODEL_VARIANTS="philips joint siemens siemens_transfer"
RUN set -e; \
    for pair in \
      "chest_segmentation:sparc-chest-segmentation" \
      "heart_segmentation:sparc-heart-segmentation" \
      "reorientation:sparc-reorientation" \
    ; do \
      component="${pair%%:*}"; \
      repo="${pair#*:}"; \
      for variant in ${MODEL_VARIANTS}; do \
        hf download "aboutill/${repo}" \
          --repo-type model \
          --revision "${variant}" \
          --local-dir "./src/sparc/pipeline/${component}/models/${variant}" \
          --include '*.pth' ; \
      done; \
    done
    
# Install SPARC python package
COPY pyproject.toml pyproject.toml
RUN pip install -e .

# Entry point workdir
WORKDIR /mnt/workdir

# Expose ports 8888/6006 for Jupyter/Tensorboard server forwarding
EXPOSE 8888
EXPOSE 6006

LABEL org.opencontainers.image.source="https://github.com/aboutill/sparc"
LABEL org.opencontainers.image.version="1.0.0"
