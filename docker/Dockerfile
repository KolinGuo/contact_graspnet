ARG OS_VERSION=22.04
ARG CUDA_VERSION=11.8.0

#FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
# docker inspect --format='{{index .RepoDigests 0}}' nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda@sha256:8f9dd0d09d3ad3900357a1cf7f887888b5b74056636cd6ef03c160c3cd4b1d95

# Run apt-get (dkpg) without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

########################################
# SECTION 1: Essentials                #
########################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim git curl wget yasm cmake unzip pkg-config \
    checkinstall build-essential ca-certificates \
    software-properties-common apt-utils bash-completion \
    # Image libraries (for OpenCV GUI) \
    graphviz \
    # OpenGL/glvnd libraries \
    libxau6 libxdmcp6 libxcb1 libxext6 libx11-6 \
    libglvnd0 libgl1 libglx0 libegl1 libgles2 \
    # Setup locale language config \
    locales \
  && apt-get upgrade -y \
  && rm -rf /var/lib/apt/lists/* \
  # Setup locale language config \
  && locale-gen "en_US.UTF-8" \
  && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
  # Set timezone \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8


#######################################
# SECTION 2: Install Python           #
#######################################
# Default python with up-to-date pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m pip install --upgrade pip


##################################################
# SECTION 3: Install Contact-GraspNet package    #
##################################################
ARG CGN_PKG_DIR="."
COPY "${CGN_PKG_DIR}" /tmp
RUN python3 -m pip install /tmp && rm -r /tmp/*


########################################
# SECTION N: Additional config & MISC  #
########################################
# Turn off open3d jupyter WebVisualizer
RUN sed -i 's/"BUILD_JUPYTER_EXTENSION" : True/"BUILD_JUPYTER_EXTENSION" : False/' \
  /usr/local/lib/python3.10/dist-packages/open3d/_build_config.py

# Set working directory to be repository directory
ARG WORKDIR="/workspace"
WORKDIR ${WORKDIR}

ENTRYPOINT []
CMD ["/bin/bash"]
