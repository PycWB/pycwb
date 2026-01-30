# CI-ready image for PycWB (mirrors .gitlab-ci.yml install steps)
FROM buildpack-deps:jammy

ARG ROOT_VERSION=6.26.14
ARG ROOT_ARCH=Linux-ubuntu22-x86_64-gcc11.4
ENV ROOTSYS=/usr/local
ENV PYTHONPATH=${ROOTSYS}/lib:${PYTHONPATH}
ENV CLING_STANDARD_PCH=none

# System deps and ROOT + healpix
RUN set -eux; \
    wget -q https://root.cern/download/root_v${ROOT_VERSION}.${ROOT_ARCH}.tar.gz; \
    tar -xf root_v${ROOT_VERSION}.${ROOT_ARCH}.tar.gz --strip-components=1 -C /usr/local/; \
    apt-get update; \
    apt-get -y install git dpkg-dev cmake pkg-config g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev tar gfortran subversion; \
    apt-get -y install libfftw3-dev libcfitsio9 libsharp0 python3-pip; \
    wget -q http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb; \
    dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb; \
    wget -q https://launchpad.net/ubuntu/+archive/primary/+files/libhealpix-cxx3_3.80.0-5_amd64.deb; \
    wget -q https://launchpad.net/ubuntu/+archive/primary/+files/libhealpix-cxx-dev_3.80.0-5_amd64.deb; \
    dpkg -i libhealpix-cxx3_3.80.0-5_amd64.deb; \
    dpkg -i libhealpix-cxx-dev_3.80.0-5_amd64.deb; \
    apt-get -y install python3-venv; \
    ln -sf /usr/bin/python3 /usr/bin/python; \
    python3 -m pip install --upgrade pip setuptools; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["python3", "--version"]
