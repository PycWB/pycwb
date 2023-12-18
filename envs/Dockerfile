FROM buildpack-deps:jammy

# Install ROOT
RUN wget -q https://root.cern/download/root_v6.26.04.Linux-ubuntu22-x86_64-gcc11.2.tar.gz
RUN tar -xf root_v6.26.04.Linux-ubuntu22-x86_64-gcc11.2.tar.gz --strip-components=1 -C /usr/local/

# Set environment variables
ENV ROOTSYS /usr/local
ENV PYTHONPATH $ROOTSYS/lib:$PYTHONPATH
ENV CLING_STANDARD_PCH none

# Install dependencies
RUN apt-get update
RUN apt-get -y install git dpkg-dev cmake pkg-config g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev tar gfortran subversion
RUN apt-get -y install libfftw3-dev libcfitsio9 libsharp0 python3-pip
RUN wget -q http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.1f-1ubuntu2_amd64.deb
RUN dpkg -i libssl1.1_1.1.1f-1ubuntu2_amd64.deb
RUN wget -q https://launchpad.net/ubuntu/+archive/primary/+files/libhealpix-cxx3_3.80.0-5_amd64.deb
RUN wget -q https://launchpad.net/ubuntu/+archive/primary/+files/libhealpix-cxx-dev_3.80.0-5_amd64.deb
RUN dpkg -i libhealpix-cxx3_3.80.0-5_amd64.deb
RUN dpkg -i libhealpix-cxx-dev_3.80.0-5_amd64.deb

# Install pycwb
RUN git clone https://git.ligo.org/yumeng.xu/pycwb.git
RUN apt-get -y install python3-pip python3-venv
RUN python3 -m pip install pip setuptools -U
RUN python3 -m pip install build
RUN cd pycwb && python3 -m build --sdist
RUN cd pycwb && pip install dist/*.tar.gz

# Test pycwb
CMD python3 -c "import pycwb; print(pycwb.__version__)"