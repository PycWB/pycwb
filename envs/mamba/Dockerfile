FROM quay.io/pypa/manylinux_2_28_x86_64
RUN curl -L micro.mamba.pm/install.sh > install.sh
RUN ["bash", "./install.sh"]
RUN ["source", "~/.bashrc"]
RUN mamba install -c conda-forge -y -q healpix_cxx=3.81 root=6.26.10 cmake pkg-config
RUN mamba install -c conda-forge -y -q nds2-client python-nds2-client
RUN pip install cibuildwheel
RUN cibuildwheel --platform linux