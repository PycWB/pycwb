#!/bin/sh

set -e  # fail on errors
set -x  # print commands as they are executed (xtrace)

# use fresh build directory
rm -rf build
mkdir -p build
cd build

# detect ICC
#if ! [ -x "$(command -v icc)" ]; then
#  echo 'Intel compiler is not installed. Reverting to gcc..' >&2
#else
#  echo 'Compiling with icc ... '
#  CMAKE_ARGS="-DCMAKE_CXX_COMPILER=icc ${CMAKE_ARGS}"
#fi
if [ -z "$1" ]; then 
   dir_name=""   
else
   dir_name="_$1"
fi 
if [ "$EUID" -ne 0 ]; then
   echo "Installing under ../../pycwb/vendor${dir_name}"
   CMAKE_ARGS="-DCMAKE_INSTALL_PREFIX:PATH=../../pycwb/vendor${dir_name} ${CMAKE_ARGS}"
  
fi
if [ ${_USE_ICC} ]; then
   # detect ICC
   if ! [ -x "$(command -v icc)" ]; then
      echo 'Intel compiler is not installed. Reverting to gcc..' >&2
   else
      echo 'Compiling with icc ... '
      CMAKE_ARGS="-DCMAKE_CXX_COMPILER=icc ${CMAKE_ARGS}"
   fi
fi


# configure
cmake .. \
    ${CMAKE_ARGS} 
   
# build
cmake --build . --verbose --parallel ${CPU_COUNT}

# install
cmake --build . --verbose --parallel ${CPU_COUNT} --target install

# return
cd ..
