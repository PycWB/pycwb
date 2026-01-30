cmake_minimum_required(VERSION 3.15.0 FATAL_ERROR)
project(
    WAT
    LANGUAGES CXX ASM
    VERSION 7.0.1.0
    DESCRIPTION "cWB pipeline source library"
)

set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE INTERNAL "" FORCE)

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
           "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
           FORCE)
endif()

message("===============================================================================")
message("-- INFO: WAT version: ${WAT_VERSION}")
message("-- INFO: build type: ${CMAKE_BUILD_TYPE}")
message("===============================================================================")

include(GNUInstallDirs)

#Get the hostname that cmake is running on
cmake_host_system_information(RESULT HOST_NAME QUERY HOSTNAME)

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
add_compile_options(-Wno-deprecated -fexceptions -mavx)

#Enable the setting of the install RPATH
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_LIBDIR lib)
#Turn the install path to an absolute path
get_filename_component(CMAKE_INSTALL_PREFIX
        ${CMAKE_INSTALL_PREFIX}
        ABSOLUTE
        BASE_DIR ${CMAKE_BINARY_DIR})

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif("${isSystemDir}" STREQUAL "-1")


if(NOT DEFINED XIFO)
   set(XIFO 4)
endif()

# Fix the macOS cross-compilation issue, see git.ligo.org issue#4
# Only apply the `_Float16=short` workaround on Apple Silicon (arm64)
# FIXME: could not find a robust way to detect Apple Silicon when rosseta is used
set(CWB_OPTS "XIFO=${XIFO}" "_USE_CPP")
if(APPLE)
   list(APPEND CWB_OPTS "_Float16=short")
endif()

#Find ROOT
list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT CONFIG REQUIRED)
if(${ROOT_FOUND})
   include(${ROOT_USE_FILE})
   list(APPEND CWB_OPTS _USE_ROOT)
   if(${ROOT_VERSION_MAJOR}  EQUAL 6)
      list(APPEND CWB_OPTS _USE_ROOT6)
   else()
      set(ROOTCINT_OPTS "-c" "-p")
   endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(PkgConfig REQUIRED)

#Find Healpix
pkg_check_modules(HEALPIX IMPORTED_TARGET "healpix_cxx")
if(NOT HEALPIX_FOUND)
   find_package(Healpix REQUIRED)
else()
   set(HEALPIX_INCLUDE_DIR ${HEALPIX_INCLUDE_DIRS})
endif()

if(${HEALPIX_FOUND})
   if(DEFINED ${HEALPIX_SUP_DIR})
      list(APPEND CWB_INCLUDE_DIRS ${HEALPIX_INCLUDE_DIR} ${HEALPIX_SUP_DIR})
   else()
      list(APPEND CWB_INCLUDE_DIRS ${HEALPIX_INCLUDE_DIR})
   endif()
   list(APPEND CWB_OPTS _USE_HEALPIX)
endif()

#Include the definition of the function to install scripts
include(install_function)

#Add all the defintions to the global definitions
add_compile_definitions(${CWB_OPTS})


#Include the master CMakeLists.txt
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Comands to get the git repository properties
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse HEAD  OUTPUT_VARIABLE GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --short HEAD  OUTPUT_VARIABLE GIT_HASH_SHORT OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND bash -c " git -C ${CMAKE_CURRENT_SOURCE_DIR} branch | grep \\* | cut -d ' ' -f2"  OUTPUT_VARIABLE GIT_BRANCH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND  bash -c " git -C ${CMAKE_CURRENT_SOURCE_DIR} show | head | grep Date" OUTPUT_VARIABLE GIT_DATE OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} config --get remote.origin.url  OUTPUT_VARIABLE GIT_REMOTE_ORIGIN OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND git -C ${CMAKE_CURRENT_SOURCE_DIR} tag -l --points-at HEAD  OUTPUT_VARIABLE GIT_TAG OUTPUT_STRIP_TRAILING_WHITESPACE)

#Get information about build system and build time
execute_process(COMMAND uname -v OUTPUT_VARIABLE UNAME_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND uname -m OUTPUT_VARIABLE UNAME_MACH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND date -u OUTPUT_VARIABLE BTIME_UTC OUTPUT_STRIP_TRAILING_WHITESPACE)
string(TIMESTAMP CMAKE_BTIME_UTC "%Y-%m-%d %I-%M-%S" UTC)


#Configure the files from the varibles already set
configure_file(watversion.hh.in ${CMAKE_CURRENT_BINARY_DIR}/watversion.hh)
configure_file(wathash.in ${CMAKE_CURRENT_BINARY_DIR}/wathash)
configure_file(wattag.in ${CMAKE_CURRENT_BINARY_DIR}/wattag)
configure_file(watbranch.in ${CMAKE_CURRENT_BINARY_DIR}/watbranch)
configure_file(waturl.in ${CMAKE_CURRENT_BINARY_DIR}/waturl)

#Add an append of the current binary directory so the above configured files can be found and add it to the parent scope
list(APPEND CWB_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
set(CWB_INCLUDE_DIRS ${CWB_INCLUDE_DIRS} PARENT_SCOPE)


set(WAT_SRC injection.cc wavecomplex.cc Wavelet.cc WaveDWT.cc Haar.cc Biorthogonal.cc Daubechies.cc Symlet.cc Meyer.cc SymmArray.cc SymmArraySSE.cc SymmObjArray.cc WDM.cc wavearray.cc wseries.cc watplot.cc cluster.cc wavecor.cc wavefft.cc waverdc.cc lossy.cc wavelinefilter.cc netpixel.cc netcluster.cc skymap.cc detector.cc network.cc netevent.cc regression.cc time.cc monster.cc sseries.cc)

#Get all the header files
string(REGEX REPLACE "[.]cc" ".hh" WAT_HDRS "${WAT_SRC}")

#Include the header file that
if(HEALPIX_FOUND)
   list(APPEND WAT_HDRS alm.hh)
endif()

#include all the created header files
list(APPEND WAT_HDRS ${CMAKE_CURRENT_BINARY_DIR}/watversion.hh watfun.hh constants.hh wat.hh skycoord.hh watavx.hh watsse.hh WDMOverlap.hh pycwb.hh)

#Add the watasm_elf64.o to the list of src
if(APPLE)
    list(APPEND WAT_SRC watasm.S)
else()
   list(APPEND WAT_SRC watasm_elf64.o)
endif()

#Create the library
add_library(${PROJECT_NAME} SHARED ${WAT_SRC})

#Set the target properties
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "wavelet-${XIFO}x")
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

#Add all the include directories to the target
target_include_directories(${PROJECT_NAME} PRIVATE ${CWB_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} ROOT::TreePlayer ROOT::Core ROOT::Graf ROOT::FFTW ROOT::Tree ROOT::Gpad ROOT::Physics ${HEALPIX_LIBRARIES})

#Generate the root dictionary
ROOT_GENERATE_DICTIONARY(wavedict ${WAT_HDRS}
    LINKDEF wavelet_LinkDef.h
    MODULE ${PROJECT_NAME})

#Create system links to other possible names of the library
# line 73 - end: add dylib support for macos
if(APPLE)
   set(lib_ext dylib)
else()
   set(lib_ext so)
endif()

add_custom_command(
        TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ln -sf wavelet-${XIFO}x.${lib_ext} wavelet.so
        COMMAND ln -sf wavelet.so libwavelet.so
        WORKING_DIRECTORY lib
)
install(FILES ${WAT_HDRS}
        TYPE INCLUDE)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/wavelet-${XIFO}x.${lib_ext}
        ${CMAKE_CURRENT_BINARY_DIR}/lib/wavelet.so
        ${CMAKE_CURRENT_BINARY_DIR}/lib/libwavelet.so
        ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${PROJECT_NAME}_rdict.pcm
        TYPE LIB)
