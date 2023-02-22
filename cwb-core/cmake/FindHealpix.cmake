# Find HealPix
# ~~~~~~~~
#
# Once run this will define:
#
# HEALPIX_FOUND:	    healpixsystem has HEALPIX lib
# HEALPIX_INCLUDE_DIR:	    full path to the HEALPIX package libraries
# HEALPIX_SUP_DIR:	    full path to the HEALPIX cxx support libraries
#

find_path(HEALPIX_INCLUDE_DIR
NAMES
        healpix_base.h
PATHS
        ${HEALPIX_PATH}/include
        ${HEALPIX_PATH}/include/healpix_cxx
        ${HEALPIX_CXX_DIR}/Healpix_cxx
        /user/local/include/healpix_cxx
        /usr/include/healpix_cxx
        /usr/local/include
        /usr/include
)
find_path(HEALPIX_SUP_DIR
NAMES
        alm.h
PATHS
        ${HEALPIX_PATH}/include
        ${HEALPIX_PATH}/include/healpix_cxx
        ${HEALPIX_CXX_DIR}/cxxsupport
        /usr/local/include/healpix_cxx
        /usr/include/healpix_cxx
        /usr/local/include
        /usr/include
)

find_library(HEALPIX_LIBRARY
NAMES
        healpix_cxx
PATHS
        ${HEALPIX_PATH}/lib
        /usr/local/lib
        /usr/lib
        /usr/local/lib64
        /usr/lib64
)


set(HEALPIX_FOUND FALSE)
if(HEALPIX_INCLUDE_DIR AND HEALPIX_SUP_DIR)
	set(HEALPIX_FOUND TRUE)
	message("===============================================================================")
	message(STATUS "HealPix Found!")
	message(STATUS "Setting HEALPIX_INCLUDE_DIR=${HEALPIX_INCLUDE_DIR}")
        message(STATUS " and HEALPIX_LIBRARY=${HEALPIX_LIBRARY}")
	message("===============================================================================")
else()
	message("===============================================================================")
    message(FATAL_ERROR "HealPix not found  \
    To use HealPix set either -DHEALPIX_PATH=/Path/to/Healpix/install/dir or -DHEALPIX_CXX_DIR=/Path/to/Healpix/cxx on the command line")
	message("===============================================================================")
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Healpix  DEFAULT_MSG
                                  HEALPIX_LIBRARY HEALPIX_INCLUDE_DIR HEALPIX_SUP_DIR)



mark_as_advanced(
HEALPIX_INCLUDE_DIR
HEALPIX_SUP_DIR
HEALPIX_FOUND
)
