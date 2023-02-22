# Find EBBH
# ~~~~~~~~
#
# Once run this will define:
#
# EBBH_FOUND:	            flag if the system has the EBBH library
# EBBH_INCLUDE_DIR:	    full path to the EBBH package  header files
#

find_path(EBBH_INCLUDE_DIR
NAMES
        eBBH.hh
PATHS
        /usr/include
        /usr/local/include
        ${EBBH_PATH}/include
)

set(EBBH_FOUND FALSE)
if(EBBH_INCLUDE_DIR)
	set(EBBH_FOUND TRUE)
        #message("===============================================================================")
	message(STATUS "EBBH Found!")
	message(STATUS "setting EBBH_INCLUDE_DIR=${EBBH_INCLUDE_DIR}")
	message("===============================================================================")
else()
	#message("===============================================================================")
        message(STATUS "EBBH not found (not required)")
        message(STATUS "To use EBBH set -DEBBH_PATH=/Path/to/EBBH/install/dir on the command line")
        message("===============================================================================")

endif()


MARK_AS_ADVANCED(
EBBH_INCLUDE_DIR
EBBH_FOUND
)
