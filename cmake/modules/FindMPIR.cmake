
if (MPIR_INCLUDES AND MPIR_LIBRARIES)
  set(MPIR_FIND_QUIETLY TRUE)
endif (MPIR_INCLUDES AND MPIR_LIBRARIES)

find_path(MPIR_INCLUDES
  NAMES
  gmp.h
  PATHS
  $ENV{MPIRDIR}
  ${INCLUDE_INSTALL_DIR}
  ${CMAKE_SOURCE_DIR}/include/
)

find_library(MPIR_LIBRARIES mpir PATHS $ENV{MPIRDIR} ${LIB_INSTALL_DIR} ${CMAKE_SOURCE_DIR}/libs/)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPIR DEFAULT_MSG
                                  MPIR_INCLUDES MPIR_LIBRARIES)
mark_as_advanced(MPIR_INCLUDES MPIR_LIBRARIES)
