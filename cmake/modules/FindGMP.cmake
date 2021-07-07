
if (GMP_INCLUDES AND GMP_LIBRARIES)
  set(GMP_FIND_QUIETLY TRUE)
endif (GMP_INCLUDES AND GMP_LIBRARIES)

find_path(GMP_INCLUDES
  NAMES
  gmp.h
  gmpxx.h
  PATHS
  $ENV{GMPDIR}
  ${INCLUDE_INSTALL_DIR}
  ${CMAKE_SOURCE_DIR}/include/
)

find_library(GMP_LIBRARIES gmp gmpxx PATHS $ENV{GMPDIR} ${LIB_INSTALL_DIR} ${CMAKE_SOURCE_DIR}/libs/)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GMP DEFAULT_MSG
                                  GMP_INCLUDES GMP_LIBRARIES)
mark_as_advanced(GMP_INCLUDES GMP_LIBRARIES)
