# ###################################################
# MACROs for installing artifacts
# ###################################################
macro(_find_artifacts result base_dir)
  file(GLOB_RECURSE ${result}
    RELATIVE ${base_dir}
    ${base_dir}/*.i
    ${base_dir}/*.pt
    ${base_dir}/*.txt
    ${base_dir}/*.vtest
    ${base_dir}/*.xml
    ${base_dir}/*.py
    ${base_dir}/*.pyi
  )
endmacro()

macro(build_artifacts dir symlink)
  _find_artifacts(ARTIFACTS ${NEML2_SOURCE_DIR}/${dir})

  if(${symlink} AND EXISTS "${NEML2_SOURCE_DIR}/.git")
    set(SYMLINK_ARTIFACT ON)
    message(STATUS "Creating symbolic links for artifacts ${dir}")
  else()
    set(SYMLINK_ARTIFACT OFF)
    message(STATUS "Copying artifacts ${dir}")
  endif()

  if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
    foreach(ARTIFACT ${ARTIFACTS})
      cmake_path(GET ARTIFACT PARENT_PATH ARTIFACT_DIR)
      file(MAKE_DIRECTORY ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT_DIR})

      if(SYMLINK_ARTIFACT)
        file(CREATE_LINK ${NEML2_SOURCE_DIR}/${dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT} SYMBOLIC)
      else()
        file(COPY_FILE ${NEML2_SOURCE_DIR}/${dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT})
      endif()
    endforeach()
  endif()
endmacro()

macro(install_artifacts dir idir symlink)
  _find_artifacts(ARTIFACTS ${NEML2_SOURCE_DIR}/${dir})

  if(${symlink} AND EXISTS "${NEML2_SOURCE_DIR}/.git")
    set(SYMLINK_ARTIFACT ON)
    message(STATUS "Creating symbolic links for artifacts ${dir}")
  else()
    set(SYMLINK_ARTIFACT OFF)
    message(STATUS "Copying artifacts ${dir}")
  endif()

  if(NOT ${NEML2_BINARY_DIR} STREQUAL ${NEML2_SOURCE_DIR})
    foreach(ARTIFACT ${ARTIFACTS})
      cmake_path(GET ARTIFACT PARENT_PATH ARTIFACT_DIR)
      file(MAKE_DIRECTORY ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT_DIR})

      if(SYMLINK_ARTIFACT)
        file(CREATE_LINK ${NEML2_SOURCE_DIR}/${dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT} SYMBOLIC)
      else()
        file(COPY_FILE ${NEML2_SOURCE_DIR}/${dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/${dir}/${ARTIFACT})
      endif()
    endforeach()
  endif()

  foreach(ARTIFACT ${ARTIFACTS})
    set(ARTIFACT_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/neml2/${idir}/${ARTIFACT})
    cmake_path(GET ARTIFACT_INSTALL_PATH PARENT_PATH ARTIFACT_INSTALL_DIR)
    install(FILES ${NEML2_SOURCE_DIR}/${dir}/${ARTIFACT} DESTINATION ${ARTIFACT_INSTALL_DIR})
  endforeach()
endmacro()
