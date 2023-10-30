# ###################################################
# MACROs for installing test artifacts
# ###################################################
macro(_find_test_artifacts result base_dir)
  file(GLOB_RECURSE ${result}
    RELATIVE ${base_dir}
    ${base_dir}/*.i
    ${base_dir}/*.pt
    ${base_dir}/*.txt
    ${base_dir}/*.vtest
    ${base_dir}/*.xml
    ${base_dir}/*.py
  )
endmacro()

macro(install_test_artifacts test_dir)
  _find_test_artifacts(ARTIFACTS ${NEML2_SOURCE_DIR}/tests/${test_dir})

  if(EXISTS "${NEML2_SOURCE_DIR}/.git")
    message(STATUS "Creating symbolic links for ${test_dir} testing artifacts")

    foreach(ARTIFACT ${ARTIFACTS})
      cmake_path(GET ARTIFACT PARENT_PATH ARTIFACT_DIR)
      file(MAKE_DIRECTORY ${NEML2_BINARY_DIR}/tests/${test_dir}/${ARTIFACT_DIR})
      file(CREATE_LINK ${NEML2_SOURCE_DIR}/tests/${test_dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/tests/${test_dir}/${ARTIFACT} SYMBOLIC)
    endforeach()
  else()
    message(STATUS "Copying unit test artifacts")

    foreach(ARTIFACT ${ARTIFACTS})
      file(COPY_FILE ${NEML2_SOURCE_DIR}/tests/${test_dir}/${ARTIFACT} ${NEML2_BINARY_DIR}/tests/${test_dir}/${ARTIFACT})
    endforeach()
  endif()
endmacro()
