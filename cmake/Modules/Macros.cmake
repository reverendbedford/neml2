# ###################################################
# MACRO to find all source subdirectories
# ###################################################
MACRO(SUBDIRLIST result curdir)
      FILE(GLOB_RECURSE children ${curdir}/*.cxx)
      SET(dirlist "")

      FOREACH(child ${children})
            get_filename_component(child_dir ${child} DIRECTORY)
            LIST(APPEND dirlist ${child_dir})
      ENDFOREACH()

      list(REMOVE_DUPLICATES dirlist)
      SET(${result} ${dirlist})
ENDMACRO()

MACRO(SETUNITYGROUP name root subdir)
      file(RELATIVE_PATH UNITY_NAME_RAW ${root} ${subdir})

      if(NOT UNITY_NAME_RAW STREQUAL "")
            string(REGEX REPLACE "/" "_" UNITY_NAME ${UNITY_NAME_RAW})

            if(NOT UNITY_NAME MATCHES "CMakeFiles")
                  if(NOT UNITY_NAME MATCHES "CMakeFiles")
                        file(GLOB_RECURSE UNITY_FILES ${subdir}/*.cxx)
                        set_source_files_properties(${UNITY_FILES} PROPERTIES UNITY_GROUP ${UNITY_NAME})
                        message(STATUS "${name}: group unity for ${UNITY_NAME}")
                  endif()
            endif()
      endif()
ENDMACRO()
