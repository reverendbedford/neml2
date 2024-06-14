# ----------------------------------------------------------------------------
# Macros for registering unity groups
# ----------------------------------------------------------------------------
macro(_src_subdirs result curdir)
      file(GLOB_RECURSE children ${curdir}/*.cxx)

      foreach(child ${children})
            get_filename_component(child_dir ${child} DIRECTORY)
            list(APPEND dirlist ${child_dir})
      endforeach()

      list(REMOVE_DUPLICATES dirlist)
      set(${result} ${dirlist})
endmacro()

macro(_set_unity_group root subdir)
      file(RELATIVE_PATH UNITY_NAME_RAW ${root} ${subdir})

      if(NOT UNITY_NAME_RAW STREQUAL "")
            string(REGEX REPLACE "/" "_" UNITY_NAME ${UNITY_NAME_RAW})

            if(NOT UNITY_NAME MATCHES "CMakeFiles")
                  if(NOT UNITY_NAME MATCHES "CMakeFiles")
                        file(GLOB_RECURSE UNITY_FILES ${subdir}/*.cxx)
                        set_source_files_properties(${UNITY_FILES} PROPERTIES UNITY_GROUP ${UNITY_NAME})
                  endif()
            endif()
      endif()
endmacro()

macro(register_unity_group target rel_root)
      get_target_property(UNITY_ENABLED ${target} UNITY_BUILD)

      if(UNITY_ENABLED)
            get_filename_component(ABS_ROOT ${rel_root} ABSOLUTE)
            _src_subdirs(SUBDIRS ${ABS_ROOT})

            foreach(subdir ${SUBDIRS})
                  _set_unity_group(${ABS_ROOT} ${subdir})
            endforeach()

            set_target_properties(${target} PROPERTIES UNITY_BUILD_MODE GROUP)
      endif()
endmacro()
