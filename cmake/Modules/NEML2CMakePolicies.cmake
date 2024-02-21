# FindPython should return the first matching Python
if(POLICY CMP0094)
  cmake_policy(SET CMP0094 NEW)
endif()

# Suppress the warning related to the new policy on fetch content's timestamp
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Suppress the warning related to the new policy on FindPythonXXX
if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()
