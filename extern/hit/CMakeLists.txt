add_library(hit SHARED
  hit/parse.cc
  hit/lex.cc
  hit/braceexpr.cc
)

if(Torch_CXX11_ABI)
  target_compile_definitions(hit PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
else()
  target_compile_definitions(hit PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
endif()

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_include_directories(hit PUBLIC .)
target_sources(hit
  PUBLIC
  FILE_SET HEADERS
  FILES
  hit/braceexpr.h
  hit/hit.h
  hit/lex.h
  hit/parse.h
)
install(
  TARGETS hit
  FILE_SET HEADERS
  COMPONENT Development
)
