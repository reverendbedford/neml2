# HIT for model (de)serialization
add_library(hit SHARED
  hit/parse.cc
  hit/lex.cc
  hit/braceexpr.cc
)
set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_include_directories(hit PUBLIC hit)
