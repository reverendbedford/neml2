# HIT for model (de)serialization
add_library(hit SHARED
  hit/parse.cc
  hit/lex.cc
  hit/braceexpr.cc
)
target_include_directories(hit PUBLIC .)
