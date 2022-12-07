# HIT for model (de)serialization
add_library(hit
  parse.cc
  lex.cc
  braceexpr.cc
)
target_include_directories(hit PUBLIC .)
