# HIT for model (de)serialization
add_library(hit SHARED
  hit/parse.cc
  hit/lex.cc
  hit/braceexpr.cc
)

# libTorch comes with two flavors: one with cxx11 abi, one without.
# We should be consistent with whatever is detected from the libTorch.
if(TORCH_CXX_FLAGS MATCHES "-D_GLIBCXX_USE_CXX11_ABI=1")
  target_compile_definitions(hit PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
else()
  target_compile_definitions(hit PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
endif()

set_target_properties(hit PROPERTIES UNITY_BUILD OFF)
target_include_directories(hit PUBLIC hit)
install(TARGETS hit LIBRARY DESTINATION neml2/lib)
