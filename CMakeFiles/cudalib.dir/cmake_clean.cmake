file(REMOVE_RECURSE
  "libcudalib.pdb"
  "libcudalib.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cudalib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
