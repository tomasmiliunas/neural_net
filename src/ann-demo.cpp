#include <stdio.h>
#include "ann.h"
#include "tests.h"

int main (int c, char *v[]) {

  printf("ANN - demo\n\n");

  run_tests();

  run_cuda_sample();

  
 return 0; 
} 
