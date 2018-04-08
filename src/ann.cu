#include "ann.h"




__global__ void
kernel(int n, float *arr){

	volatile int idx = threadIdx.x + blockDim.x*blockIdx.x;
	if(idx >= n) return;

        arr[idx] *= 2.0f;

}

void run_cuda_sample(){

  int deviceCount = 0;
  checkCudaErrors( cudaGetDeviceCount(&deviceCount));
  if(deviceCount == 0){
    printf("*** there is no CUDE device\n");
    return;
  }

  checkCudaErrors( cudaSetDevice(0) );

  int n = 11; // number of elements

  float *arr = new float[n];
  for(int i = 0; i < n; i++)
    arr[i] = i;

  int h = 4; // number of threads in block
  int g = (n + (h-n%h))/h; // number of grids

  printf("n=%d, h=%d, g=%d\n", n, h, g);

    
  

  
  int bc_arr = sizeof(float)*n;

  float *dv_arr = NULL;

  checkCudaErrors( cudaMalloc((void **)&dv_arr, bc_arr) );

  checkCudaErrors( cudaMemcpy(dv_arr, arr, bc_arr, cudaMemcpyHostToDevice) ); 

  dim3 grid_dim(g, 1, 1);
  dim3 block_dim(h, 1, 1);

  kernel<<<grid_dim, block_dim>>>(n, dv_arr);



  checkCudaErrors( cudaMemcpy(arr, dv_arr, bc_arr, cudaMemcpyDeviceToHost) );

  for(int i = 0; i < n; i++)
    printf("[%d] = %f\n", i, arr[i]);

  checkCudaErrors( cudaFree(dv_arr) );
 
  checkCudaErrors(cudaDeviceReset());

}

