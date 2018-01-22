//
// cooperative_group
//   -> nvcc ... -rdc = true  and require TCC mode
//
#include <cstdio>
#include <cstdlib>
#include <cuda_profiler_api.h>

#include "header/my_gettime.hpp"
#include "header/my_cuda_device.cuh"
#include "header/my_cuda_host.cuh"
#include "header/fp16_dev.h"

#define NUMTHREADS 128

__global__ void kernel(const int* __restrict__ GIN, int *GOUT){
	int laneId = threadIdx.x & 0x1F;
	int warpId = threadIdx.x >> 5;
	int tx = blockDim.x*blockIdx.x+threadIdx.x;

	__shared__ int s_mem[NUMTHREADS];
	s_mem[threadIdx.x]=GIN[tx];
	__syncthreads();
	int val = s_mem[threadIdx.x];
	val = warp_scan<32>(val,laneId);
	val = warp_sum(val,laneId);
	val = block_scan<int, NUMTHREADS>(val,warpId,laneId);
	val = block_sum <int, NUMTHREADS>(val,warpId,laneId);
	GOUT[tx]=val;



	__threadfence();
}

void execGPUkerenel(){
	cudaProfilerStart();
	cudatimeStamp ts(10);
	printf("%d\n",CUDART_VERSION);
	int numthreads= NUMTHREADS;
	int numblocks = 256;

	size_t num_items = (size_t) numblocks*numthreads;
	//host memory
	int *h_in=(int *)malloc(sizeof(int)*num_items);
	int *h_out=(int *)malloc(sizeof(int)*num_items);
	for(int i=0;i<num_items;i++) h_in[i]=i;
	//device memory
	int *d_in;
	cudaMalloc((void **)&d_in,sizeof(int)*num_items);
	int *d_out;
	cudaMalloc((void **)&d_out,sizeof(int)*num_items);

	// Change dynamic Shared memory size (CC7.0~)
	//int maxbytes = 98304; // 96 KB 
	//cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

	//error check
	checkCudaStatus();
	//memcpy Host->Device
	ts.stamp();
	cudaMemcpy(d_in,h_in,sizeof(int)*num_items,cudaMemcpyHostToDevice);
	// memset
	ts.stamp();
	cudaMemset(d_out,0,sizeof(int)*num_items);
	//kernel
	ts.stamp();
	// cooperativeKernel (Pascal~)
	// void **args = { (void*)&d_in , (void*)&d_out  };
	// cudaLaunchCooperativeKernel(kernel, numblocks, numthreads, args);
	kernel <<< numblocks , numthreads >>> (d_in,d_out);

	ts.stamp();
	//memcpy Device->Host
	cudaMemcpy(h_out,d_out,sizeof(int)*num_items,cudaMemcpyDeviceToHost);
	ts.stamp();

	ts.print();//ts.print_hori();
	printCudaLastError();

	printf("occupancy,%4.3f,SMcount,%d,activeblock,%d\n",occupancy(kernel,NUMTHREADS),get_sm_count(),get_activeblock_per_device(kernel,NUMTHREADS));

	//memory free
	free(h_in);
	free(h_out);
	cudaFree(d_in);
	cudaFree(d_out);

	cudaProfilerStop();

}

int main(int argc,char **argv){
	
	GPUBoost(1);

	execGPUkerenel();

	return 0;
}
