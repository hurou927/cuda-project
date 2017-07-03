#ifndef _MY_CUDA_DEVICE
#define _MY_CUDA_DEVICE


#include <iostream>
#include <string>
#include "my_cuda_warp_scan.cuh"



__device__ __forceinline__ int get_smid(){
    int SMID = -1;
	asm("mov.u32 %0, %%smid;" : "=r"(SMID));
	return SMID;
}
__device__ __forceinline__ int get_sm_num(){
    int NSMID = -1;
	asm("mov.u32 %0, %%nsmid;" : "=r"(NSMID));
	return NSMID;
}

__device__ __forceinline__ unsigned int BFE(unsigned int source, unsigned int bit_start, unsigned int num_bits){
    unsigned int bits;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
    return bits;
}
// __device__ __forceinline__ unsigned long long int BFE(unsigned long long int source, unsigned int bit_start, unsigned int num_bits){
//     const unsigned long long MASK = (1ull << num_bits) - 1;
//     return (source >> bit_start) & MASK;
// }


// PREFIX-SUM in SUM
template <typename T>
__device__ __forceinline__ T warp_sum(const T val,const int laneId){
    T output;
	output = val + warpShuffleIdx(val, laneId + 1, 32);
	output += warpShuffleIdx(output, laneId + 2, 32);
	output += warpShuffleIdx(output, laneId + 4, 32);
	output += warpShuffleIdx(output, laneId + 8, 32);
	output += warpShuffleIdx(output, laneId + 16, 32);
	return output;
}

// sum in block
// THREAD_NUM must be multiples of 32. --> 64 96 128...
template<typename T, int THREAD_NUM>
__device__ __forceinline__ T block_sum(T val,const int warpId,const int laneId){
	__shared__ T warpSum[ THREAD_NUM>>5 ];
	T sum=warp_sum(val,laneId);
	if( laneId == 31){
		warpSum[warpId] = sum;
	}
	__syncthreads();

    if(warpId==0){
	    T val = 0;
	    if(laneId < (THREAD_NUM>>5) )
		    val = warpSum[laneId];

        val = warp_sum(val,laneId);

	    if( laneId == 0 )
		    warpSum[0] = val;
    }
    __syncthreads();

	sum = warpSum[0];

	return sum;
}


// PREFIX-SUM in block
// THREAD_NUM must be multiples of 32. --> 64 96 128...
template<typename T, int THREAD_NUM>
__device__ __forceinline__ T block_scan(T val,const int warpId,const int laneId){
    //(1<<Log2< ELE_NUM >::VALUE)
    //(1<<Log2< (ELE_NUM>>5) >::VALUE)
	__shared__ T warpSum[ THREAD_NUM>>5 ];
	T sum=warp_scan<32>(val,laneId);
	if( laneId == 31){
		warpSum[warpId] = sum;
	}
	__syncthreads();
    if(THREAD_NUM>64){
	    if(warpId==0){
		    T val = 0;
		    if(laneId < (THREAD_NUM>>5) )
			    val = warpSum[laneId];

            val = warp_scan < (THREAD_NUM>>5) > (val,laneId);

		    if(laneId < (THREAD_NUM>>5))
			    warpSum[laneId] = val;
	    }
	    __syncthreads();
    }
	if(warpId > 0){
		sum += warpSum[warpId-1];
	}
	return sum;
}
#endif
