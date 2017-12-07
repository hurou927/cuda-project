#ifndef _MY_CUDA_WARP_SCAN
#define _MY_CUDA_WARP_SCAN


#include <iostream>
#include "my_cuda_util.cuh"

/*
template <typename T, int ELE_NUM>//(1<<Log2< ELE_NUM >::VALUE)
__device__ __forceinline__ T warp_scan(T val,const int laneId){
    int limit = (1<<Log2< ELE_NUM >::VALUE);
    for(int j=1;j<limit;j=j<<1){
		T n = __shfl_up(val, j, 32);
		if (laneId >= j) val += n;
	}
	return val;
}
*/

template <int ELE_NUM>
__device__ __forceinline__ int warp_scan (int val,const int laneId){

    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .s32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.s32 r0, r0, %4;"
            "  mov.s32 %0, r0;"
            "}"
            : "=r"(val) : "r"(val), "r"(j), "r"(32), "r"(val));
	}
	return val;
}


template <int ELE_NUM>
__device__ __forceinline__ unsigned int warp_scan(unsigned int val,const int laneId){
    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .u32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.u32 r0, r0, %4;"
            "  mov.u32 %0, r0;"
            "}"
            : "=r"(val) : "r"(val), "r"(j), "r"(32), "r"(val));
	}
	return val;
}


template <int ELE_NUM>
__device__ __forceinline__ long long int warp_scan(long long int val,const int laneId){
    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .s64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.s64 r0, r0, %4;"
            "  mov.s64 %0, r0;"
            "}"
            : "=l"(val) : "l"(val), "r"(j), "r"(32), "l"(val));
	}
	return val;
}


template <int ELE_NUM>
__device__ __forceinline__ unsigned long long int warp_scan(unsigned long long int val,const int laneId){
    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .u64 r0;"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.u64 r0, r0, %4;"
            "  mov.u64 %0, r0;"
            "}"
            : "=l"(val) : "l"(val), "r"(j), "r"(32), "l"(val));
	}
	return val;
}


template <int ELE_NUM>
__device__ __forceinline__ float warp_scan(float val,const int laneId){
    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.up.b32 r0|p, %1, %2, %3;"
            "  @p add.f32 r0, r0, %4;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(val) : "f"(val), "r"(j), "r"(32), "f"(val));
	}
	return val;
}


template <int ELE_NUM>
__device__ __forceinline__ double warp_scan(double val,const int laneId){
    for(int j=1;j<(1<<Log2< ELE_NUM >::VALUE);j=j<<1){
        asm volatile(
            "{"
            "  .reg .u32 lo;"
            "  .reg .u32 hi;"
            "  .reg .pred p;"
            "  .reg .f64 r0;"
            "  mov.b64 %0, %1;"
            "  mov.b64 {lo, hi}, %1;"
            "  shfl.up.b32 lo|p, lo, %2, %3;"
            "  shfl.up.b32 hi|p, hi, %2, %3;"
            "  mov.b64 r0, {lo, hi};"
            "  @p add.f64 %0, %0, r0;"
            "}"
            : "=d"(val) : "d"(val), "r"(j), "r"(32));
	}
	return val;
}


#endif
