#ifndef _MY_CUDA_UTIL
#define _MY_CUDA_UTIL

#include <iostream>

/**
 * Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
 //inductive
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2{
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };
};
//base
template <int N, int COUNT>
struct Log2<N, 0, COUNT>{
    enum {VALUE = (1 << (COUNT - 1) < N) ? COUNT :  COUNT - 1 };
};


#if CUDART_VERSION >= 9000
template <typename T>
__device__ __forceinline__ T warpShuffleIdx(const T var,const  int srcLane,const  int width){
	return __shfl_sync(0xFFFFFFFF,var,srcLane,width);
}
#else

template <typename T>
__device__ __forceinline__ T warpShuffleIdx(const T var,const  int srcLane,const  int width){
    T output;
    unsigned int *p_out = (unsigned int *)(&output);
    unsigned int *p_in  = (unsigned int *)(&var);
    for(int i=0;i<sizeof(T)/4;i++){
        *p_out = __shfl(*p_in,srcLane,width);
        p_in++;
        p_out++;
    }
    unsigned char *p_out_1byte = (unsigned char *)(p_out);
    unsigned char *p_in_1byte  = (unsigned char *)(p_in);
    for(int i=0;i<sizeof(T)-sizeof(T)/4*4;i++){
        *p_out_1byte=__shfl(*p_in_1byte,srcLane,width);
        p_out_1byte++;
        p_in_1byte++;
    }
    return output;
}

__device__ __forceinline__  int warpShuffleIdx(const int var,const  int srcLane,const  int width){
    return __shfl(var, srcLane, width);
}

__device__ __forceinline__  unsigned int warpShuffleIdx(const unsigned int var,const  int srcLane,const  int width){
    return __shfl(var, srcLane, width);
}

__device__ __forceinline__  long long int warpShuffleIdx(const long long int var,const  int srcLane,const  int width){
    long long int output;
    unsigned int lo,hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl(hi ,srcLane ,width);
    lo = __shfl(lo ,srcLane ,width);
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(output) : "r"(lo), "r"(hi));
    return output;
}

__device__ __forceinline__  unsigned long long int warpShuffleIdx(const unsigned long long int var,const  int srcLane,const  int width){
    unsigned long long int output;
    unsigned int lo,hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(var));
    hi = __shfl(hi ,srcLane ,width);
    lo = __shfl(lo ,srcLane ,width);
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(output) : "r"(lo), "r"(hi));
    return output;
}

__device__ __forceinline__  float warpShuffleIdx(const float var,const  int srcLane,const  int width){
    return __shfl(var, srcLane, width);
}
__device__ __forceinline__  double warpShuffleIdx(const double var,const  int srcLane,const  int width){
    double output;
    unsigned int lo,hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "d"(var));
    hi = __shfl(hi ,srcLane ,width);
    lo = __shfl(lo ,srcLane ,width);
    asm volatile("mov.b64 %0, {%1, %2};" : "=d"(output) : "r"(lo), "r"(hi));
    return output;
}
#endif


#endif
