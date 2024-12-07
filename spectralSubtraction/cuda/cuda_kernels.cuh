#pragma once

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpAllReduce(T val)
{
    auto func = ReductionOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = func(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <typename T>
__inline__ __device__ T blockAllReduceSum(T val)
{
    static __shared__ T shared[32];
    __shared__ T result;
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = warpAllReduce<SumOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = warpAllReduce<SumOp, T>(val);
    if (threadIdx.x == 0)
        result = val;
    __syncthreads();
    return result;
}
