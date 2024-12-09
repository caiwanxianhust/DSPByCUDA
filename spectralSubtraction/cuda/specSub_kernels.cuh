#pragma once


#include "cuda_kernels.cuh"
#include "utils.h"

#include <assert.h>
#include <cufft.h>

namespace DSPSpectralSubtraction {

// Complex abs
__device__ __host__ inline float ComplexABS(cufftComplex a)
{
    return sqrtf(a.x * a.x + a.y * a.y);
}

// Complex square
__device__ __host__ inline float ComplexSquare(cufftComplex a)
{
    return a.x * a.x + a.y * a.y;
}

// complex pow
__device__ __host__ inline float ComplexPow(cufftComplex a, float power)
{
    return powf(a.x * a.x + a.y * a.y, power / 2.0f);
}

// Complex norm
__device__ __host__ inline cufftComplex ComplexNorm(cufftComplex a)
{
    cufftComplex b;
    float a_abs = (ComplexABS(a) + 1e-19);
    b.x = a.x / a_abs;
    b.y = a.y / a_abs;
    return b;
}

// Complex scale
__device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s)
{
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

void launchInitKernel(int *overlap_counts, float *denoise_signal, const uint32_t length, cudaStream_t stream = 0);

template <typename T>
void launchQuantizedKernel(const float *__restrict__ src, T *__restrict__ dst, const int length, const float quantized_scale, cudaStream_t stream = 0);

void launchDeWindowingDeFramingKernel(const cufftReal *__restrict__ src, float *__restrict__ dst, const int32_t *__restrict__ overlap_counts,
                                      const uint32_t length, const int hop_len, const int frame_len, const int batch,
                                      const float fft_scale, cudaStream_t stream = 0);

template <typename T>
void launchSingleSrcFramingAndWindowingKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                              const uint32_t length,
                                              const int hop_len, const int frame_len, const int batch,
                                              const float quantized_scale, cudaStream_t stream = 0);

template <typename T>
void launchSingleSrcFramingAndWindowingCountOverlapKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                                          int32_t *__restrict__ overlap_counts, const uint32_t length,
                                                          const int hop_len, const int frame_len, const int batch,
                                                          const float quantized_scale, cudaStream_t stream = 0);

void launchPowerBlockReduceSum(const cufftComplex *d_x, float *d_y, const uint32_t nrows, const uint32_t ncols,
                               const float power, cudaStream_t stream = 0);

void launchSpecSubKernel(cufftComplex *__restrict__ noisy_spectrum, cufftReal *__restrict__ mean_noise_power_spectrum,
                         const int spectrum_len, const int noisy_batch, const float alpha, const float beta,
                         cudaStream_t stream = 0);

template <typename T1, typename T2>
void launchFramingAndWindowingKernel(const T1 *__restrict__ src_noise, const T1 *__restrict__ src_noisy,
                                     T2 *__restrict__ noise, T2 *__restrict__ noisy, const uint32_t noise_len,
                                     const uint32_t noisy_len, const int hop_len, const int frame_len, const float scale,
                                     cudaStream_t stream = 0);

template <typename T1, typename T2>
void launchFramingAndWindowingKernel(const T1 *__restrict__ src, T2 *__restrict__ noise, T2 *__restrict__ noisy,
                                     const uint32_t noise_len, const uint32_t noisy_len,
                                     const int hop_len, const int frame_len, const float scale,
                                     cudaStream_t stream = 0);



/**谱减、加相位建谱
 * noisy_spectrum: [noisy_batch, spectrum_len]
 * mean_noise_power_spectrum_: [spectrum_len]
 */
__global__ void specSubKernel(cufftComplex *__restrict__ noisy_spectrum, cufftReal *__restrict__ mean_noise_power_spectrum_,
                              const int spectrum_len, const int noisy_batch, const float alpha, const float beta)
{
    int offset = blockIdx.x * spectrum_len;
    float noisy, noise;
    cufftComplex temp;
    for (int tid = threadIdx.x; tid < spectrum_len; tid += blockDim.x)
    {
        // 谱减
        temp = noisy_spectrum[offset + tid];
        noise = __ldg(mean_noise_power_spectrum_ + tid);
        noisy = ComplexSquare(temp) - noise * alpha;
        noisy = noisy > 0 ? noisy : beta * noise;
        // 建谱
        temp = ComplexScale(ComplexNorm(temp), sqrtf(noisy));
        noisy_spectrum[offset + tid] = temp;
    }
}

void launchSpecSubKernel(cufftComplex *__restrict__ noisy_spectrum, cufftReal *__restrict__ mean_noise_power_spectrum,
                         const int spectrum_len, const int noisy_batch, const float alpha, const float beta, cudaStream_t stream)
{
    dim3 grid(noisy_batch);
    dim3 block(256);
    specSubKernel<<<grid, block, 0, stream>>>(noisy_spectrum, mean_noise_power_spectrum, spectrum_len, noisy_batch, alpha, beta);
}

/** noise 和 noisy 来自不同信号源
 * grid(batch), block(256)
 * T1: int8 or int16
 * T2: cufftReal or cufftComplex
 */
template <typename T1, typename T2>
__global__ void framingAndWindowingKernel(const T1 *__restrict__ src_noise, const T1 *__restrict__ src_noisy,
                                          T2 *__restrict__ noise, T2 *__restrict__ noisy, const uint32_t noise_len,
                                          const uint32_t noisy_len, const int hop_len, const int frame_len,
                                          const int noise_batch, const float scale)
{
    int src_idx, dst_idx;
    T2 val1, val2, w;
    int batch_id = blockIdx.x;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * hop_len + tid;
        dst_idx = blockIdx.x * frame_len + tid;
        val2 = src_idx < noisy_len ? static_cast<T2>(src_noisy[src_idx]) * scale : static_cast<T2>(0.0f);
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        noisy[dst_idx] = val2 * w;
        if (batch_id < noise_batch)
        {
            val1 = src_idx < noise_len ? static_cast<T2>(src_noise[src_idx]) * scale : static_cast<T2>(0.0f);
            noise[dst_idx] = val1 * w;
        }
    }
}

template <typename T1, typename T2>
void launchFramingAndWindowingKernel(const T1 *__restrict__ src_noise, const T1 *__restrict__ src_noisy,
                                     T2 *__restrict__ noise, T2 *__restrict__ noisy, const uint32_t noise_len,
                                     const uint32_t noisy_len, const int hop_len, const int frame_len, const float scale,
                                     cudaStream_t stream)
{
    printf("in launchFramingAndWindowingKernel\n");

    assert(noise_len < noisy_len);

    int nosie_batch = (noise_len - hop_len) / hop_len + 1;
    int nosiy_batch = (noisy_len - hop_len) / hop_len + 1;

    int block_size = 256;
    int grid_size = nosiy_batch;
    framingAndWindowingKernel<T1, T2><<<grid_size, block_size, 0, stream>>>(src_noise, src_noisy, noise, noisy, noise_len, noisy_len, hop_len, frame_len, nosie_batch, scale);
}

/**noise 和 noisy 来自同一个信号源
 * grid(batch), block(256)
 * T1: int8 or int16
 * T2: cufftReal or cufftComplex
 */
template <typename T1, typename T2>
__global__ void framingAndWindowingKernel(const T1 *__restrict__ src, T2 *__restrict__ noise, T2 *__restrict__ noisy,
                                          const uint32_t length, const int hop_len, const int frame_len, const int noise_batch,
                                          const float scale)
{
    int src_idx, dst_idx;
    T2 val, w;
    int batch_id = blockIdx.x;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * hop_len + tid;
        dst_idx = blockIdx.x * frame_len + tid;
        val = src_idx < length ? static_cast<T2>(src[src_idx]) * scale : static_cast<T2>(0.0f);
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        noisy[dst_idx] = val * w;
        if (batch_id < noise_batch)
        {
            noise[dst_idx] = val * w;
        }
    }
}

template <typename T1, typename T2>
void launchFramingAndWindowingKernel(const T1 *__restrict__ src, T2 *__restrict__ noise, T2 *__restrict__ noisy,
                                     const uint32_t noise_len, const uint32_t noisy_len,
                                     const int hop_len, const int frame_len, const float scale,
                                     cudaStream_t stream)
{
    assert(noise_len < noisy_len);

    int nosie_batch = (noise_len - hop_len) / hop_len + 1;
    int nosiy_batch = (noisy_len - hop_len) / hop_len + 1;

    int block_size = 256;
    int grid_size = nosiy_batch;
    framingAndWindowingKernel<T1, T2><<<grid_size, block_size, 0, stream>>>(src, noise, noisy, noisy_len, hop_len, frame_len, nosie_batch, scale);
}

/**分帧、加窗、记录信号重叠次数
 * grid(batch), block(256)
 * T: int8 or int16
 * overlap_counts: [length, ]
 */
template <typename T>
__global__ void singleSrcFramingAndWindowingCountOverlapKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                                               int32_t *__restrict__ overlap_counts,
                                                               const uint32_t length, const int hop_len, const int frame_len, const int batch,
                                                               const float quantized_scale)
{
    int src_idx, dst_idx;
    float val, w;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * hop_len + tid;
        dst_idx = blockIdx.x * frame_len + tid;
        if (src_idx < length)
        {
            val = static_cast<cufftReal>(src[src_idx]) * quantized_scale;
            atomicAdd(&overlap_counts[src_idx], 1);
        }
        else
        {
            val = static_cast<float>(0.0f);
        }
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        dst[dst_idx] = val * w;
    }
}

template <typename T>
void launchSingleSrcFramingAndWindowingCountOverlapKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                                          int32_t *__restrict__ overlap_counts, const uint32_t length,
                                                          const int hop_len, const int frame_len, const int batch,
                                                          const float quantized_scale, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = batch;
    singleSrcFramingAndWindowingCountOverlapKernel<T><<<grid_size, block_size, 0, stream>>>(src, dst, overlap_counts, length, hop_len, frame_len, batch, quantized_scale);
}

/**分帧、加窗
 * grid(batch), block(256)
 * T: int8 or int16
 */
template <typename T>
__global__ void singleSrcFramingAndWindowingKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                                   const uint32_t length, const int hop_len, const int frame_len, const int batch,
                                                   const float quantized_scale)
{
    int src_idx, dst_idx;
    cufftReal val, w;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * hop_len + tid;
        dst_idx = blockIdx.x * frame_len + tid;
        val = src_idx < length ? static_cast<cufftReal>(src[src_idx]) * quantized_scale : static_cast<cufftReal>(0.0f);
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        dst[dst_idx] = val * w;
    }
}

template <typename T>
void launchSingleSrcFramingAndWindowingKernel(const T *__restrict__ src, cufftReal *__restrict__ dst,
                                              const uint32_t length,
                                              const int hop_len, const int frame_len, const int batch,
                                              const float quantized_scale, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = batch;
    singleSrcFramingAndWindowingKernel<T><<<grid_size, block_size, 0, stream>>>(src, dst, length, hop_len, frame_len, batch, quantized_scale);
}

/**
 * grid(batch), block(256)
 * T1: int8 or int16
 * T2: cufftReal or cufftComplex
 */
template <typename T1, typename T2>
__global__ void fftInitKernel(const T1 *__restrict__ src, T2 *__restrict__ dst,
                              const uint32_t length, const int hop_len, const int frame_len)
{
    int bitsPerSample = sizeof(T1) * 8;
    T2 alpha = static_cast<T2>(1.0f / powf(2.0f, bitsPerSample - 1));

    int src_idx, dst_idx;
    T2 val, w;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * hop_len + tid;
        dst_idx = blockIdx.x * frame_len + tid;
        val = src_idx < length ? static_cast<T2>(src[src_idx]) * alpha : static_cast<T2>(0.0);
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        dst[dst_idx] = val * w;
    }
}

template <typename T1, typename T2>
void launchFftInitKernel(const T1 *__restrict__ src, T2 *__restrict__ dst, const uint32_t length,
                         const int hop_len, const int frame_len, const int batch, cudaStream_t stream = 0)
{
    int block_size = 256;
    int grid_size = batch;
    fftInitKernel<T1, T2><<<grid_size, block_size, 0, stream>>>(src, dst, length, hop_len, frame_len);
}

/**去窗、合帧
 * grid(batch), block(256)
 *
 */
__global__ void deWindowingDeFramingKernel(const cufftReal *__restrict__ src, float *__restrict__ dst,
                                           const int32_t *__restrict__ overlap_counts,
                                           const uint32_t length, const int hop_len, const int frame_len,
                                           const float fft_scale)
{
    int src_idx, dst_idx;
    float val, w;

    for (int tid = threadIdx.x; tid < frame_len; tid += blockDim.x)
    {
        src_idx = blockIdx.x * frame_len + tid;
        dst_idx = blockIdx.x * hop_len + tid;
        w = 0.54f - 0.46f * cosf(2.0f * 3.1415926f * tid / (frame_len - 1));
        val = dst_idx < length ? (src[src_idx] * fft_scale / w) : 0.0f;
        val /= overlap_counts[dst_idx];
        atomicAdd(&dst[dst_idx], val);
    }
}

void launchDeWindowingDeFramingKernel(const cufftReal *__restrict__ src, float *__restrict__ dst, const int32_t *__restrict__ overlap_counts,
                                      const uint32_t length, const int hop_len, const int frame_len, const int batch,
                                      const float fft_scale, cudaStream_t stream)
{
    deWindowingDeFramingKernel<<<batch, 256, 0, stream>>>(src, dst, overlap_counts, length, hop_len, frame_len, fft_scale);
}

/**量化
 * grid((length -1) / 256 + 1) block(256)
 * T: int8 or int16
 */
template <typename T>
__global__ void quantizedKernel(const float *__restrict__ src, T *__restrict__ dst, const int length, const float quantized_scale)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        dst[tid] = static_cast<T>(src[tid] / (quantized_scale + 1e-19));
    }
}

template <typename T>
void launchQuantizedKernel(const float *__restrict__ src, T *__restrict__ dst, const int length, const float quantized_scale, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (length - 1) / block_size + 1;
    quantizedKernel<T><<<grid_size, block_size, 0, stream>>>(src, dst, length, quantized_scale);
}

__global__ void powerBlockReduceSum(const cufftComplex *inp, float *out, const uint32_t ncols, const float power)
{
    float val = 0.0f;
    uint32_t offset = blockIdx.x * ncols;
    for (uint32_t i = threadIdx.x; i < ncols; i += blockDim.x)
    {
        val += ComplexSquare(inp[offset + i]);
    }
    __syncthreads();
    float blockSum;
    blockSum = blockAllReduceSum(val);
    if (threadIdx.x == 0)
    {
        out[blockIdx.x] = blockSum / ncols;
    }
}

void launchPowerBlockReduceSum(const cufftComplex *d_x, float *d_y, const uint32_t nrows, const uint32_t ncols,
                               const float power, cudaStream_t stream)
{
    uint32_t gird_size = nrows;
    uint32_t block_size = 256;
    powerBlockReduceSum<<<gird_size, block_size, 0, stream>>>(d_x, d_y, ncols, power);
}

__global__ void initKernel(int *overlap_counts, float *denoise_signal, const uint32_t length)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length)
    {
        overlap_counts[tid] = 0;
        denoise_signal[tid] = 0.0f;
    }
}

void launchInitKernel(int *overlap_counts, float *denoise_signal, const uint32_t length, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (length - 1) / block_size + 1;
    initKernel<<<grid_size, block_size, 0, stream>>>(overlap_counts, denoise_signal, length);
}

}