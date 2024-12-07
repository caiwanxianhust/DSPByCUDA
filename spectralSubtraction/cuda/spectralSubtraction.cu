#include "spectralSubtraction.h"
#include "specSub_kernels.cuh"
#include "utils.h"

template <BitsPerSample BITS_>
SpectralSubtraction<BITS_>::SpectralSubtraction(int32_t frame_len, int32_t hop_len, int32_t rank, DataType_ *noise_signal, uint32_t noise_length, uint32_t max_length, float alpha, float beta) : 
frame_len_(frame_len), hop_len_(hop_len), rank_(rank), max_length_(max_length), alpha_(alpha), beta_(beta)
{
    int spectrum_len = frame_len_ / 2 + 1;
    int max_batch = (max_length_ - 1) / hop_len_ + 1;
    int noise_batch = (noise_length - 1) / hop_len_ + 1;

    size_t mem_size = sizeof(DataType_) * noise_length +
                      sizeof(FFTReal_) * noise_batch * frame_len_ +
                      sizeof(FFTComplex_) * noise_batch * spectrum_len;

    CHECK_CUDA_ERROR(cudaMalloc((void **)&noise_signal_buf_, mem_size));

    noise_buf_ = (FFTReal_ *)(noise_signal_buf_ + noise_length);
    noise_spectrum_buf_ = (FFTComplex_ *)(noise_buf_ + noise_batch * frame_len_);

    CHECK_CUDA_ERROR(cudaMalloc((void **)&mean_noise_power_spectrum_, sizeof(FFTReal_) * spectrum_len));

    size_t nosiy_mem = sizeof(FFTReal_) * max_batch * frame_len_ +
                       sizeof(FFTComplex_) * max_batch * spectrum_len +
                       sizeof(float) * max_length_ +
                       sizeof(int) * max_length_ +
                       sizeof(DataType_) * max_length_;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&noisy_buf_, nosiy_mem));
    noisy_spectrum_buf_ = (FFTComplex_ *)(noisy_buf_ + max_batch * frame_len_);
    denoised_signal_buf_ = (float *)(noisy_spectrum_buf_ + max_batch * spectrum_len);
    overlap_counts_buf_ = (int *)(denoised_signal_buf_ + max_length_);
    noisy_signal_buf_ = (DataType_ *)(overlap_counts_buf_ + max_length_);

    // 从主机端拷贝数据到设备端 noise_signal -> noise_signal_buf_
    CHECK_CUDA_ERROR(cudaMemcpy(noise_signal_buf_, noise_signal, sizeof(DataType_) * noise_length, cudaMemcpyHostToDevice));

    printf("malloc device mem: %gGB\n", (mem_size + nosiy_mem) / 1024.0f / 1024.0f / 1024.0f);

    quantized_scale_ = (sizeof(DataType_) == 2) ? 1.0f / powf(2.0f, 15.0f) : 1.0f / powf(2.0f, 7.0f);

    launchSingleSrcFramingAndWindowingKernel<DataType_>(noise_signal_buf_, noise_buf_, noise_length, hop_len_, frame_len_, noise_batch, quantized_scale_);

    cufftHandle noise_fft_plan;
    // 计算噪声的傅里叶变换，noise_buf_ 形状为[frame_len_ / 2 + 1, noise_batch]
    int n[1];
    n[0] = frame_len_;
    int istride = 1;
    int idist = frame_len_;
    int ostride = noise_batch;
    int odist = 1;
    int inembed[2];
    int onembed[2];
    inembed[0] = frame_len_;
    onembed[0] = noise_batch;
    inembed[1] = noise_batch;
    onembed[1] = spectrum_len;

    CHECK_CUFFT_STATUS(cufftPlanMany(&noise_fft_plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_R2C, noise_batch));
    CHECK_CUFFT_STATUS(cufftExecR2C(noise_fft_plan, noise_buf_, noise_spectrum_buf_));

    launchPowerBlockReduceSum(noise_spectrum_buf_, mean_noise_power_spectrum_, spectrum_len, noise_batch, 2.0f);
    CHECK_CUFFT_STATUS(cufftDestroy(noise_fft_plan));
}


template <BitsPerSample BITS_>
void SpectralSubtraction<BITS_>::resetNoise(DataType_ *noise_signal, int noise_length)
{
    // 从主机端拷贝数据到设备端 noise_signal -> noise_signal_buf_
    CHECK_CUDA_ERROR(cudaMemcpy(noise_signal_buf_, noise_signal, sizeof(DataType_) * noise_length, cudaMemcpyHostToDevice));

    int noise_batch = (noise_length - 1) / hop_len_ + 1;
    int spectrum_len = frame_len_ / 2 + 1;

    launchSingleSrcFramingAndWindowingKernel<DataType_>(noise_signal_buf_, noise_buf_, noise_length, hop_len_, frame_len_, noise_batch, quantized_scale_);

    // 计算噪声的傅里叶变换，noise_buf_ 形状为[frame_len_ / 2 + 1, noise_batch]
    cufftHandle noise_fft_plan;
    int n[1];
    n[0] = frame_len_;
    int istride = 1;
    int idist = frame_len_;
    int ostride = noise_batch;
    int odist = 1;
    int inembed[2];
    int onembed[2];
    inembed[0] = frame_len_;
    onembed[0] = noise_batch;
    inembed[1] = noise_batch;
    onembed[1] = spectrum_len;

    CHECK_CUFFT_STATUS(cufftPlanMany(&noise_fft_plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_R2C, noise_batch));
    CHECK_CUFFT_STATUS(cufftExecR2C(noise_fft_plan, noise_buf_, noise_spectrum_buf_));

    launchPowerBlockReduceSum(noise_spectrum_buf_, mean_noise_power_spectrum_, spectrum_len, noise_batch, 2.0f);
    CHECK_CUFFT_STATUS(cufftDestroy(noise_fft_plan));
}

template <BitsPerSample BITS_>
void SpectralSubtraction<BITS_>::fftR2C(FFTReal_ *signal, FFTComplex_ *spectrum, const int batch)
{
    cufftHandle plan;
    int n[1];
    n[0] = frame_len_;
    int istride = 1;
    int idist = frame_len_;
    int ostride = 1;
    int odist = frame_len_ / 2 + 1;
    int inembed[2];
    int onembed[2];
    inembed[0] = frame_len_;
    onembed[0] = frame_len_ / 2 + 1;
    inembed[1] = batch;
    onembed[1] = batch;

    CHECK_CUFFT_STATUS(cufftPlanMany(&plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_R2C, batch));
    CHECK_CUFFT_STATUS(cufftExecR2C(plan, signal, spectrum));
    CHECK_CUFFT_STATUS(cufftDestroy(plan));
}

template <BitsPerSample BITS_>
void SpectralSubtraction<BITS_>::ifftC2R(FFTReal_ *signal, FFTComplex_ *spectrum, const int batch)
{
    cufftHandle plan;
    int n[1];
    n[0] = frame_len_;
    int istride = 1;
    int idist = frame_len_ / 2 + 1;
    int ostride = 1;
    int odist = frame_len_;
    int inembed[2];
    int onembed[2];
    inembed[0] = frame_len_ / 2 + 1;
    onembed[0] = frame_len_;
    inembed[1] = batch;
    onembed[1] = batch;

    CHECK_CUFFT_STATUS(cufftPlanMany(&plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_C2R, batch));
    CHECK_CUFFT_STATUS(cufftExecC2R(plan, spectrum, signal));
    CHECK_CUFFT_STATUS(cufftDestroy(plan));
}

/**
 * nosie: 分帧、加窗后的噪声
 * noisy: 分帧、加窗后的含噪声信号
 */
template <BitsPerSample BITS_>
void SpectralSubtraction<BITS_>::run(DataType_ *nosiy_signal, const uint32_t noisy_length)
{
    assert(noisy_length < max_length_);

    launchInitKernel(overlap_counts_buf_, denoised_signal_buf_, noisy_length);
    CHECK_CUDA_ERROR(cudaMemcpy(noisy_signal_buf_, nosiy_signal, sizeof(DataType_) * noisy_length, cudaMemcpyHostToDevice));

    int spectrum_len = frame_len_ / 2 + 1;
    int noisy_batch = (noisy_length - 1) / hop_len_ + 1;
    // 对含噪声信号分帧、加窗
    launchSingleSrcFramingAndWindowingCountOverlapKernel<DataType_>(noisy_signal_buf_, noisy_buf_, overlap_counts_buf_, noisy_length, hop_len_, frame_len_, noisy_batch, quantized_scale_);

    // 计算含噪声信号的傅里叶变换，d_noisy_spectrum 形状为[noisy_batch, spectrum_len]
    cufftHandle noisy_fft_plan;
    int n[1];
    n[0] = frame_len_;
    int istride = 1;
    int idist = frame_len_;
    int ostride = 1;
    int odist = spectrum_len;
    int inembed[2];
    int onembed[2];
    inembed[0] = frame_len_;
    onembed[0] = spectrum_len;
    inembed[1] = noisy_batch;
    onembed[1] = noisy_batch;

    CHECK_CUFFT_STATUS(cufftPlanMany(&noisy_fft_plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_R2C, noisy_batch));
    CHECK_CUFFT_STATUS(cufftExecR2C(noisy_fft_plan, noisy_buf_, noisy_spectrum_buf_));

    // 谱减、加相位建谱
    launchSpecSubKernel(noisy_spectrum_buf_, mean_noise_power_spectrum_, spectrum_len, noisy_batch, alpha_, beta_);

    cufftHandle ifft_plan;
    idist = spectrum_len;
    odist = frame_len_;
    inembed[0] = spectrum_len;
    onembed[0] = frame_len_;

    // 计算去噪声信号频谱的逆傅里叶变换
    CHECK_CUFFT_STATUS(cufftPlanMany(&ifft_plan, rank_, n, inembed, istride, idist, onembed, ostride,
                                     odist, CUFFT_C2R, noisy_batch));
    CHECK_CUFFT_STATUS(cufftExecC2R(ifft_plan, noisy_spectrum_buf_, noisy_buf_));

    // 去窗、恢复信号
    float fft_scale = 1.0f / frame_len_;
    launchDeWindowingDeFramingKernel(noisy_buf_, denoised_signal_buf_, overlap_counts_buf_, noisy_length, hop_len_, frame_len_, noisy_batch, fft_scale);

    // 量化
    launchQuantizedKernel<DataType_>(denoised_signal_buf_, noisy_signal_buf_, noisy_length, quantized_scale_);

    // 拷贝到主机端
    CHECK_CUDA_ERROR(cudaMemcpy(nosiy_signal, noisy_signal_buf_, sizeof(DataType_) * noisy_length, cudaMemcpyDeviceToHost));

    CHECK_CUFFT_STATUS(cufftDestroy(noisy_fft_plan));
    CHECK_CUFFT_STATUS(cufftDestroy(ifft_plan));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

template <BitsPerSample BITS_>
SpectralSubtraction<BITS_>::~SpectralSubtraction()
{
    CHECK_CUDA_ERROR(cudaFree(noise_signal_buf_));
    CHECK_CUDA_ERROR(cudaFree(mean_noise_power_spectrum_));
    CHECK_CUDA_ERROR(cudaFree(noisy_buf_));
}



template class SpectralSubtraction<BitsPerSample::BIT8>;
template class SpectralSubtraction<BitsPerSample::BIT16>;