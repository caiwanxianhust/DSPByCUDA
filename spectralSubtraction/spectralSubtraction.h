#include <cstdint>
#include <cfloat>

enum class BitsPerSample
{
    BIT16,
    BIT8
};

template <BitsPerSample BITS_>
class SpectralSubtractionTraits;

template <>
class SpectralSubtractionTraits<BitsPerSample::BIT16>
{
public:
    typedef int16_t DataType;
    typedef float FFTReal;
    typedef float2 FFTComplex;
};

template <>
class SpectralSubtractionTraits<BitsPerSample::BIT8>
{
public:
    typedef int8_t DataType;
    typedef float FFTReal;
    typedef float2 FFTComplex;
};

template <BitsPerSample BITS_>
class SpectralSubtraction
{
public:
    typedef SpectralSubtractionTraits<BITS_> Traits_;
    typedef typename Traits_::DataType DataType_;
    typedef typename Traits_::FFTReal FFTReal_;
    typedef typename Traits_::FFTComplex FFTComplex_;

    // 帧长，如每 25ms 一帧，则帧长为 25 * sampleRate_ / 1000
    int32_t frame_len_;
    // 帧移，若每次移动 10ms，则帧移为 10 * sampleRate_ / 1000
    int32_t hop_len_;
    // 维度
    int32_t rank_;
    // 最大含噪声信号的长度
    uint32_t max_length_;
    // 过减因子
    float alpha_;
    // 增益补偿因子
    float beta_;
    // 量化因子
    float quantized_scale_;

    DataType_ *noise_signal_buf_;
    DataType_ *noisy_signal_buf_;
    FFTReal_ *noise_buf_;
    FFTComplex_ *noise_spectrum_buf_;
    FFTReal_ *noisy_buf_;
    FFTComplex_ *noisy_spectrum_buf_;
    float *denoised_signal_buf_;
    int *overlap_counts_buf_;
    // 噪声平均功率谱 [spectrum_len, ]
    FFTReal_ *mean_noise_power_spectrum_;

    SpectralSubtraction(int32_t frame_len, int32_t hop_len, int32_t rank, DataType_ *noise_signal,
                        uint32_t noise_length, uint32_t max_length, float alpha = 10.0f, float beta = 0.003f);

    void resetNoise(DataType_ *noise_signal, int noise_length);

    void fftR2C(FFTReal_ *signal, FFTComplex_ *spectrum, const int batch);

    void ifftC2R(FFTReal_ *signal, FFTComplex_ *spectrum, const int batch);

    void run(DataType_ *nosiy_signal, const uint32_t noisy_length);

    ~SpectralSubtraction();
};

