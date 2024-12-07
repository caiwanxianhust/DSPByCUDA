#include "wave.h"
#include "spectralSubtraction.h"
#include "utils.h"

#include <assert.h>


template <BitsPerSample BITS_>
void deNoise(int32_t frame_len, int32_t hop_len, char *noise_signal, uint32_t noise_len, uint32_t max_length, char *noisy_signal, uint32_t noisy_len)
{
    typedef SpectralSubtractionTraits<BITS_> Traits_;
    typedef typename Traits_::DataType DataType_;

    SpectralSubtraction<BITS_> model(frame_len, hop_len, 1, (DataType_ *)noise_signal, noise_len, max_length);
    
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    cudaEventQuery(start);

    model.run((DataType_ *)noisy_signal, noisy_len);

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    printf("Time = %g ms.\n", elapsedTime);
}

template void deNoise<BitsPerSample::BIT16>(int32_t frame_len, int32_t hop_len, char *noise_signal, uint32_t noise_len, uint32_t max_length, char *noisy_signal, uint32_t noisy_len);

template void deNoise<BitsPerSample::BIT8>(int32_t frame_len, int32_t hop_len, char *noise_signal, uint32_t noise_len, uint32_t max_length, char *noisy_signal, uint32_t noisy_len);


int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Missing command-line arguments\n");
        return 1;
    }
    char* nosiyPath = argv[1];
    char* noisePath = argv[2];
    char* outputPath;
    outputPath = (argc >= 3) ? argv[3] : nosiyPath;

    WaveSignal noise;
    noise.readWave(noisePath);

    WaveSignal noisy;
    noisy.readWave(nosiyPath);
 
    int32_t frame_len = 400;
    int32_t hop_len = 160;

    assert(noise.bitsPerSample == 8 || noise.bitsPerSample == 16);
    uint32_t noise_length, noisy_length;

    if (noise.bitsPerSample == 8) 
    {
        noise_length = noise.dataSize;
        noisy_length = noisy.dataSize;
        deNoise<BitsPerSample::BIT8>(frame_len, hop_len, noise.signal, noise_length, noisy.sampleRate * 300, noisy.signal, noisy_length);
    }
    else {
        noise_length = noise.dataSize / 2;
        noisy_length = noisy.dataSize / 2;
        deNoise<BitsPerSample::BIT16>(frame_len, hop_len, noise.signal, 32000, noisy.sampleRate * 300, noisy.signal, noisy_length);
    }

    noisy.writeWave(outputPath);
    return 0;
}