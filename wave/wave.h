#pragma once

#include <cstdint>
#include <cstring>

namespace DSPWave {
struct WaveSignal
{
    /* data */
    char riffId[4];
    uint32_t riffSize;
    char waveId[4];
    char fmtId[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char dataId[4];
    uint32_t dataSize;
    char *signal;

    WaveSignal() : riffSize(0), fmtSize(16), audioFormat(1), numChannels(1), sampleRate(16000),
                   byteRate(32000), blockAlign(2), bitsPerSample(16), dataSize(0)
    {
        strcpy(riffId, "RIFF");
        strcpy(waveId, "WAVE");
        strcpy(fmtId, "fmt ");
        strcpy(dataId, "data");
    }

    int readWave(const char *filePath);
    int writeWave(const char *filePath);
    void mixing(const WaveSignal &other_wave);
    void resetTimeLen(const float time_len);

private:
    WaveSignal(const WaveSignal &wave);
    WaveSignal &operator=(const WaveSignal &wave);
};


} // namespace DSPWave

