#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_INFO

#include "wave.h"

#include <cstdio>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace DSPWave {
int WaveSignal::readWave(const char *filePath)
{
    // 第零步：从硬盘读取文件
    FILE *fp;
    fp = fopen(filePath, "rb");
    if (!fp)
    {
        spdlog::error("no such file or directory: {}", filePath);
        return 1;
    }
    else
    {
        spdlog::info("open the wave file success! : {}", filePath);
    }

    // 第一步：对RIFF区块的读取
    fread(riffId, sizeof(char), 4, fp);                                                       // 读取'RIFF'
    if (('R' != riffId[0]) || ('I' != riffId[1]) || ('F' != riffId[2]) || ('F' != riffId[3])) // 零	这两个字符串相等。
    {
        spdlog::error("invalid file! : riffId is not RIFF");
        return 2;
    }

    fread(&riffSize, sizeof(uint32_t), 1, fp); // 读取文件大小(整个文件的长度减去ID和Size的长度)
    spdlog::info("riffSize: {}", riffSize);

    fread(waveId, sizeof(char), 4, fp); // 读取'RIFF'
    if (('W' != waveId[0]) || ('A' != waveId[1]) || ('V' != waveId[2]) || ('E' != waveId[3]))
    {
        spdlog::error("invalid file! : waveId is not WAVE");
        return 3;
    }

    // 第二步：FORMAT区块
    fread(fmtId, sizeof(char), 4, fp); // 读取4字节 "fmt ";

    fread(&fmtSize, sizeof(uint32_t), 1, fp); // Size表示该区块数据的长度（不包含ID和Size的长度）
    // 若Format Chunk的size大小为18，则该模块的最后两个字节为附加信息
    spdlog::info("fmtSize: {}", fmtSize);
    if (16 != fmtSize)
    {
        spdlog::error("the fmtSize {} is not supported!", fmtSize);
        return 4;
    }

    fread(&audioFormat, sizeof(uint16_t), 1, fp); // 读取文件tag  音频格式 PCM信号采样 = 1
    spdlog::info("audioFormat: {}", audioFormat);

    fread(&numChannels, sizeof(uint16_t), 1, fp); // 读取通道数目
    spdlog::info("numChannels: {}", numChannels);

    fread(&sampleRate, sizeof(uint32_t), 1, fp); // 读取采样率大小
    spdlog::info("sampleRate: {}", sampleRate);

    fread(&byteRate, sizeof(uint32_t), 1, fp); // 每秒数据字节数 SampleRate * NumChannels * BitsPerSample / 8
    spdlog::info("byteRate: {} bytes/s", byteRate);

    fread(&blockAlign, sizeof(uint16_t), 1, fp); // 每个采样所需的字节数 = NumChannels * BitsPerSample / 8
    spdlog::info("blockAlign: {}", blockAlign);

    fread(&bitsPerSample, sizeof(uint16_t), 1, fp); // 每个采样存储的bit数，8：8bit，16：16bit，32：32bit
    spdlog::info("bitsPerSample: {}", bitsPerSample);
    if (bitsPerSample != 8 && bitsPerSample != 16)
    {
        spdlog::error("the bitsPerSample {} is not supported!", bitsPerSample);
        return 5;
    }

    // 第三步： DATA区块
    fread(dataId, sizeof(char), 4, fp); // 读入'data'
    if (('d' != dataId[0]) || ('a' != dataId[1]) || ('t' != dataId[2]) || ('a' != dataId[3]))
    {
        spdlog::error("invalid file! : dataId is not data");
        return 6;
    }
    // 信号的字节数目
    fread(&dataSize, sizeof(uint32_t), 1, fp); // 读取数据大小
    spdlog::info("dataSize: {} bytes", dataSize);

    // 信号数据本身
    signal = new char[dataSize]; // 读取数据
    fread(signal, sizeof(char), dataSize, fp);

    fclose(fp);
    spdlog::info("read wave file {} successful!", filePath);

    return 0;
}

int WaveSignal::writeWave(const char *filePath)
{
    FILE *fp;
    fp = fopen(filePath, "wb");

    fwrite(riffId, sizeof(char), 4, fp);
    fwrite(&riffSize, sizeof(uint32_t), 1, fp);
    fwrite(waveId, sizeof(char), 4, fp);
    fwrite(fmtId, sizeof(char), 4, fp);
    fwrite(&fmtSize, sizeof(uint32_t), 1, fp);
    fwrite(&audioFormat, sizeof(uint16_t), 1, fp);
    fwrite(&numChannels, sizeof(uint16_t), 1, fp);
    fwrite(&sampleRate, sizeof(uint32_t), 1, fp);
    fwrite(&byteRate, sizeof(uint32_t), 1, fp);
    fwrite(&blockAlign, sizeof(uint16_t), 1, fp);
    fwrite(&bitsPerSample, sizeof(uint16_t), 1, fp);
    fwrite(dataId, sizeof(char), 4, fp);
    fwrite(&dataSize, sizeof(uint32_t), 1, fp);
    fwrite(signal, sizeof(char), dataSize, fp);

    fclose(fp);
    spdlog::info("write wave file {} successful!", filePath);

    return 0;
}


void WaveSignal::mixing(const WaveSignal &other_wave)
{
    if (this->bitsPerSample != other_wave.bitsPerSample || this->numChannels != other_wave.numChannels || this->sampleRate != other_wave.sampleRate) {
        spdlog::error("The two wave files do not match.");
        return;
    }
    if (this->dataSize < other_wave.dataSize) {
        spdlog::error("The added audio file is larger than the source file.");
        return;
    }
    uint32_t a_length = this->dataSize * 8 / bitsPerSample;
    uint32_t b_length = other_wave.dataSize * 8 / bitsPerSample;
    if (bitsPerSample == 16) {
        int16_t * a = (int16_t *)this->signal;
        int16_t * b = (int16_t *)other_wave.signal;
        for (uint32_t i=0; i<a_length; ++i) {
            if (i < b_length) {
                int tmp = static_cast<int>(a[i]) + static_cast<int>(b[i]);
                tmp = tmp > INT16_MAX ? INT16_MAX : tmp;
                tmp = tmp < INT16_MIN ? INT16_MIN : tmp;
                a[i] = static_cast<int16_t>(tmp);
            }
        }
        this->signal = (char *)a;
    }
    else
    {
        int8_t * a = (int8_t *)this->signal;
        int8_t * b = (int8_t *)other_wave.signal;
        for (uint32_t i=0; i<a_length; ++i) {
            if (i < b_length) {
                int tmp = static_cast<int>(a[i]) + static_cast<int>(b[i]);
                tmp = tmp > INT8_MAX ? INT8_MAX : tmp;
                tmp = tmp < INT8_MIN ? INT8_MIN : tmp;
                a[i] = static_cast<int8_t>(tmp);
            }
        }
        this->signal = (char *)a;
    }
    spdlog::info("the signal has been mixed successful!");
}

void WaveSignal::resetTimeLen(const float time_len)
{
    uint32_t new_data_size = time_len * this->byteRate;
    if (new_data_size > this->dataSize) {
        spdlog::error("new_data_size is larger than current data.");
        return;
    }
    this->riffSize -= (this->dataSize - new_data_size);
    this->dataSize = new_data_size;
    spdlog::info("the signal is reset to %f seconds", time_len);
}

} // namespace DSPWave


