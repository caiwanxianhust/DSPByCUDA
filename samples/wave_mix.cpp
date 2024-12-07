#include "wave.h"
#include <cstdio>

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

    noisy.mixing(noise);

    noisy.writeWave(outputPath);
    noisy.readWave(outputPath);

    return 0;
}