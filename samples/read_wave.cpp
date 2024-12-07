#include "wave.h"
#include <cstdio>

int main(int argc, char** argv) 
{
    WaveSignal noise;
    if (argc < 1) {
        printf("Missing command-line arguments\n");
        return 1;
    }
    char* filePath = argv[1];
    noise.readWave(filePath);

    return 0;
}