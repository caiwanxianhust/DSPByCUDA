set (read_wave_files
    read_wave.cpp
)

set (wave_mix_files
    wave_mix.cpp
)

set (spec_sub_16bit_files
    specSub_16bit.cpp
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/wave
    ${PROJECT_SOURCE_DIR}/spectralSubtraction
    ${PROJECT_SOURCE_DIR}/spectralSubtraction/cuda
)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/spectralSubtraction/lib
    ${PROJECT_SOURCE_DIR}/wave/lib
)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

add_executable(read_wave ${read_wave_files})
target_compile_features(read_wave PUBLIC cxx_std_14)
target_link_libraries(read_wave PUBLIC -lwavesignal)

add_executable(wave_mix ${wave_mix_files})
target_compile_features(wave_mix PUBLIC cxx_std_14)
target_link_libraries(wave_mix PUBLIC -lwavesignal)

add_executable(specSub_16bit ${spec_sub_16bit_files})
target_compile_features(specSub_16bit PUBLIC cxx_std_14)
target_link_libraries(specSub_16bit PUBLIC -lwavesignal -lspectralSubtraction)
