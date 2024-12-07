set(wavp_files
    wave.cpp
)


set(WAVE_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/wave
)

include_directories(
    ${WAVE_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/wave/lib)

message("-- Assign fasterLlama include directories (include_directories=${WAVE_HEADER_DIR})")
message("-- in wave cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(wavesignal SHARED ${wavp_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(wavesignal PUBLIC cxx_std_14)
# target_compile_options(wavesignal PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
# set_target_properties(wavesignal PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(wavesignal PUBLIC)
