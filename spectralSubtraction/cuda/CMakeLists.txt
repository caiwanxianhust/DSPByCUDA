set(spectral_subtraction_kernel_files
    spectralSubtraction.cu
)


set(SPECTRAL_SUBTRACTION_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/spectralSubtraction
    ${PROJECT_SOURCE_DIR}/spectralSubtraction/cuda
)

include_directories(
    ${SPECTRAL_SUBTRACTION_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/spectralSubtraction/lib)

message("-- Assign fasterLlama include directories (include_directories=${SPECTRAL_SUBTRACTION_HEADER_DIR})")
message("-- in fasterLlama cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(spectralSubtraction SHARED ${spectral_subtraction_kernel_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(spectralSubtraction PUBLIC cxx_std_14)
target_compile_options(spectralSubtraction PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(spectralSubtraction PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(spectralSubtraction PUBLIC -lcufft -lcudart)
