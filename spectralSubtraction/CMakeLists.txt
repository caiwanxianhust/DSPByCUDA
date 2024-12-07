set (SPECTRAL_SUBTRACTION_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/spectralSubtraction
    ${PROJECT_SOURCE_DIR}/spectralSubtraction/cuda
)

include_directories(
    ${SPECTRAL_SUBTRACTION_HEADER_DIR}
)

add_subdirectory(cuda)

