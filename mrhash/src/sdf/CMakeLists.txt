add_library(${GEOM_LIB_NAME} STATIC

  camera.cu

  voxel_data_structures.cu 
  voxel_data_structures.cpp
  directional_tsdf.cu
  streamer.cu
  streamer.cpp

)

target_include_directories(${GEOM_LIB_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/src/sdf
  ${GTEST_INCLUDE_DIR}
)

target_link_libraries(${GEOM_LIB_NAME}
  PUBLIC
    ${OpenCV_LIBS}
    CUDA::cudart
    Eigen3::Eigen
)

target_compile_options(${GEOM_LIB_NAME} PRIVATE -lineinfo $<$<COMPILE_LANGUAGE:CUDA>: -v -g -lineinfo --expt-relaxed-constexpr>)

set_target_properties(${GEOM_LIB_NAME} PROPERTIES
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  CUDA_SEPARABLE_COMPILATION ON
)
add_subdirectory(pybind)
