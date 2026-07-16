add_library(${GEOM_LIB_NAME} STATIC

  camera.cu

  voxel_data_structures.cu 
  voxel_data_structures.cpp
  streamer.cu
  streamer.cpp

  mesh_extractor.cpp
  mesh_extractor.cu
  marching_cubes.cu

  gaussian_data_structures.cu
  gaussian_data_structures.cpp

  #madtree stuff
  surface_normal_estimator/mad_tree.cpp
)

target_include_directories(${GEOM_LIB_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/src/gs/
  ${eigen_SOURCE_DIR}
  ${GTEST_INCLUDE_DIR}
)

target_link_libraries(${GEOM_LIB_NAME}
  PUBLIC
    ${GS_LIB_NAME}
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
