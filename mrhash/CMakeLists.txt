cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

project(mrhash LANGUAGES C CXX)

# Allow user to override build type, default to Release for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug, Release, RelWithDebInfo)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
option(MRHASH_BUILD_APPS "Build mrhash example applications" OFF)
option(MRHASH_BUILD_TESTS "Build mrhash C++/CUDA tests" OFF)

macro(mrhash_reset_findcuda_cache)
  unset(CUDA_TOOLKIT_INCLUDE CACHE)
  unset(CUDA_CUDART_LIBRARY CACHE)
  unset(CUDA_CUDA_LIBRARY CACHE)
  unset(CUDA_cudart_static_LIBRARY CACHE)
  unset(CUDA_cudadevrt_LIBRARY CACHE)
  unset(CUDA_cublas_LIBRARY CACHE)
  unset(CUDA_cublas_device_LIBRARY CACHE)
  unset(CUDA_cublasemu_LIBRARY CACHE)
  unset(CUDA_cufft_LIBRARY CACHE)
  unset(CUDA_cufftemu_LIBRARY CACHE)
  unset(CUDA_cupti_LIBRARY CACHE)
  unset(CUDA_curand_LIBRARY CACHE)
  unset(CUDA_cusolver_LIBRARY CACHE)
  unset(CUDA_cusparse_LIBRARY CACHE)
  unset(CUDA_npp_LIBRARY CACHE)
  unset(CUDA_nppc_LIBRARY CACHE)
  unset(CUDA_nppi_LIBRARY CACHE)
  unset(CUDA_npps_LIBRARY CACHE)
  unset(CUDA_nvcuvenc_LIBRARY CACHE)
  unset(CUDA_nvcuvid_LIBRARY CACHE)
  unset(CUDA_nvToolsExt_LIBRARY CACHE)
  unset(CUDA_OpenCL_LIBRARY CACHE)
  unset(CUDA_GPU_DETECT_OUTPUT CACHE)
  unset(CUDA_VERSION CACHE)
endmacro()

macro(mrhash_reset_cudatoolkit_cache)
  unset(CUDAToolkit_BIN_DIR CACHE)
  unset(CUDAToolkit_CUPTI_INCLUDE_DIR CACHE)
  unset(CUDAToolkit_NVCC_EXECUTABLE CACHE)
  unset(CUDAToolkit_ROOT CACHE)
  unset(CUDAToolkit_nvToolsExt_INCLUDE_DIR CACHE)
  unset(CUDAToolkit_rt_LIBRARY CACHE)
  unset(FIND_PACKAGE_MESSAGE_DETAILS_CUDAToolkit CACHE)
  unset(_cmake_CUDAToolkit_implicit_link_directories CACHE)
  unset(_cmake_CUDAToolkit_include_directories CACHE)
endmacro()

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "${PROJECT_NAME}: OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# cuda
set(MRHASH_CUDA_SELECTION_SOURCE "auto-discovery")
set(MRHASH_REQUESTED_CUDA_ROOT "")
set(MRHASH_REQUESTED_CUDA_NVCC "")

if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
  if(NOT EXISTS "$ENV{CUDACXX}")
    message(FATAL_ERROR "CUDACXX is set to '$ENV{CUDACXX}', but that path does not exist.")
  endif()
  set(MRHASH_REQUESTED_CUDA_NVCC "$ENV{CUDACXX}")
  get_filename_component(_mrhash_requested_cuda_bin_dir "${MRHASH_REQUESTED_CUDA_NVCC}" DIRECTORY)
  get_filename_component(MRHASH_REQUESTED_CUDA_ROOT "${_mrhash_requested_cuda_bin_dir}" DIRECTORY)
  set(MRHASH_CUDA_SELECTION_SOURCE "CUDACXX")
elseif(CUDAToolkit_ROOT)
  set(MRHASH_REQUESTED_CUDA_ROOT "${CUDAToolkit_ROOT}")
  if(EXISTS "${MRHASH_REQUESTED_CUDA_ROOT}/bin/nvcc")
    set(MRHASH_REQUESTED_CUDA_NVCC "${MRHASH_REQUESTED_CUDA_ROOT}/bin/nvcc")
  endif()
  set(MRHASH_CUDA_SELECTION_SOURCE "CUDAToolkit_ROOT")
elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/nvcc")
  set(MRHASH_REQUESTED_CUDA_ROOT "$ENV{CONDA_PREFIX}")
  set(MRHASH_REQUESTED_CUDA_NVCC "$ENV{CONDA_PREFIX}/bin/nvcc")
  set(MRHASH_CUDA_SELECTION_SOURCE "CONDA_PREFIX")
endif()

if(MRHASH_REQUESTED_CUDA_ROOT)
  set(MRHASH_CUDATOOLKIT_CACHE_REFRESH FALSE)
  if(DEFINED CUDAToolkit_ROOT AND CUDAToolkit_ROOT AND NOT CUDAToolkit_ROOT STREQUAL MRHASH_REQUESTED_CUDA_ROOT)
    set(MRHASH_CUDATOOLKIT_CACHE_REFRESH TRUE)
  endif()
  if(MRHASH_REQUESTED_CUDA_NVCC AND DEFINED CUDAToolkit_NVCC_EXECUTABLE AND CUDAToolkit_NVCC_EXECUTABLE
     AND NOT CUDAToolkit_NVCC_EXECUTABLE STREQUAL MRHASH_REQUESTED_CUDA_NVCC)
    set(MRHASH_CUDATOOLKIT_CACHE_REFRESH TRUE)
  endif()
  if(MRHASH_CUDATOOLKIT_CACHE_REFRESH)
    message(STATUS "Refreshing cached CUDAToolkit discovery for ${MRHASH_REQUESTED_CUDA_ROOT}")
    mrhash_reset_cudatoolkit_cache()
  endif()
  message(STATUS "Preferring CUDA toolkit from ${MRHASH_CUDA_SELECTION_SOURCE}: ${MRHASH_REQUESTED_CUDA_ROOT}")
  set(CUDAToolkit_ROOT "${MRHASH_REQUESTED_CUDA_ROOT}" CACHE PATH "CUDA toolkit root" FORCE)
endif()
if(MRHASH_REQUESTED_CUDA_NVCC)
  set(ENV{CUDACXX} "${MRHASH_REQUESTED_CUDA_NVCC}")
endif()

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS 12.6 OR NOT CUDAToolkit_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR
    "mrhash supports CUDA toolkit versions 12.6 through 12.x, but found ${CUDAToolkit_VERSION} at ${CUDAToolkit_BIN_DIR}. "
    "Set CUDAToolkit_ROOT or CUDACXX to a supported CUDA 12.x installation.")
endif()

set(MRHASH_CUDA_ROOT "")
if(CUDAToolkit_ROOT)
  set(MRHASH_CUDA_ROOT "${CUDAToolkit_ROOT}")
elseif(CUDAToolkit_NVCC_EXECUTABLE)
  get_filename_component(_mrhash_cuda_bin_dir "${CUDAToolkit_NVCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(MRHASH_CUDA_ROOT "${_mrhash_cuda_bin_dir}" DIRECTORY)
elseif(CUDAToolkit_BIN_DIR)
  get_filename_component(MRHASH_CUDA_ROOT "${CUDAToolkit_BIN_DIR}" DIRECTORY)
elseif(CUDAToolkit_LIBRARY_ROOT)
  set(MRHASH_CUDA_ROOT "${CUDAToolkit_LIBRARY_ROOT}")
endif()

set(MRHASH_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}")
if(NOT MRHASH_CUDA_COMPILER)
  set(MRHASH_CUDA_COMPILER "${MRHASH_REQUESTED_CUDA_NVCC}")
endif()

if(NOT MRHASH_CUDA_ROOT)
  message(FATAL_ERROR "Unable to determine the CUDA toolkit root from CUDAToolkit discovery.")
endif()
if(NOT MRHASH_CUDA_COMPILER)
  message(FATAL_ERROR "Unable to determine the CUDA compiler from CUDAToolkit discovery.")
endif()

set(MRHASH_CUDA_CACHE_REFRESH FALSE)
if(DEFINED CMAKE_CUDA_COMPILER AND CMAKE_CUDA_COMPILER AND NOT CMAKE_CUDA_COMPILER STREQUAL MRHASH_CUDA_COMPILER)
  set(MRHASH_CUDA_CACHE_REFRESH TRUE)
endif()
if(DEFINED CUDA_TOOLKIT_ROOT_DIR AND CUDA_TOOLKIT_ROOT_DIR AND NOT CUDA_TOOLKIT_ROOT_DIR STREQUAL MRHASH_CUDA_ROOT)
  set(MRHASH_CUDA_CACHE_REFRESH TRUE)
endif()
set(MRHASH_CUDA_LEGACY_INCLUDE_DIR "")
foreach(_mrhash_cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
  if(EXISTS "${_mrhash_cuda_include_dir}/device_functions.h")
    set(MRHASH_CUDA_LEGACY_INCLUDE_DIR "${_mrhash_cuda_include_dir}")
    break()
  endif()
endforeach()

if(MRHASH_CUDA_CACHE_REFRESH)
  message(STATUS "Refreshing cached CUDA toolchain selection to ${MRHASH_CUDA_COMPILER}")
  mrhash_reset_findcuda_cache()
endif()

set(CMAKE_CUDA_COMPILER "${MRHASH_CUDA_COMPILER}" CACHE FILEPATH "CUDA compiler" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "${MRHASH_CUDA_ROOT}" CACHE PATH "CUDA toolkit root for legacy FindCUDA consumers" FORCE)
set(CUDA_TOOLKIT_TARGET_DIR "${MRHASH_CUDA_ROOT}" CACHE PATH "CUDA target dir for legacy FindCUDA consumers" FORCE)
set(CUDA_BIN_PATH "${MRHASH_CUDA_ROOT}" CACHE PATH "CUDA toolkit root for legacy FindCUDA consumers" FORCE)
set(CUDA_NVCC_EXECUTABLE "${MRHASH_CUDA_COMPILER}" CACHE FILEPATH "CUDA nvcc for legacy FindCUDA consumers" FORCE)
if(MRHASH_CUDA_LEGACY_INCLUDE_DIR)
  set(CUDA_TOOLKIT_INCLUDE "${MRHASH_CUDA_LEGACY_INCLUDE_DIR}" CACHE PATH "CUDA include dir for legacy FindCUDA consumers" FORCE)
endif()
set(CUDA_TOOLKIT_ROOT_DIR_INTERNAL "${MRHASH_CUDA_ROOT}" CACHE INTERNAL "Pinned CUDA toolkit root for legacy FindCUDA consumers" FORCE)
set(CUDA_TOOLKIT_TARGET_DIR_INTERNAL "${MRHASH_CUDA_ROOT}" CACHE INTERNAL "Pinned CUDA target dir for legacy FindCUDA consumers" FORCE)
set(ENV{CUDA_HOME} "${MRHASH_CUDA_ROOT}")
set(ENV{CUDA_PATH} "${MRHASH_CUDA_ROOT}")
set(ENV{CUDA_BIN_PATH} "${MRHASH_CUDA_ROOT}")
set(ENV{CUDA_NVCC_EXECUTABLE} "${MRHASH_CUDA_COMPILER}")
set(ENV{CUDACXX} "${MRHASH_CUDA_COMPILER}")

enable_language(CUDA)
message(STATUS "CUDA toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")

# Note: Suppressing warning 20012/20014 for float*/Eigen::Matrix type compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress 20012 -diag-suppress 20014")

if(DEFINED ENV{CUDA_ARCHITECTURES} AND NOT "$ENV{CUDA_ARCHITECTURES}" STREQUAL "")
  set(CUDA_ARCH_LIST "$ENV{CUDA_ARCHITECTURES}")
  message(STATUS "Using CUDA architectures from CUDA_ARCHITECTURES: ${CUDA_ARCH_LIST}")
else()
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")

  if(NOT CUDA_ARCH_LIST)
    set(CUDA_ARCH_LIST "75;80;86;89" CACHE STRING "CUDA architectures" FORCE)
    message(WARNING "Could not auto-detect CUDA architecture, using fallback: ${CUDA_ARCH_LIST}")
  else()
    message(STATUS "Detected CUDA architectures: ${CUDA_ARCH_LIST}")
  endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
foreach(MRHASH_CUDA_ARCH IN LISTS CUDA_ARCH_LIST)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_${MRHASH_CUDA_ARCH},code=sm_${MRHASH_CUDA_ARCH}>)
endforeach()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

include(FetchContent)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)
message(STATUS "Eigen Include Directories: ${eigen_SOURCE_DIR}")

FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.7.0
)
FetchContent_MakeAvailable(nanobind)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  # GIT_TAG yaml-cpp-0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
)
FetchContent_MakeAvailable(yaml-cpp)

# Libigl
# list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
# option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)
# include(libigl)
# igl_include(core)

# OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Torch setup
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# Define the absolute path to LibTorch
get_filename_component(PROJ_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}" ABSOLUTE)
set(LIBTORCH_DIR "${PROJ_ROOT_DIR}/third_party/libtorch")
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(Torch REQUIRED)
if(DEFINED ENV{CONDA_PREFIX} AND TARGET torch)
  get_target_property(MRHASH_TORCH_INCLUDE_DIRS torch INTERFACE_INCLUDE_DIRECTORIES)
  if(MRHASH_TORCH_INCLUDE_DIRS)
    list(FILTER MRHASH_TORCH_INCLUDE_DIRS EXCLUDE REGEX "site-packages/torch/include")
  endif()
  list(APPEND MRHASH_TORCH_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
  list(APPEND MRHASH_TORCH_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include/torch/csrc/api/include")
  set_target_properties(torch PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${MRHASH_TORCH_INCLUDE_DIRS}")
endif()
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${LIBTORCH_DIR}/lib")

add_subdirectory(third_party)

set(GEOM_LIB_NAME "libgeometric")
set(GS_LIB_NAME "libgsmodel")

add_subdirectory(src)
if(MRHASH_BUILD_TESTS)
  find_package(GTest REQUIRED)
  add_subdirectory(tests)
  enable_testing()
endif()
if(MRHASH_BUILD_APPS)
  add_subdirectory(apps)
endif()
