#pragma once
#include "cuda_math.cuh"
#include <ostream>

namespace cupanutils {
  namespace cugeoutils {

    // 3x3 mat for CUDA computation
    struct CUDAMat3 {
      // def constructor, leaves the matrix uninitialized
      __forceinline__ __host__ __device__ CUDAMat3() {
      }

      __forceinline__ __host__ __device__ CUDAMat3(const float3& first, const float3& second, const float3& third) :
        row0(first), row1(second), row2(third) {
      }

      // copy constructor
      __forceinline__ __host__ __device__ CUDAMat3(const CUDAMat3& other) :
        row0(other.row0), row1(other.row1), row2(other.row2) {
      }

      // constructs the matrix from an array-like matrix object (works with Eigen matrices)
      template <typename T>
      __host__ explicit CUDAMat3(const T& matrix) {
        row0.x = matrix(0, 0);
        row0.y = matrix(0, 1);
        row0.z = matrix(0, 2);
        row1.x = matrix(1, 0);
        row1.y = matrix(1, 1);
        row1.z = matrix(1, 2);
        row2.x = matrix(2, 0);
        row2.y = matrix(2, 1);
        row2.z = matrix(2, 2);
      }

      // constructs the matrix from an array object, row major
      __host__ explicit CUDAMat3(float* arr) {
        row0.x = arr[0];
        row0.y = arr[1];
        row0.z = arr[2];
        row1.x = arr[3];
        row1.y = arr[4];
        row1.z = arr[5];
        row2.x = arr[6];
        row2.y = arr[7];
        row2.z = arr[8];
      }

      // assignment operator
      __forceinline__ __host__ __device__ CUDAMat3& operator=(const CUDAMat3& other) {
        this->row0 = other.row0;
        this->row1 = other.row1;
        this->row2 = other.row2;
        return *this;
      }

// define algebraic operations only in device code
#ifdef __CUDACC__
      __forceinline__ __device__ static CUDAMat3 zero() {
        return CUDAMat3(make_float3(0.f), make_float3(0.f), make_float3(0.f));
      }

      __forceinline__ __device__ static CUDAMat3 identity() {
        return CUDAMat3(make_float3(1.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), make_float3(0.f, 0.f, 1.f));
      }

      __forceinline__ __device__ float& at(const int row, const int column) {
        return (&row0.x)[3 * row + column];
      }

      __forceinline__ __device__ const float& at(const int row, const int column) const {
        return (&row0.x)[3 * row + column];
      }

      __forceinline__ __device__ float3 column(const int column_index) const {
        return make_float3(at(0, column_index), at(1, column_index), at(2, column_index));
      }

      __forceinline__ __device__ CUDAMat3 transpose() const {
        return CUDAMat3(column(0), column(1), column(2));
      }

      __forceinline__ __device__ float3 operator*(const float3& vector) const {
        return make_float3(dot(row0, vector), dot(row1, vector), dot(row2, vector));
      }

      __forceinline__ __device__ CUDAMat3& operator+=(const CUDAMat3& matrix) {
        row0 += matrix.row0;
        row1 += matrix.row1;
        row2 += matrix.row2;
        return *this;
      }

      // matrix-matrix multiplication
      __forceinline__ __device__ CUDAMat3 operator*(const CUDAMat3& mat) const {
        const CUDAMat3 transposed = mat.transpose();
        return CUDAMat3(transposed * row0, transposed * row1, transposed * row2);
      }
#endif

      // row-wise storage.
      float3 row0;
      float3 row1;
      float3 row2;
    };

    struct CUDAMatSE3 {
      // def constructor, leaves the matrix uninitialized
      __forceinline__ __host__ __device__ CUDAMatSE3() {
      }

      // copy constructor
      __forceinline__ __host__ CUDAMatSE3(const CUDAMatSE3& other) : rotation(other.rotation), translation(other.translation) {
      }

      template <typename T>
      __host__ explicit CUDAMatSE3(const T& matrix) {
        rotation.row0.x = matrix(0, 0);
        rotation.row0.y = matrix(0, 1);
        rotation.row0.z = matrix(0, 2);
        rotation.row1.x = matrix(1, 0);
        rotation.row1.y = matrix(1, 1);
        rotation.row1.z = matrix(1, 2);
        rotation.row2.x = matrix(2, 0);
        rotation.row2.y = matrix(2, 1);
        rotation.row2.z = matrix(2, 2);
        translation.x   = matrix(0, 3);
        translation.y   = matrix(1, 3);
        translation.z   = matrix(2, 3);
      }

      // assignment operator
      __forceinline__ __host__ __device__ CUDAMatSE3& operator=(const CUDAMatSE3& other) {
        this->rotation    = other.rotation;
        this->translation = other.translation;
        return *this;
      }

// define operators only in device code
// we have eigen in host
#ifdef __CUDACC__

      __forceinline__ __device__ CUDAMatSE3 inverse() const {
        CUDAMatSE3 res;
        res.rotation    = this->rotation.transpose();
        res.translation = res.rotation * this->translation;
        res.translation = -res.translation;
        return res;
      }

      // matrix-vector multiplication.
      __forceinline__ __device__ float3 operator*(const float3& point) const {
        return rotation * point + translation;
      }

      __forceinline__ __device__ CUDAMatSE3 operator*(const CUDAMatSE3& other) const {
        CUDAMatSE3 res;
        res.rotation    = this->rotation * other.rotation;
        res.translation = this->rotation * other.translation + this->translation;
        return res;
      }

#endif

      friend std::ostream& operator<<(std::ostream& out, const CUDAMatSE3& m) {
        return out << m.rotation.row0.x << " " << m.rotation.row0.y << " " << m.rotation.row0.z << " " << m.translation.x << " "
                   << m.rotation.row1.x << " " << m.rotation.row1.y << " " << m.rotation.row1.z << " " << m.translation.y << " "
                   << m.rotation.row2.x << " " << m.rotation.row2.y << " " << m.rotation.row2.z << " " << m.translation.z << " "
                   << 0 << " " << 0 << " " << 0 << " " << 1 << std::endl;
      }

      CUDAMat3 rotation;
      float3 translation;
    };

  } // namespace cugeoutils
} // namespace cupanutils
