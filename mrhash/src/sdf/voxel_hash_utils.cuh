#pragma once
#include "cuda_algebra.cuh"
#include "cuda_utils.cuh"
#include <type_traits>

namespace cupanutils {
  namespace cugeoutils {

    inline constexpr float tudf_qef_eigenvalue_ratio = 0.0064f;
    inline constexpr float tudf_folded_variance_scale = 0.5f;

    struct Voxel {
      __host__ __device__ Voxel() {
        sdf    = 0.f;
        rgb    = make_uchar3(0, 0, 0);
        weight = 0;
      }
      float sdf         = 0.f; // signed distance function
      float sum_squared = 0.f;
      uchar3 rgb        = make_uchar3(0, 0, 0); // color
      uchar weight      = 0;                    // accumulated sdf weight

      auto cista_members() const {
        return std::tie(sdf, sum_squared, rgb, weight);
      }
    };

    enum class TSDFDirection : signed char {
      x_positive = 0,
      x_negative = 1,
      y_positive = 2,
      y_negative = 3,
      z_positive = 4,
      z_negative = 5,
      none = -1,
    };

    inline constexpr int directional_tsdf_count = 6;

#ifdef __CUDACC__
    __host__ __device__ inline float3 directionVector(const TSDFDirection direction) {
      switch (direction) {
        case TSDFDirection::x_positive:
          return make_float3(1.f, 0.f, 0.f);
        case TSDFDirection::x_negative:
          return make_float3(-1.f, 0.f, 0.f);
        case TSDFDirection::y_positive:
          return make_float3(0.f, 1.f, 0.f);
        case TSDFDirection::y_negative:
          return make_float3(0.f, -1.f, 0.f);
        case TSDFDirection::z_positive:
          return make_float3(0.f, 0.f, 1.f);
        case TSDFDirection::z_negative:
          return make_float3(0.f, 0.f, -1.f);
        case TSDFDirection::none:
          return make_float3(0.f);
      }
      return make_float3(0.f);
    }
#endif

    __host__ __device__ inline uchar twoSidedSurfaceWeight(const Voxel& voxel) {
      return voxel.weight;
    }

    __host__ __device__ inline float twoSidedSurfaceDistance(const Voxel& voxel) {
      if (voxel.sum_squared < 0.f)
        return voxel.sdf;
      const float variance = fmaxf(0.f, voxel.sum_squared - voxel.sdf * voxel.sdf);
      return sqrtf(fmaxf(0.f, voxel.sdf * voxel.sdf - tudf_folded_variance_scale * variance));
    }

#ifdef __CUDACC__
    struct TudfQef {
      CUDAMat3 ata = CUDAMat3::zero();
      float3 atb = make_float3(0.f);
      float3 normal_sum = make_float3(0.f);
      float3 reference_normal = make_float3(0.f);
      int plane_count = 0;
    };

    __device__ inline CUDAMat3 outerProduct(const float3 vector) {
      return CUDAMat3(vector.x * vector, vector.y * vector, vector.z * vector);
    }

    __device__ inline void addTudfPlane(TudfQef& qef, float3 normal, const float3 point) {
      if (qef.plane_count == 0)
        qef.reference_normal = normal;
      else if (dot(normal, qef.reference_normal) < 0.f)
        normal = -normal;
      const float offset = dot(normal, point);
      qef.ata += outerProduct(normal);
      qef.atb += offset * normal;
      qef.normal_sum += normal;
      ++qef.plane_count;
    }

    __device__ inline void rotateSymmetricTudfMatrix(
      CUDAMat3& matrix, CUDAMat3& vectors, const int first, const int second) {
      const float off_diagonal = matrix.at(first, second);
      if (fabsf(off_diagonal) <= 1e-7f)
        return;
      const float tau = (matrix.at(second, second) - matrix.at(first, first)) / (2.f * off_diagonal);
      const float tangent = copysignf(1.f, tau) / (fabsf(tau) + sqrtf(1.f + tau * tau));
      const float cosine = 1.f / sqrtf(1.f + tangent * tangent);
      const float sine = tangent * cosine;
      const float first_diagonal = matrix.at(first, first);
      const float second_diagonal = matrix.at(second, second);
      matrix.at(first, first) = first_diagonal - tangent * off_diagonal;
      matrix.at(second, second) = second_diagonal + tangent * off_diagonal;
      matrix.at(first, second) = 0.f;
      matrix.at(second, first) = 0.f;
      for (int axis = 0; axis < 3; ++axis) {
        if (axis != first && axis != second) {
          const float first_value = matrix.at(axis, first);
          const float second_value = matrix.at(axis, second);
          matrix.at(axis, first) = cosine * first_value - sine * second_value;
          matrix.at(first, axis) = matrix.at(axis, first);
          matrix.at(axis, second) = sine * first_value + cosine * second_value;
          matrix.at(second, axis) = matrix.at(axis, second);
        }
        const float first_vector = vectors.at(axis, first);
        const float second_vector = vectors.at(axis, second);
        vectors.at(axis, first) = cosine * first_vector - sine * second_vector;
        vectors.at(axis, second) = sine * first_vector + cosine * second_vector;
      }
    }

    __device__ inline int solveTudfQef(
      const TudfQef& qef,
      const float3 center,
      float3& point,
      float3& normal,
      float3& direction) {
      if (qef.plane_count == 0)
        return 0;
      CUDAMat3 matrix = qef.ata;
      CUDAMat3 vectors = CUDAMat3::identity();
      for (int sweep = 0; sweep < 6; ++sweep) {
        rotateSymmetricTudfMatrix(matrix, vectors, 0, 1);
        rotateSymmetricTudfMatrix(matrix, vectors, 0, 2);
        rotateSymmetricTudfMatrix(matrix, vectors, 1, 2);
      }
      const float3 centered_atb = qef.atb - qef.ata * center;
      const float maximum_eigenvalue =
        fmaxf(matrix.at(0, 0), fmaxf(matrix.at(1, 1), matrix.at(2, 2)));
      const float eigenvalue_threshold = maximum_eigenvalue * tudf_qef_eigenvalue_ratio;
      float3 displacement = make_float3(0.f);
      int rank = 0;
      direction = make_float3(0.f);
      for (int axis = 0; axis < 3; ++axis) {
        const float eigenvalue = matrix.at(axis, axis);
        if (eigenvalue <= eigenvalue_threshold) {
          direction = vectors.column(axis);
          continue;
        }
        const float3 eigenvector = vectors.column(axis);
        displacement += dot(eigenvector, centered_atb) / eigenvalue * eigenvector;
        ++rank;
      }
      point = center + displacement;
      normal = normalize(qef.normal_sum);
      return rank;
    }
#endif

    __host__ __device__ inline float surfaceDerivative(
      const Voxel& negative, const Voxel& center, const Voxel& positive, const uchar minimum_weight) {
      const bool negative_observed = negative.weight >= minimum_weight;
      const bool positive_observed = positive.weight >= minimum_weight;
      if (negative_observed && positive_observed)
        return 0.5f * (positive.sdf - negative.sdf);
      if (positive_observed)
        return positive.sdf - center.sdf;
      if (negative_observed)
        return center.sdf - negative.sdf;
      return 0.f;
    }

    template <typename T>
    struct is_voxel_derived : std::is_base_of<Voxel, T> {};

    struct HashEntry {
      __host__ __device__ HashEntry() {
        pos        = make_int3(0, 0, 0);
        offset     = NO_OFFSET;
        ptr        = FREE_ENTRY;
        resolution = 0;
        direction  = static_cast<signed char>(TSDFDirection::none);
      }
      int3 pos;    // hash position (lower left corner of SDFBlock))
      uint offset; // offset for collisions
      int ptr;     // pointer into heap to SDFBlock
      int resolution;
      signed char direction;
    };

    struct RayCastSample {
      float sdf;
      float alpha;
      uint weight;
    };

    struct Vertex {
      __host__ __device__ Vertex() {
        p = make_float3(0.f, 0.f, 0.f);
        c = make_float3(0.f, 0.f, 0.f);
      }
      float3 p;
      float3 c;
    };

    struct Triangle {
      __host__ __device__ Triangle() {
        v0 = Vertex();
        v1 = Vertex();
        v2 = Vertex();
      }
      Vertex v0;
      Vertex v1;
      Vertex v2;
    };

    __forceinline__ __device__ float3 virtualVoxelPosToWorld(const float virtual_voxel_size, const int3& voxel_pos) {
      return make_float3(voxel_pos.x * virtual_voxel_size, voxel_pos.y * virtual_voxel_size, voxel_pos.z * virtual_voxel_size);
    }

    __forceinline__ __device__ float3 virtualVoxelPosToWorld(const float virtual_voxel_size, const float3& voxel_pos) {
      return make_float3(voxel_pos.x * virtual_voxel_size, voxel_pos.y * virtual_voxel_size, voxel_pos.z * virtual_voxel_size);
    }

#ifdef __CUDACC__
    __forceinline__ __device__ int3 virtualVoxelPosToSDFBlock(const int3 virtual_voxel_pos,
                                                              const float virtual_voxel_size,
                                                              const float3 voxel_extents,
                                                              const int block_size = sdf_block_size) {
      int3 voxel_pos = virtual_voxel_pos;
      float epsilon  = 1e-5;
      if (virtual_voxel_pos.x < 0)
        voxel_pos.x -= (block_size - 1); // i.e voxelBlock virtual_voxel_size -1
      if (virtual_voxel_pos.y < 0)
        voxel_pos.y -= (block_size - 1);
      if (virtual_voxel_pos.z < 0)
        voxel_pos.z -= (block_size - 1);

      const float3 pw = virtualVoxelPosToWorld(virtual_voxel_size, voxel_pos);

      const float3 metric_block_size = make_float3((voxel_extents.x * sdf_block_size * virtual_voxel_size),
                                                   (voxel_extents.y * sdf_block_size * virtual_voxel_size),
                                                   (voxel_extents.z * sdf_block_size * virtual_voxel_size));

      int3 sdf_block_pos;
      sdf_block_pos.x =
        (pw.x >= 0) ? floorf((pw.x + epsilon) / metric_block_size.x) : ceilf((pw.x - epsilon) / metric_block_size.x);
      sdf_block_pos.y =
        (pw.y >= 0) ? floorf((pw.y + epsilon) / metric_block_size.y) : ceilf((pw.y - epsilon) / metric_block_size.y);
      sdf_block_pos.z =
        (pw.z >= 0) ? floorf((pw.z + epsilon) / metric_block_size.z) : ceilf((pw.z - epsilon) / metric_block_size.z);

      return sdf_block_pos;
    }
#endif

    __forceinline__ __host__ __device__ uint linearizeVoxelPos(const int3& pos, const int block_size = sdf_block_size) {
      return pos.z * block_size * block_size + pos.y * block_size + pos.x;
    }

    __forceinline__ __host__ __device__ uint virtualVoxelPosToSDFBlockIndex(const int3& virtual_voxel_pos,
                                                                            const int block_size = sdf_block_size) {
      const int scaling_factor = sdf_block_size / block_size;
      int3 local_voxel_pos     = make_int3(
        virtual_voxel_pos.x % sdf_block_size, virtual_voxel_pos.y % sdf_block_size, virtual_voxel_pos.z % sdf_block_size);

      if (local_voxel_pos.x < 0)
        local_voxel_pos.x += sdf_block_size;
      if (local_voxel_pos.y < 0)
        local_voxel_pos.y += sdf_block_size;
      if (local_voxel_pos.z < 0)
        local_voxel_pos.z += sdf_block_size;

      local_voxel_pos.x /= scaling_factor;
      local_voxel_pos.y /= scaling_factor;
      local_voxel_pos.z /= scaling_factor;

      return linearizeVoxelPos(local_voxel_pos, sdf_block_size);
    }

    __forceinline__ __host__ __device__ uint3 delinearizeVoxelPos(const uint index, const int block_size = sdf_block_size) {
      const uint size2 = block_size * block_size;
      const uint x     = (index % block_size);
      const uint y     = ((index % (size2)) / block_size);
      const uint z     = (index / (size2));
      return make_uint3(x, y, z);
    }

    __forceinline__ __host__ __device__ int3 SDFBlockToVirtualVoxelPos(const int3& sdf_block) {
      return make_int3(sdf_block.x * sdf_block_size, sdf_block.y * sdf_block_size, sdf_block.z * sdf_block_size);
    }

#ifdef __CUDACC__
    __forceinline__ __host__ __device__ int3 worldPointToVirtualVoxelPos(const float virtual_voxel_size, const float3& point) {
      float3 p       = point / virtual_voxel_size;
      float epsilon  = 1e-5;
      float3 aprox_p = p + make_float3(sign(p)) * 0.5f;
      aprox_p.x      = (aprox_p.x >= 0) ? floorf(aprox_p.x + epsilon) : ceilf(aprox_p.x - epsilon);
      aprox_p.y      = (aprox_p.y >= 0) ? floorf(aprox_p.y + epsilon) : ceilf(aprox_p.y - epsilon);
      aprox_p.z      = (aprox_p.z >= 0) ? floorf(aprox_p.z + epsilon) : ceilf(aprox_p.z - epsilon);
      return make_int3(aprox_p);
    }

    __forceinline__ __device__ float3 worldPointToVirtualVoxelPosFloat(const float virtual_voxel_size, const float3& point) {
      return point / virtual_voxel_size;
    }

    __forceinline__ __device__ int3 worldPointToSDFBlock(const float virtual_voxel_size,
                                                         const float3 voxel_extents,
                                                         const float3& point) {
      return virtualVoxelPosToSDFBlock(worldPointToVirtualVoxelPos(virtual_voxel_size, point), virtual_voxel_size, voxel_extents);
    }

    __forceinline__ __device__ float3 SDFBlockToWorldPoint(const float virtual_voxel_size, const int3& sdf_block) {
      return virtualVoxelPosToWorld(virtual_voxel_size, SDFBlockToVirtualVoxelPos(sdf_block));
    }

    template <typename T>
    // ! merges two voxels (v0 the currently stored voxel, v1 is the input voxel)
    __forceinline__ __device__ void combineVoxel(const T& v0, const T& v1, const int integration_weight_max, T& out) {
      // interpolate colors
      const float3 c0  = make_float3(v0.rgb.x, v0.rgb.y, v0.rgb.z);
      const float3 c1  = make_float3(v1.rgb.x, v1.rgb.y, v1.rgb.z);
      const float3 res = 0.5f * c0 + 0.5f * c1;
      out.rgb.x        = (uchar) (res.x + 0.5f);
      out.rgb.y        = (uchar) (res.y + 0.5f);
      out.rgb.z        = (uchar) (res.z + 0.5f);

      // merge sdf and weight
      out.sdf    = (v0.sdf * v0.weight + v1.sdf * v1.weight) / (v0.weight + v1.weight);
      out.weight = min(integration_weight_max, v0.weight + v1.weight);
    }

    template <typename T>
    __forceinline__ __host__ __device__ void combineTwoSidedSurfaceVoxel(
      const T& stored, const T& input, const int integration_weight_max, T& output) {
      output = stored;
      const int weight = stored.weight + input.weight;
      output.sdf = (stored.sdf * stored.weight + input.sdf * input.weight) / weight;
      output.sum_squared =
        (stored.sum_squared * stored.weight + input.sum_squared * input.weight) / weight;
      output.weight = min(integration_weight_max, weight);
    }

    //! returns the truncation of the SDF for a given distance value
    __forceinline__ __host__ __device__ float
    getTruncation(const float z, const float sdf_truncation, const float sdf_truncation_scale) {
      return sdf_truncation + sdf_truncation_scale * z;
    }

    // ! delete stuff, voxel and hash entries
    __forceinline__ __device__ void deleteHashEntry(HashEntry& hashEntry) {
      hashEntry.pos    = make_int3(0, 0, 0);
      hashEntry.offset = NO_OFFSET;
      hashEntry.ptr    = FREE_ENTRY;
      hashEntry.direction = static_cast<signed char>(TSDFDirection::none);
    }

    template <typename T>
    __forceinline__ __device__ void deleteVoxel(T& v) {
      v.sdf         = 0.f;
      v.rgb         = make_uchar3(0, 0, 0);
      v.sum_squared = 0.f;
      v.weight      = 0;
    }

    __forceinline__ __device__ uint linearizeChunkPos(const int3& chunk_pos,
                                                      const int3& min_grid_pos,
                                                      const int3& grid_dimensions) {
      const int3 p = chunk_pos - min_grid_pos;
      return p.z * grid_dimensions.x * grid_dimensions.y + p.y * grid_dimensions.x + p.x;
    }

    __forceinline__ __device__ int3 worldToChunks(const float3& pw, const float3& voxel_extents) {
      float3 p;
      p.x = pw.x / voxel_extents.x;
      p.y = pw.y / voxel_extents.y;
      p.z = pw.z / voxel_extents.z;

      float3 s;
      s.x = (float) sign(p.x);
      s.y = (float) sign(p.y);
      s.z = (float) sign(p.z);

      return make_int3(p + s * 0.5f);
    }

#endif

  } // namespace cugeoutils
} // namespace cupanutils
