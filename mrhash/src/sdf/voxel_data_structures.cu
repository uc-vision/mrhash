#include "voxel_data_structures.cuh"

#include <cfloat>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace cupanutils {
  namespace cugeoutils {

    constexpr float tudf_min_sample_distance_voxels = 0.25f;
    constexpr float tudf_min_gradient = 0.25f;
    constexpr float tudf_min_interpolation_support = 1.f;
    constexpr float tudf_projection_distance_voxels = 0.75f;
    constexpr float tudf_max_relative_variance = 1.f;
    constexpr float tudf_half_cell_diagonal = 0.8660254037844386f;
    constexpr float tudf_promoted_candidate_radius = 0.4f;
    constexpr float tudf_completed_candidate_radius = 0.5f;
    constexpr int tudf_max_connector_coordinate_span = 5;
    constexpr float tudf_min_connector_normal_alignment = 0.5f;
    constexpr float tudf_min_fill_normal_alignment = 0.7071067811865476f;
    constexpr int tudf_min_completion_neighbors = 3;
    constexpr uint tudf_extraction_threads = 256;

    enum class TudfSampleSupport : int {
      completed = -2,
      promoted = -1,
      observed = 0,
    };

    enum class TudfFieldValue {
      distance,
      mean,
      second_moment,
    };

    __global__ void initializeVoxelContainerBuffersKernel(const uint num_sdf_blocks,
                                                           const uint total_size,
                                                           const uint hash_num_buckets,
                                                           uint* heap_high,
                                                           uint* heap_low,
                                                           HashEntry* hash_table,
                                                           HashEntry* compact_hash_table,
                                                           int* hash_table_bucket_mutex) {
      const uint index = blockIdx.x * blockDim.x + threadIdx.x;
      const uint low_heap_size = num_sdf_blocks * octree_branching_factor;
      if (index < num_sdf_blocks) {
        heap_high[index] = num_sdf_blocks - 1 - index;
        compact_hash_table[index] = HashEntry();
      }
      if (index < low_heap_size)
        heap_low[index] = low_heap_size;
      if (index < total_size)
        hash_table[index] = HashEntry();
      if (index < hash_num_buckets)
        hash_table_bucket_mutex[index] = FREE_ENTRY;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::resetBuffers() {
      const uint initialization_size = std::max(total_size_, num_sdf_blocks_ * octree_branching_factor);
      const dim3 threads_per_block(n_threads_reduce_hashtable, 1);
      const dim3 blocks((initialization_size + threads_per_block.x - 1) / threads_per_block.x, 1);
      initializeVoxelContainerBuffersKernel<<<blocks, threads_per_block>>>(num_sdf_blocks_,
                                                                           total_size_,
                                                                           hash_num_buckets_,
                                                                           d_heap_high_,
                                                                           d_heap_low_,
                                                                           d_hashTable_,
                                                                           d_compactHashTable_,
                                                                           d_hashTableBucketMutex_);
      CUDA_CHECK(cudaMemset(d_SDFBlocks_, 0, sizeof(T) * num_sdf_blocks_ * voxel_block_volume_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T>
    __global__ void resetCompactHashTableKernel(const VoxelContainer<T>* container) {
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= container->num_sdf_blocks_)
        return;
      deleteHashEntry(container->d_compactHashTable_[idx]);
    }

    template <typename T>
    __global__ void resetHashBucketMutexKernel(const VoxelContainer<T>* container) {
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= container->hash_num_buckets_)
        return;
      container->d_hashTableBucketMutex_[idx] = FREE_ENTRY;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::resetHashBucketMutex() {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      const dim3 n_blocks((hash_num_buckets_ + threads_per_block.x - 1) / threads_per_block.x, 1);
      resetHashBucketMutexKernel<<<n_blocks, threads_per_block>>>(d_instance_);
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::consumeHeapHigh() {
      int addr = atomicSub(&d_heapCounterHigh_[0], 1);

      if (addr < 0)
        return -1;

      return d_heap_high_[addr];
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::consumeHeapLow() {
      int addr = atomicSub(&d_heapCounterLow_[0], 1);

      if (addr < 0)
        return -1;

      return d_heap_low_[addr];
    }

    template <typename T>
    __device__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::appendHeapHigh(const uint ptr) {
      int addr               = atomicAdd(&d_heapCounterHigh_[0], 1);
      d_heap_high_[addr + 1] = ptr;
    }

    template <typename T>
    __device__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::appendHeapLow(const uint ptr) {
      int addr              = atomicAdd(&d_heapCounterLow_[0], 1);
      d_heap_low_[addr + 1] = ptr;
    }

    template <typename T>
    // ! fast-approx check to see if sdf block is in camera frustum
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::isSDFBlockInCameraFrustumApprox(const Camera* camera,
                                                                                                     const int3& sdf_block) {
      for (int i = 0; i < vertex_offset_camera; i++) {
        int3 vertex            = vert_offset[i];
        int3 virtual_voxel_pos = SDFBlockToVirtualVoxelPos(sdf_block) + vertex;
        float3 world_point     = virtualVoxelPosToWorld(virtual_voxel_size_, virtual_voxel_pos);
        if (camera->isInCameraFrustumApprox(world_point))
          return true;
      }
      return false;
    }

    template <typename T>
    __device__ HashEntry
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHashEntry(const int3& sdf_block) const {
      HashEntry entry;
      entry.pos    = sdf_block;
      entry.ptr    = FREE_ENTRY;
      entry.offset = 0;

      uint64_t h = calculateHash(sdf_block);

      for (uint i = 0; i < hash_bucket_size_; ++i) {
        const uint hash_idx = h * hash_bucket_size_ + i;
        HashEntry curr      = d_hashTable_[hash_idx];
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }
      }

#ifdef RESOLVE_COLLISION
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;

      int max_iter            = 0;
      int max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) { // traverse list until end

        curr = d_hashTable_[i];

        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }

        if (curr.offset == 0) {
          break;
        }

        i = idx_last_entry_in_bucket + curr.offset;
        i %= (hash_bucket_size_ * hash_num_buckets_);

        max_iter++;
      }
#endif

      return entry;
    }

    template <typename T>
    __device__ HashEntry
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getHashEntryReintegrate(const int3& sdf_block) const {
      HashEntry entry;
      entry.pos    = sdf_block;
      entry.ptr    = FREE_ENTRY;
      entry.offset = 0;

      uint64_t h = calculateHash(sdf_block);

      for (uint i = 0; i < d_num_reintegrate_[0]; ++i) {
        const uint hash_idx = d_reintegrate_[i];
        HashEntry curr      = d_hashTable_[hash_idx];
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          return curr;
        }
      }

      return entry;
    }

    template <typename T>
    __device__ uint64_t
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::calculateHash(const int3& virtual_voxel_pos) const {
      unsigned int x = (unsigned int) virtual_voxel_pos.x;
      unsigned int y = (unsigned int) virtual_voxel_pos.y;
      unsigned int z = (unsigned int) virtual_voxel_pos.z;
      int res        = ((x * (unsigned int) p0) ^ (y * (unsigned int) p1) ^ (z * (unsigned int) p2)) % hash_num_buckets_;
      if (res < 0)
        res += hash_num_buckets_;
      return res;
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const int3& virtual_voxel_pos) const {
      T v;
      const HashEntry& entry = getHashEntry(virtualVoxelPosToSDFBlock(virtual_voxel_pos, virtual_voxel_size_, voxel_extents_));
      if (entry.ptr == FREE_ENTRY) {
        deleteVoxel<T>(v);
        return v;
      } else {
        const int scaling_factor = 1 << entry.resolution;

        v = d_SDFBlocks_[entry.ptr + virtualVoxelPosToSDFBlockIndex(virtual_voxel_pos, sdf_block_size / scaling_factor)];
        return v;
      }
    }
    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const int3& virtual_voxel_pos,
                                                                                           int& block_res) const {
      T v;
      const HashEntry& entry = getHashEntry(virtualVoxelPosToSDFBlock(virtual_voxel_pos, virtual_voxel_size_, voxel_extents_));
      if (entry.ptr == FREE_ENTRY) {
        deleteVoxel<T>(v);
        return v;
      } else {
        const int scaling_factor = 1 << entry.resolution;
        block_res                = entry.resolution;

        uint voxel_index                   = virtualVoxelPosToSDFBlockIndex(virtual_voxel_pos, sdf_block_size / scaling_factor);
        v                                  = d_SDFBlocks_[entry.ptr + voxel_index];
        uint3 delinearized_local_voxel_pos = delinearizeVoxelPos(voxel_index, sdf_block_size / scaling_factor);
        int3 delinearized_voxel_pos        = SDFBlockToVirtualVoxelPos(entry.pos) + scaling_factor * delinearized_local_voxel_pos;
        return v;
      }
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const float3& pos) const {
      return getVoxel(worldPointToVirtualVoxelPos(virtual_voxel_size_, pos));
    }

    template <typename T>
    __device__ T VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxel(const float3& pos,
                                                                                           int& block_res) const {
      return getVoxel(worldPointToVirtualVoxelPos(virtual_voxel_size_, pos), block_res);
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const HashEntry& entry) const {
      const int scale = 1 << (finest_block_log2_dim - entry.resolution);
      return scale * scale * scale;
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const int3& pos) const {
      const HashEntry& entry = getHashEntry(pos);
      return getNumVoxels(entry);
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getNumVoxels(const float3& pos) const {
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(virtual_voxel_size_, voxel_extents_, pos));
      return getNumVoxels(entry);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const HashEntry& entry) const {
      return virtual_voxel_size_ * (1 << entry.resolution);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const int3& pos) const {
      const HashEntry& entry = getHashEntry(pos);
      return getVoxelSize(entry);
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelSize(const float3& pos) const {
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(virtual_voxel_size_, voxel_extents_, pos));
      return getVoxelSize(entry);
    }

    template <typename T>
    __global__ void getVoxelWeightKernel(const float3 pw, uchar* weight, VoxelContainer<T>* container) {
      const T& voxel = container->getVoxel(pw);
      if (voxel.weight > 0) {
        weight[0] = voxel.weight;
      }
    }

    template <typename T>
    uchar VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::getVoxelWeight(const Eigen::Vector3f& pw) {
      getVoxelWeightKernel<<<1, 1>>>(Eig2CUDA(pw), d_weight_, d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      uchar weight = 0;
      CUDA_CHECK(cudaMemcpy(&weight, &d_weight_[0], sizeof(uchar), cudaMemcpyDeviceToHost));
      return weight;
    }

    template <typename T>
    __device__ bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::trilinearInterpolation(const float3& pos,
                                                                                                            float& dist) const {
      float voxel_size       = getVoxelSize(pos);
      const float3 pos_dual  = pos - make_float3(voxel_size * 0.5f);
      const HashEntry& entry = getHashEntry(worldPointToSDFBlock(voxel_size, voxel_extents_, pos));

      const int base_resolution = entry.resolution;
      dist                      = 0.f;
      const float pos_sdf       = getVoxel(pos_dual).sdf;

      const float x0 = pos_dual.x;
      const float y0 = pos_dual.y;
      const float z0 = pos_dual.z;
      float x1       = x0;
      float y1       = y0;
      float z1       = z0;

      int resolution = 0;

      float sdf[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

      for (int i = 0; i < 8; ++i) {
        int dx = i & 1;
        int dy = (i >> 1) & 1;
        int dz = (i >> 2) & 1;

        const float3 voxel_pos = pos_dual + make_float3(dx, dy, dz) * voxel_size;

        const T& v = getVoxel(voxel_pos, resolution);

        if (!v.weight)
          return false;

        if (resolution > base_resolution) {
          const float new_voxel_size = voxel_size * 2;

          const float3 new_voxel_pos = pos - make_float3(new_voxel_size * 0.5f) + make_float3(dx, dy, dz) * new_voxel_size;
          // interpolate sdf at voxel_pos between pos_dual and new_voxel_pos based on their coordinates
          const float new_voxel_pos_sdf = getVoxel(new_voxel_pos).sdf;
          float alpha                   = 0.5f;
          const float dist_inside       = (1 - alpha) * pos_sdf + alpha * new_voxel_pos_sdf;

          sdf[i] = dist_inside;
        }

        else {
          sdf[i] = v.sdf;
        }

        if (voxel_pos.x > x1)
          x1 = voxel_pos.x;
        if (voxel_pos.y > y1)
          y1 = voxel_pos.y;
        if (voxel_pos.z > z1)
          z1 = voxel_pos.z;

        resolution = 0;
      }

      // Avoid division by zero in delta calculation
      const float dx = (x1 - x0) > 1e-6f ? (pos.x - x0) / (x1 - x0) : 0.5f;
      const float dy = (y1 - y0) > 1e-6f ? (pos.y - y0) / (y1 - y0) : 0.5f;
      const float dz = (z1 - z0) > 1e-6f ? (pos.z - z0) / (z1 - z0) : 0.5f;
      float3 delta   = {dx, dy, dz};

      float c[8] = {sdf[0],
                    (sdf[1] - sdf[0]),
                    (sdf[2] - sdf[0]),
                    (sdf[4] - sdf[0]),
                    (sdf[3] - sdf[2] - sdf[1] + sdf[0]),
                    (sdf[6] - sdf[4] - sdf[2] + sdf[0]),
                    (sdf[5] - sdf[4] - sdf[1] + sdf[0]),
                    (sdf[7] - sdf[6] - sdf[5] - sdf[3] + sdf[1] + sdf[4] + sdf[2] - sdf[0])};

      dist = c[0] + c[1] * delta.x + c[2] * delta.y + c[3] * delta.z + c[4] * delta.x * delta.y + c[5] * delta.y * delta.z +
             c[6] * delta.x * delta.z + c[7] * delta.x * delta.y * delta.z;

      return true;
    }

    template <typename T>
    __device__ float VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::findIntersectionLinear(float t_near,
                                                                                                             float t_far,
                                                                                                             float d_near,
                                                                                                             float d_far) const {
      return t_near + (d_near / (d_near - d_far)) * (t_far - t_near);
    }

    template <typename T>
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::findIntersectionBisection(const Camera* camera,
                                                                                               const float3& world_cam_pos,
                                                                                               const float3& world_dir,
                                                                                               float d0,
                                                                                               float r0,
                                                                                               float d1,
                                                                                               float r1,
                                                                                               float& alpha) const {
      float a      = r0;
      float a_dist = d0;
      float b      = r1;
      float b_dist = d1;
      float c      = 0.f;

#pragma unroll 1
      for (uint i = 0; i < n_iteration_bisection; ++i) {
        c = findIntersectionLinear(a, b, a_dist, b_dist);
        float c_dist;
        if (!trilinearInterpolation(world_cam_pos + c * world_dir, c_dist)) {
          return false;
        }
        if (a_dist * c_dist > 0) {
          a      = c;
          a_dist = c_dist;
        } else {
          b      = c;
          b_dist = c_dist;
        }
      }

      alpha = c;

      return true;
    }

    template <typename T>
    __device__ uint8_t VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::checkVoxelNeighbors(const float3& pw) {
      const int3& pi             = worldPointToVirtualVoxelPos(virtual_voxel_size_, pw);
      const int3& block_pos      = virtualVoxelPosToSDFBlock(pi, virtual_voxel_size_, voxel_extents_);
      const int voxel_resolution = getHashEntry(block_pos).resolution;
      const int scaling_factor   = 1 << voxel_resolution;

      uint8_t bit_mask = 0;
      for (int i = 0; i < neighbor_voxels; ++i) {
        const int3& voxel_neighbor_pos = pi + scaling_factor * neighbor_offsets[i];
        const int3& block_neighbor_pos = virtualVoxelPosToSDFBlock(voxel_neighbor_pos, virtual_voxel_size_, voxel_extents_);
        const auto neighbor_entry      = getHashEntry(block_neighbor_pos);

        if (neighbor_entry.ptr != FREE_ENTRY && neighbor_entry.resolution != voxel_resolution) {
          bit_mask |= (1 << i);
        }
      }
      return bit_mask;
    }

    template <typename T>
    __global__ void flatAndReduceHashTableKernel(const Camera* camera, VoxelContainer<T>* container) {
      const int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= container->total_size_)
        return;

      __shared__ int local_counter;
      if (threadIdx.x == 0)
        local_counter = 0;
      __syncthreads();

      int local_addr         = -1;
      const HashEntry& entry = container->d_hashTable_[idx];
      if (entry.ptr != FREE_ENTRY && container->isSDFBlockInCameraFrustumApprox(camera, entry.pos)) {
        local_addr = atomicAdd(&local_counter, 1);
      }

      __syncthreads();

      __shared__ int global_addr;
      if (threadIdx.x == 0 && local_counter > 0) {
        global_addr = atomicAdd(&container->d_compactHashCounter_[0], local_counter);
      }
      __syncthreads();

      if (local_addr != -1) {
        const uint addr                      = global_addr + local_addr;
        container->d_compactHashTable_[addr] = entry;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::flatAndReduceHashTable(const Camera& camera) {
      const dim3 threads_per_block(n_threads_reduce_hashtable, 1);
      const dim3 n_blocks((total_size_ + threads_per_block.x - 1) / threads_per_block.x, 1);
      const dim3 compact_blocks((num_sdf_blocks_ + threads_per_block.x - 1) / threads_per_block.x, 1);

      resetCompactHashTableKernel<<<compact_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaMemset(d_compactHashCounter_, 0, sizeof(int)));

      flatAndReduceHashTableKernel<<<n_blocks, threads_per_block>>>(camera.deviceInstance(), d_instance_);
      CUDA_CHECK(cudaMemcpy(&current_occupied_blocks_, &d_compactHashCounter_[0], sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(VoxelContainer), cudaMemcpyHostToDevice));
    }

    template <typename T>
    __global__ void flatAndReduceHashTableKernel(VoxelContainer<T>* container) {
      const int idx = blockDim.x * blockIdx.x + threadIdx.x;
      if (idx >= container->total_size_)
        return;

      __shared__ int local_counter;
      if (threadIdx.x == 0)
        local_counter = 0;
      __syncthreads();

      // local address within block
      int local_addr         = -1;
      const HashEntry& entry = container->d_hashTable_[idx];
      if (entry.ptr != FREE_ENTRY) {
        local_addr = atomicAdd(&local_counter, 1);
      }

      __syncthreads();

      // update global count of occupied blocks
      __shared__ int global_addr;
      if (threadIdx.x == 0 && local_counter > 0) {
        global_addr = atomicAdd(&container->d_compactHashCounter_[0], local_counter);
      }
      __syncthreads();

      // assign local address and copy
      if (local_addr != -1) {
        const uint addr                      = global_addr + local_addr;
        container->d_compactHashTable_[addr] = entry;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::flatAndReduceHashTable() {
      const dim3 threads_per_block(n_threads_reduce_hashtable, 1);
      const dim3 n_blocks((total_size_ + threads_per_block.x - 1) / threads_per_block.x, 1);
      const dim3 compact_blocks((num_sdf_blocks_ + threads_per_block.x - 1) / threads_per_block.x, 1);

      resetCompactHashTableKernel<<<compact_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaMemset(d_compactHashCounter_, 0, sizeof(int)));

      flatAndReduceHashTableKernel<<<n_blocks, threads_per_block>>>(d_instance_);
      CUDA_CHECK(cudaMemcpy(&current_occupied_blocks_, &d_compactHashCounter_[0], sizeof(uint), cudaMemcpyDeviceToHost));
      // copy current ptr to gpu, make sure current_occupied_blocks_ is updated in gpu
      CUDA_CHECK(cudaMemcpy(d_instance_, this, sizeof(VoxelContainer), cudaMemcpyHostToDevice));
    }

    template <typename T>
    __global__ void surfaceVoxelCountsKernel(const VoxelContainer<T>* container, const float surface_band, uint* counts) {
      const uint block_idx = blockIdx.x;
      if (block_idx >= container->current_occupied_blocks_)
        return;

      __shared__ uint count;
      if (threadIdx.x == 0)
        count = 0;
      __syncthreads();

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      const uint voxel_idx   = threadIdx.x;
      if (voxel_idx < container->getNumVoxels(entry)) {
        const T& voxel = container->d_SDFBlocks_[entry.ptr + voxel_idx];
        if (voxel.weight > 0 && fabsf(voxel.sdf) <= surface_band)
          atomicAdd(&count, 1);
      }

      __syncthreads();
      if (threadIdx.x == 0)
        counts[block_idx] = count;
    }

    template <typename T>
    __global__ void surfaceVoxelFillKernel(const VoxelContainer<T>* container,
                                           const uint* offsets,
                                           const float surface_band,
                                           float* voxels) {
      const uint block_idx = blockIdx.x;
      if (block_idx >= container->current_occupied_blocks_)
        return;

      __shared__ uint local_count;
      if (threadIdx.x == 0)
        local_count = 0;
      __syncthreads();

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      const uint voxel_idx   = threadIdx.x;
      if (voxel_idx < container->getNumVoxels(entry)) {
        const T& voxel = container->d_SDFBlocks_[entry.ptr + voxel_idx];
        if (voxel.weight > 0 && fabsf(voxel.sdf) <= surface_band) {
          const uint output_idx = offsets[block_idx] + atomicAdd(&local_count, 1);
          const int scale       = 1 << entry.resolution;
          const int3 base       = SDFBlockToVirtualVoxelPos(entry.pos);
          const int3 local =
            scale * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scale));
          const float3 point = virtualVoxelPosToWorld(container->virtual_voxel_size_, base + local);
          voxels[output_idx * 5 + 0] = point.x;
          voxels[output_idx * 5 + 1] = point.y;
          voxels[output_idx * 5 + 2] = point.z;
          voxels[output_idx * 5 + 3] = voxel.sdf;
          voxels[output_idx * 5 + 4] = static_cast<float>(voxel.weight);
        }
      }
    }

    template <typename T>
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::surfaceVoxels(const float surface_band) {
      flatAndReduceHashTable();
      if (current_occupied_blocks_ == 0)
        return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(0, 5);

      uint* d_counts  = nullptr;
      uint* d_offsets = nullptr;
      float* d_voxels = nullptr;
      CUDA_CHECK(cudaMalloc((void**) &d_counts, sizeof(uint) * current_occupied_blocks_));
      CUDA_CHECK(cudaMalloc((void**) &d_offsets, sizeof(uint) * (current_occupied_blocks_ + 1)));
      CUDA_CHECK(cudaMemset(d_offsets, 0, sizeof(uint)));

      surfaceVoxelCountsKernel<<<current_occupied_blocks_, voxel_block_volume_>>>(d_instance_, surface_band, d_counts);
      CUDA_CHECK(cudaDeviceSynchronize());

      thrust::device_ptr<uint> counts_ptr(d_counts);
      thrust::device_ptr<uint> offsets_ptr(d_offsets);
      thrust::inclusive_scan(thrust::device, counts_ptr, counts_ptr + current_occupied_blocks_, offsets_ptr + 1);

      uint surface_count = 0;
      CUDA_CHECK(cudaMemcpy(&surface_count, d_offsets + current_occupied_blocks_, sizeof(uint), cudaMemcpyDeviceToHost));
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> surface_voxels(surface_count, 5);
      if (surface_count > 0) {
        CUDA_CHECK(cudaMalloc((void**) &d_voxels, sizeof(float) * surface_count * 5));
        surfaceVoxelFillKernel<<<current_occupied_blocks_, voxel_block_volume_>>>(d_instance_, d_offsets, surface_band, d_voxels);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(surface_voxels.data(), d_voxels, sizeof(float) * surface_count * 5, cudaMemcpyDeviceToHost));
      }

      CUDA_CHECK(cudaFree(d_counts));
      CUDA_CHECK(cudaFree(d_offsets));
      if (d_voxels != nullptr)
        CUDA_CHECK(cudaFree(d_voxels));
      return surface_voxels;
    }

    __device__ int floorHalfCoordinate(const int coordinate) {
      return coordinate >= 0 ? coordinate / 2 : -((-coordinate + 1) / 2);
    }

    template <TudfSampleSupport sample_support, TudfFieldValue field_value, typename T>
    __device__ bool sampleTudfImpl(
      const VoxelContainer<T>* container,
      const int3 half_position,
      float& distance) {
      const int3 lower = make_int3(
        floorHalfCoordinate(half_position.x),
        floorHalfCoordinate(half_position.y),
        floorHalfCoordinate(half_position.z));
      const float3 fraction = 0.5f * make_float3(half_position - 2 * lower);
      distance = 0.f;
      float interpolation_support = 0.f;
      for (int z = 0; z <= (fraction.z > 0.f); ++z) {
        for (int y = 0; y <= (fraction.y > 0.f); ++y) {
          for (int x = 0; x <= (fraction.x > 0.f); ++x) {
            const float coefficient =
              (x == 0 ? 1.f - fraction.x : fraction.x) *
              (y == 0 ? 1.f - fraction.y : fraction.y) *
              (z == 0 ? 1.f - fraction.z : fraction.z);
            const T voxel = container->getVoxel(lower + make_int3(x, y, z));
            if (twoSidedSurfaceWeight(voxel) >= container->min_weight_threshold_ &&
                voxel.sum_squared >= static_cast<int>(sample_support)) {
              float value;
              if constexpr (field_value == TudfFieldValue::distance)
                value = twoSidedSurfaceDistance(voxel);
              else if constexpr (field_value == TudfFieldValue::mean)
                value = voxel.sdf;
              else
                value = voxel.sum_squared;
              distance += coefficient * value;
              interpolation_support += coefficient;
            }
          }
        }
      }
      if (interpolation_support < tudf_min_interpolation_support)
        return false;
      distance /= interpolation_support;
      return true;
    }

    template <TudfSampleSupport sample_support, typename T>
    __device__ bool sampleTudf(
      const VoxelContainer<T>* container, const int3 half_position, float& distance) {
      return sampleTudfImpl<sample_support, TudfFieldValue::distance>(container, half_position, distance);
    }

    template <TudfSampleSupport sample_support, typename T>
    __device__ bool sampleTudfMean(
      const VoxelContainer<T>* container, const int3 half_position, float& distance) {
      return sampleTudfImpl<sample_support, TudfFieldValue::mean>(container, half_position, distance);
    }

    template <TudfSampleSupport sample_support, typename T>
    __device__ bool sampleTudfSecondMoment(
      const VoxelContainer<T>* container, const int3 half_position, float& second_moment) {
      return sampleTudfImpl<sample_support, TudfFieldValue::second_moment>(
        container, half_position, second_moment);
    }

    template <typename T>
    __device__ bool tudfGradient(
      const VoxelContainer<T>* container, const int3 half_position, const int scale, float3& gradient) {
      gradient = make_float3(0.f);
      for (int axis = 0; axis < 3; ++axis) {
        const int3 direction = scale * basisVector(axis);
        float negative;
        float positive;
        if (!sampleTudf<TudfSampleSupport::completed>(container, half_position - direction, negative) ||
            !sampleTudf<TudfSampleSupport::completed>(container, half_position + direction, positive))
          return false;
        gradient += (positive - negative) / (scale * container->virtual_voxel_size_) *
                    make_float3(basisVector(axis));
      }
      return true;
    }

    template <typename T>
    __device__ bool isTudfProjectionValid(
      const VoxelContainer<T>* container, const float3 projection, const int scale) {
      const int3 half_position = worldPointToVirtualVoxelPos(
        0.5f * container->virtual_voxel_size_, projection);
      float distance;
      return sampleTudf<TudfSampleSupport::completed>(container, half_position, distance) &&
             distance <= tudf_projection_distance_voxels * scale * container->virtual_voxel_size_;
    }

    template <typename T>
    __device__ bool isTudfCandidateCell(
      const VoxelContainer<T>* container,
      const int3 position,
      const int scale,
      float& mean_distance_voxels) {
      float center_distance;
      const int3 center_half_position = 2 * position + make_int3(scale);
      bool candidate = false;
      if (sampleTudf<TudfSampleSupport::observed>(container, center_half_position, center_distance))
        candidate = center_distance <= tudf_half_cell_diagonal * scale * container->virtual_voxel_size_;
      else if (sampleTudf<TudfSampleSupport::promoted>(container, center_half_position, center_distance))
        candidate = center_distance <= tudf_promoted_candidate_radius * scale * container->virtual_voxel_size_;
      else if (sampleTudf<TudfSampleSupport::completed>(container, center_half_position, center_distance))
        candidate = center_distance <= tudf_completed_candidate_radius * scale * container->virtual_voxel_size_;
      float mean_distance;
      if (!candidate ||
          !sampleTudfMean<TudfSampleSupport::completed>(container, center_half_position, mean_distance))
        return false;
      float observed_mean;
      float observed_second_moment;
      if (sampleTudfMean<TudfSampleSupport::observed>(container, center_half_position, observed_mean) &&
          sampleTudfSecondMoment<TudfSampleSupport::observed>(
            container, center_half_position, observed_second_moment) &&
          observed_second_moment - observed_mean * observed_mean >
            tudf_max_relative_variance * observed_mean * observed_mean)
        return false;
      mean_distance_voxels = mean_distance / container->virtual_voxel_size_;
      return true;
    }

    __device__ int dominantAxis(const float3 vector) {
      const float3 magnitude = fabs(vector);
      if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z)
        return 0;
      return magnitude.y >= magnitude.z ? 1 : 2;
    }

    struct TudfSurfaceEstimate {
      int3 position;
      float3 normal;
      float density;
      float mean_distance_voxels;
    };

    struct TudfSurfaceCache {
      int coordinate;
      uint packed_normal;
      ushort mean_distance;
      uchar density;
      uchar state;
    };
    static_assert(sizeof(TudfSurfaceCache) == 12);

    __device__ uint packTudfNormal(const float3 normal) {
      const uint3 encoded = make_uint3(clamp(0.5f * normal + 0.5f, 0.f, 1.f) * 1023.f + 0.5f);
      return encoded.x | (encoded.y << 10) | (encoded.z << 20);
    }

    __device__ float3 unpackTudfNormal(const uint packed) {
      const uint3 encoded = make_uint3(packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023);
      return normalize(2.f * make_float3(encoded) / 1023.f - 1.f);
    }

    __device__ TudfSurfaceCache cacheTudfSurface(
      const TudfSurfaceEstimate surface, const uchar state) {
      const int normal_axis = dominantAxis(surface.normal);
      return {
        component(surface.position, normal_axis),
        packTudfNormal(surface.normal) | (normal_axis << 30),
        static_cast<ushort>(__float2uint_rn(16384.f * surface.mean_distance_voxels)),
        static_cast<uchar>(__float2uint_rn(surface.density)),
        state,
      };
    }

    __device__ TudfSurfaceEstimate unpackTudfSurface(
      const TudfSurfaceCache cached, const int3 cell_position) {
      const float3 normal = unpackTudfNormal(cached.packed_normal);
      return {
        withComponent(cell_position, cached.packed_normal >> 30, cached.coordinate),
        normal,
        static_cast<float>(cached.density),
        cached.mean_distance / 16384.f,
      };
    }

    template <typename T>
    __device__ float tudfCellDensity(
      const VoxelContainer<T>* container, const int3 position, const int scale) {
      float density = 0.f;
      for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 2; ++y)
          for (int x = 0; x < 2; ++x)
            density += twoSidedSurfaceWeight(
              container->getVoxel(position + scale * make_int3(x, y, z)));
      return density / 8.f;
    }

    template <typename T>
    __device__ bool solveObservedTudfSurfaceCell(const VoxelContainer<T>* container,
                                                 const int3 position,
                                                 const int scale,
                                                 TudfSurfaceEstimate& surface) {
      if (!isTudfCandidateCell(container, position, scale, surface.mean_distance_voxels))
        return false;

      const float voxel_size = scale * container->virtual_voxel_size_;
      TudfQef qef;
      for (int z = 0; z < 3; ++z) {
        for (int y = 0; y < 3; ++y) {
          for (int x = 0; x < 3; ++x) {
            const int3 half_position = 2 * position + scale * make_int3(x, y, z);
            float distance;
            float3 gradient;
            if (!sampleTudf<TudfSampleSupport::completed>(container, half_position, distance) ||
                distance < tudf_min_sample_distance_voxels * voxel_size ||
                !tudfGradient(container, half_position, scale, gradient))
              continue;
            const float gradient_magnitude = length(gradient);
            if (gradient_magnitude < tudf_min_gradient)
              continue;
            const float3 normal = gradient / gradient_magnitude;
            const float3 sample_point =
              0.5f * container->virtual_voxel_size_ * make_float3(half_position);
            const float3 projection = sample_point - distance * normal;
            if (isTudfProjectionValid(container, projection, scale))
              addTudfPlane(qef, normal, projection);
          }
        }
      }
      const float3 center = virtualVoxelPosToWorld(
        container->virtual_voxel_size_, make_float3(position) + 0.5f * scale);
      float3 surface_point;
      float3 surface_direction;
      if (qef.plane_count < 3)
        return false;
      if (solveTudfQef(qef, center, surface_point, surface.normal, surface_direction) == 0)
        return false;
      surface.position = position;
      const int normal_axis = dominantAxis(surface.normal);
      const int3 qef_position = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, surface_point);
      surface.position = withComponent(surface.position, normal_axis, component(qef_position, normal_axis));
      surface.density = tudfCellDensity(container, position, scale);
      return true;
    }

    __device__ int3 tudfCellPosition(const HashEntry& entry, const uint voxel_idx) {
      return SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(delinearizeVoxelPos(voxel_idx));
    }

    template <typename T>
    __device__ bool getCachedTudfSurface(const VoxelContainer<T>* container,
                                         const TudfSurfaceCache* cache,
                                         const uint* block_to_compact,
                                         const int3 cell_position,
                                         const uchar source_state,
                                         TudfSurfaceEstimate& surface) {
      const HashEntry entry = container->getHashEntry(
        virtualVoxelPosToSDFBlock(cell_position, container->virtual_voxel_size_, container->voxel_extents_));
      if (entry.ptr == FREE_ENTRY)
        return false;
      const uint compact_index = block_to_compact[entry.ptr / total_sdf_block_size];
      if (compact_index == UINT_MAX)
        return false;
      const uint voxel_idx = virtualVoxelPosToSDFBlockIndex(cell_position);
      const TudfSurfaceCache cached = cache[compact_index * total_sdf_block_size + voxel_idx];
      const uchar state = cached.state & 0x7f;
      if (state == 0 || state > source_state)
        return false;
      surface = unpackTudfSurface(cached, tudfCellPosition(entry, voxel_idx));
      return true;
    }

    template <typename T>
    __global__ void initializeTudfSurfaceCacheKernel(VoxelContainer<T>* container,
                                                     TudfSurfaceCache* cache,
                                                     uint* block_to_compact) {
      const uint compact_index = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[compact_index];
      if (threadIdx.x == 0)
        block_to_compact[entry.ptr / total_sdf_block_size] = compact_index;
      for (uint voxel_idx = threadIdx.x; voxel_idx < total_sdf_block_size; voxel_idx += blockDim.x) {
        TudfSurfaceEstimate surface;
        TudfSurfaceCache& cached = cache[compact_index * total_sdf_block_size + voxel_idx];
        cached.state = 0;
        if (solveObservedTudfSurfaceCell(container, tudfCellPosition(entry, voxel_idx), 1, surface))
          cached = cacheTudfSurface(surface, 1);
      }
    }

    template <typename T>
    __global__ void markNonminimalTudfSurfacesKernel(const VoxelContainer<T>* container,
                                                     TudfSurfaceCache* cache,
                                                     const uint* block_to_compact) {
      const uint compact_index = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[compact_index];
      for (uint voxel_idx = threadIdx.x; voxel_idx < total_sdf_block_size; voxel_idx += blockDim.x) {
        TudfSurfaceCache& cached = cache[compact_index * total_sdf_block_size + voxel_idx];
        if ((cached.state & 0x7f) != 1)
          continue;
        const int3 cell_position = tudfCellPosition(entry, voxel_idx);
        const TudfSurfaceEstimate surface = unpackTudfSurface(cached, cell_position);
        const int normal_axis = cached.packed_normal >> 30;
        for (int side = -1; side <= 1; side += 2) {
          TudfSurfaceEstimate neighbor;
          if (!getCachedTudfSurface(
                container,
                cache,
                block_to_compact,
                cell_position + side * basisVector(normal_axis),
                1,
                neighbor) ||
              fabsf(dot(surface.normal, neighbor.normal)) < tudf_min_fill_normal_alignment ||
              maxComponent(abs(surface.position - neighbor.position)) > tudf_max_connector_coordinate_span)
            continue;
          if (neighbor.mean_distance_voxels < surface.mean_distance_voxels) {
            cached.state |= 0x80;
            break;
          }
        }
      }
    }

    __global__ void removeMarkedTudfSurfacesKernel(TudfSurfaceCache* cache) {
      const uint compact_index = blockIdx.x;
      for (uint voxel_idx = threadIdx.x; voxel_idx < total_sdf_block_size; voxel_idx += blockDim.x) {
        TudfSurfaceCache& cached = cache[compact_index * total_sdf_block_size + voxel_idx];
        if ((cached.state & 0x80) != 0)
          cached.state = 0;
      }
    }

    __device__ bool appendTudfEstimate(const TudfSurfaceEstimate candidate,
                                       const int normal_axis,
                                       TudfSurfaceEstimate* estimates,
                                       int& count) {
      if (dominantAxis(candidate.normal) != normal_axis ||
          (count > 0 && fabsf(dot(estimates[0].normal, candidate.normal)) < tudf_min_fill_normal_alignment))
        return false;
      estimates[count++] = candidate;
      return true;
    }

    __device__ int medianCoordinate(int* coordinates, const int count) {
      for (int index = 1; index < count; ++index) {
        const int value = coordinates[index];
        int insertion = index;
        while (insertion > 0 && coordinates[insertion - 1] > value) {
          coordinates[insertion] = coordinates[insertion - 1];
          --insertion;
        }
        coordinates[insertion] = value;
      }
      const int middle = count / 2;
      return count % 2 == 0
               ? __float2int_rn(0.5f * (coordinates[middle - 1] + coordinates[middle]))
               : coordinates[middle];
    }

    __device__ bool hasOpposingPair(const bool accepted[4]) {
      return (accepted[0] && accepted[1]) || (accepted[2] && accepted[3]);
    }

    template <typename T>
    __global__ void smoothUnderobservedTudfSurfacesKernel(const VoxelContainer<T>* container,
                                                          TudfSurfaceCache* cache,
                                                          const uint* block_to_compact) {
      const uint compact_index = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[compact_index];
      const float well_observed_density = container->min_weight_threshold_ + 1.f;
      for (uint voxel_idx = threadIdx.x; voxel_idx < total_sdf_block_size; voxel_idx += blockDim.x) {
        TudfSurfaceCache& cached = cache[compact_index * total_sdf_block_size + voxel_idx];
        if (cached.state != 1 || cached.density >= well_observed_density)
          continue;
        const int3 cell_position = tudfCellPosition(entry, voxel_idx);
        const TudfSurfaceEstimate surface = unpackTudfSurface(cached, cell_position);
        const int normal_axis = cached.packed_normal >> 30;
        const int first_axis = (normal_axis + 1) % 3;
        const int second_axis = (normal_axis + 2) % 3;
        int coordinates[8];
        int coordinate_count = 0;
        for (int first_offset = -1; first_offset <= 1; ++first_offset) {
          for (int second_offset = -1; second_offset <= 1; ++second_offset) {
            if (first_offset == 0 && second_offset == 0)
              continue;
            const int3 offset = first_offset * basisVector(first_axis) +
                                second_offset * basisVector(second_axis);
            TudfSurfaceEstimate neighbor;
            if (getCachedTudfSurface(
                  container,
                  cache,
                  block_to_compact,
                  cell_position + offset,
                  1,
                  neighbor) &&
                neighbor.density >= well_observed_density &&
                dominantAxis(neighbor.normal) == normal_axis &&
                fabsf(dot(surface.normal, neighbor.normal)) >= tudf_min_fill_normal_alignment)
              coordinates[coordinate_count++] = component(neighbor.position, normal_axis);
          }
        }
        if (coordinate_count < 3)
          continue;
        const int coordinate = medianCoordinate(coordinates, coordinate_count);
        if (coordinates[coordinate_count - 1] - coordinates[0] <= tudf_max_connector_coordinate_span)
          cached.coordinate = coordinate;
      }
    }

    template <typename T>
    __device__ bool fillCachedTudfSurfaceCell(const VoxelContainer<T>* container,
                                              const TudfSurfaceCache* cache,
                                              const uint* block_to_compact,
                                              const int3 position,
                                              const uchar source_state,
                                              const bool require_opposing_support,
                                              TudfSurfaceEstimate& surface) {
      bool valid[6];
      TudfSurfaceEstimate axial_estimates[6];
      for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
          const int3 direction = (2 * side - 1) * basisVector(axis);
          const int neighbor = 2 * axis + side;
          valid[neighbor] = getCachedTudfSurface(
            container,
            cache,
            block_to_compact,
            position + direction,
            source_state,
            axial_estimates[neighbor]);
        }
      }

      for (int normal_axis = 0; normal_axis < 3; ++normal_axis) {
        const int first_axis = (normal_axis + 1) % 3;
        const int second_axis = (normal_axis + 2) % 3;
        const int neighbors[4] = {2 * first_axis, 2 * first_axis + 1, 2 * second_axis, 2 * second_axis + 1};
        TudfSurfaceEstimate estimates[8];
        bool axial_accepted[4] = {};
        int estimate_count = 0;
        for (int index = 0; index < 4; ++index) {
          const int neighbor = neighbors[index];
          if (valid[neighbor])
            axial_accepted[index] = appendTudfEstimate(
              axial_estimates[neighbor], normal_axis, estimates, estimate_count);
        }
        bool opposing_support = hasOpposingPair(axial_accepted);
        if (estimate_count < 3) {
          bool diagonal_accepted[4] = {};
          int diagonal_index = 0;
          for (int first_side = -1; first_side <= 1; first_side += 2) {
            for (int second_side = -1; second_side <= 1; second_side += 2) {
              const int3 offset = first_side * basisVector(first_axis) +
                                  second_side * basisVector(second_axis);
              TudfSurfaceEstimate diagonal;
              if (getCachedTudfSurface(
                    container,
                    cache,
                    block_to_compact,
                    position + offset,
                    source_state,
                    diagonal))
                diagonal_accepted[diagonal_index] = appendTudfEstimate(
                  diagonal, normal_axis, estimates, estimate_count);
              ++diagonal_index;
            }
          }
          opposing_support |= hasOpposingPair(diagonal_accepted);
        }
        if (estimate_count < 2 || (require_opposing_support ? !opposing_support : estimate_count < 3))
          continue;

        int coordinates[8];
        for (int index = 0; index < estimate_count; ++index)
          coordinates[index] = component(estimates[index].position, normal_axis);
        const int surface_coordinate = medianCoordinate(coordinates, estimate_count);
        const int maximum_span = require_opposing_support ? 1 : 2;
        if (coordinates[estimate_count - 1] - coordinates[0] > maximum_span)
          continue;

        surface.position = withComponent(position, normal_axis, surface_coordinate);
        surface.normal = estimates[0].normal;
        surface.density = estimates[0].density;
        surface.mean_distance_voxels = estimates[0].mean_distance_voxels;
        for (int index = 1; index < estimate_count; ++index) {
          surface.normal += copysignf(1.f, dot(surface.normal, estimates[index].normal)) *
                            estimates[index].normal;
          surface.density += estimates[index].density;
          surface.mean_distance_voxels += estimates[index].mean_distance_voxels;
        }
        surface.normal = normalize(surface.normal);
        surface.density /= estimate_count;
        surface.mean_distance_voxels /= estimate_count;
        return true;
      }
      return false;
    }

    template <typename T>
    __global__ void fillTudfSurfaceCacheKernel(const VoxelContainer<T>* container,
                                               TudfSurfaceCache* cache,
                                               const uint* block_to_compact,
                                               const uchar source_state,
                                               const bool require_opposing_support) {
      const uint compact_index = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[compact_index];
      for (uint voxel_idx = threadIdx.x; voxel_idx < total_sdf_block_size; voxel_idx += blockDim.x) {
        TudfSurfaceCache& cached = cache[compact_index * total_sdf_block_size + voxel_idx];
        if (cached.state != 0)
          continue;
        TudfSurfaceEstimate surface;
        if (fillCachedTudfSurfaceCell(
              container,
              cache,
              block_to_compact,
              tudfCellPosition(entry, voxel_idx),
              source_state,
              require_opposing_support,
              surface))
          cached = cacheTudfSurface(surface, source_state + 1);
      }
    }

    __device__ int3 tudfConnectorOffset(const TudfSurfaceEstimate surface, const int index) {
      if (index < 3)
        return basisVector(index);
      const int2 tangent_offsets[42] = {
        make_int2(0, 2),
        make_int2(1, -2),
        make_int2(1, -1),
        make_int2(1, 1),
        make_int2(1, 2),
        make_int2(2, -2),
        make_int2(2, -1),
        make_int2(2, 0),
        make_int2(2, 1),
        make_int2(2, 2),
        make_int2(0, 3),
        make_int2(1, -3),
        make_int2(1, 3),
        make_int2(2, -3),
        make_int2(2, 3),
        make_int2(3, -3),
        make_int2(3, -2),
        make_int2(3, -1),
        make_int2(3, 0),
        make_int2(3, 1),
        make_int2(3, 2),
        make_int2(3, 3),
        make_int2(0, 4),
        make_int2(1, -4),
        make_int2(1, 4),
        make_int2(2, -4),
        make_int2(2, 4),
        make_int2(3, -4),
        make_int2(3, 4),
        make_int2(4, -4),
        make_int2(4, -3),
        make_int2(4, -2),
        make_int2(4, -1),
        make_int2(4, 0),
        make_int2(4, 1),
        make_int2(4, 2),
        make_int2(4, 3),
        make_int2(4, 4),
        make_int2(0, 5),
        make_int2(5, 0),
        make_int2(5, -5),
        make_int2(5, 5),
      };
      const int normal_axis = dominantAxis(surface.normal);
      const int first_axis = (normal_axis + 1) % 3;
      const int second_axis = (normal_axis + 2) % 3;
      const int2 offset = tangent_offsets[index - 3];
      return offset.x * basisVector(first_axis) + offset.y * basisVector(second_axis);
    }

    template <typename T>
    __device__ int cachedTudfConnectorCount(const VoxelContainer<T>* container,
                                            const TudfSurfaceCache* cache,
                                            const uint* block_to_compact,
                                            const int3 cell_position,
                                            const TudfSurfaceEstimate surface) {
      int connector_count = 0;
      for (int neighbor_index = 0; neighbor_index < 45; ++neighbor_index) {
        TudfSurfaceEstimate neighbor;
        if (!getCachedTudfSurface(
              container,
              cache,
              block_to_compact,
              cell_position + tudfConnectorOffset(surface, neighbor_index),
              3,
              neighbor) ||
            fabsf(dot(surface.normal, neighbor.normal)) < tudf_min_connector_normal_alignment)
          continue;
        const int3 span = abs(neighbor.position - surface.position);
        if (maxComponent(span) <= tudf_max_connector_coordinate_span)
          connector_count += max(0, dot(span, make_int3(1)) - 1);
      }
      return connector_count;
    }

    __device__ float packedNormalColor(const float3 normal) {
      const float3 color = clamp(0.5f * normal + 0.5f, 0.f, 1.f) * 255.f;
      const uint packed = __float2uint_rn(color.x) |
                          (__float2uint_rn(color.y) << 8) |
                          (__float2uint_rn(color.z) << 16);
      return __uint_as_float(packed);
    }

    template <typename T>
    __device__ void writeTwoSidedSurfaceVoxel(const VoxelContainer<T>* container,
                                              const int3 position,
                                              const float density,
                                              const float3 normal,
                                              const uint output_idx,
                                              float* voxels) {
      const float3 point = virtualVoxelPosToWorld(container->virtual_voxel_size_, position);
      voxels[output_idx * 5 + 0] = point.x;
      voxels[output_idx * 5 + 1] = point.y;
      voxels[output_idx * 5 + 2] = point.z;
      voxels[output_idx * 5 + 3] = density;
      voxels[output_idx * 5 + 4] = packedNormalColor(normal);
    }

    __device__ float median(float* values, const int count) {
      for (int index = 1; index < count; ++index) {
        const float value = values[index];
        int insertion = index;
        while (insertion > 0 && values[insertion - 1] > value) {
          values[insertion] = values[insertion - 1];
          --insertion;
        }
        values[insertion] = value;
      }
      const int middle = count / 2;
      return count % 2 == 0 ? 0.5f * (values[middle - 1] + values[middle]) : values[middle];
    }

    template <typename T>
    __global__ void completeTudfFieldKernel(VoxelContainer<T>* container) {
      const HashEntry& entry = container->d_compactHashTable_[blockIdx.x];
      const int scale = 1 << entry.resolution;
      for (uint voxel_idx = threadIdx.x; voxel_idx < container->getNumVoxels(entry); voxel_idx += blockDim.x) {
        T& voxel = container->d_SDFBlocks_[entry.ptr + voxel_idx];
        if (twoSidedSurfaceWeight(voxel) >= container->min_weight_threshold_)
          continue;

        const int3 position = SDFBlockToVirtualVoxelPos(entry.pos) +
                              scale * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scale));
        float distances[6];
        int distance_count = 0;
        int neighbor_count = 0;
        for (int axis = 0; axis < 3; ++axis) {
          const int3 direction = scale * basisVector(axis);
          for (int side = -1; side <= 1; side += 2) {
            const T neighbor = container->getVoxel(position + side * direction);
            if (twoSidedSurfaceWeight(neighbor) >= container->min_weight_threshold_) {
              distances[distance_count++] = twoSidedSurfaceDistance(neighbor);
              ++neighbor_count;
            }
          }
        }
        if (neighbor_count < tudf_min_completion_neighbors)
          continue;
        const bool measured = twoSidedSurfaceWeight(voxel) > 0;
        voxel.sdf = measured
                      ? twoSidedSurfaceDistance(voxel)
                      : median(distances, distance_count);
        voxel.weight = container->min_weight_threshold_;
        voxel.sum_squared = static_cast<int>(
          measured ? TudfSampleSupport::promoted : TudfSampleSupport::completed);
      }
    }

    template <typename T>
    __global__ void selectOwnerBlocksKernel(const VoxelContainer<T>* container,
                                            const int3 owner_chunk,
                                            const float3 chunk_extents,
                                            uint* owner_blocks,
                                            uint* owner_block_count) {
      const uint block_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (block_idx >= container->current_occupied_blocks_)
        return;
      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      const float3 block_position = SDFBlockToWorldPoint(container->virtual_voxel_size_, entry.pos);
      const int3 block_chunk = worldToChunks(block_position, chunk_extents);
      if (block_chunk.x == owner_chunk.x && block_chunk.y == owner_chunk.y && block_chunk.z == owner_chunk.z)
        owner_blocks[atomicAdd(owner_block_count, 1)] = block_idx;
    }

    template <typename T>
    __launch_bounds__(tudf_extraction_threads) __global__ void tudfSurfaceVoxelCountsKernel(
      const VoxelContainer<T>* container,
      const TudfSurfaceCache* cache,
      const uint* block_to_compact,
      const uint* owner_blocks,
      uint* counts) {
      const uint owner_block_idx = blockIdx.x;
      const uint block_idx = owner_blocks[owner_block_idx];

      __shared__ uint count;
      if (threadIdx.x == 0)
        count = 0;
      __syncthreads();

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      for (uint voxel_idx = threadIdx.x; voxel_idx < container->getNumVoxels(entry); voxel_idx += blockDim.x) {
        const TudfSurfaceCache cached = cache[block_idx * total_sdf_block_size + voxel_idx];
        if (cached.state == 0)
          continue;
        const int3 cell_position = tudfCellPosition(entry, voxel_idx);
        const TudfSurfaceEstimate surface = unpackTudfSurface(cached, cell_position);
        atomicAdd(
          &count,
          1 + cachedTudfConnectorCount(
                container, cache, block_to_compact, cell_position, surface));
      }

      __syncthreads();
      if (threadIdx.x == 0)
        counts[owner_block_idx] = count;
    }

    template <typename T>
    __launch_bounds__(tudf_extraction_threads) __global__ void tudfSurfaceVoxelFillKernel(
      const VoxelContainer<T>* container,
      const TudfSurfaceCache* cache,
      const uint* block_to_compact,
      const uint* offsets,
      const uint* owner_blocks,
      const uint start_owner_block,
      float* voxels) {
      const uint owner_block_idx = start_owner_block + blockIdx.x;
      const uint block_idx = owner_blocks[owner_block_idx];

      __shared__ uint local_count;
      if (threadIdx.x == 0)
        local_count = 0;
      __syncthreads();

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      for (uint voxel_idx = threadIdx.x; voxel_idx < container->getNumVoxels(entry); voxel_idx += blockDim.x) {
        const TudfSurfaceCache cached = cache[block_idx * total_sdf_block_size + voxel_idx];
        if (cached.state == 0)
          continue;
        const int3 cell_position = tudfCellPosition(entry, voxel_idx);
        const TudfSurfaceEstimate surface = unpackTudfSurface(cached, cell_position);
        const int connector_count = cachedTudfConnectorCount(
          container, cache, block_to_compact, cell_position, surface);
        uint output_idx = offsets[owner_block_idx] - offsets[start_owner_block] +
                          atomicAdd(&local_count, 1 + connector_count);
        writeTwoSidedSurfaceVoxel(
          container, surface.position, surface.density, surface.normal, output_idx, voxels);
        ++output_idx;

        for (int neighbor_index = 0; neighbor_index < 45; ++neighbor_index) {
          TudfSurfaceEstimate neighbor;
          if (!getCachedTudfSurface(
                container,
                cache,
                block_to_compact,
                cell_position + tudfConnectorOffset(surface, neighbor_index),
                3,
                neighbor) ||
              fabsf(dot(surface.normal, neighbor.normal)) < tudf_min_connector_normal_alignment)
            continue;
          const int3 span = abs(neighbor.position - surface.position);
          if (maxComponent(span) > tudf_max_connector_coordinate_span)
            continue;
          const int manhattan_distance = dot(span, make_int3(1));
          const float orientation = copysignf(1.f, dot(surface.normal, neighbor.normal));
          const float3 connector_normal = normalize(surface.normal + orientation * neighbor.normal);
          const float connector_density = 0.5f * (surface.density + neighbor.density);
          int3 connector_position = surface.position;
          int connector_index = 0;
          for (int axis = 0; axis < 3; ++axis) {
            const int coordinate = component(neighbor.position, axis);
            const int current_coordinate = component(connector_position, axis);
            const int step = coordinate >= current_coordinate ? 1 : -1;
            while (component(connector_position, axis) != coordinate) {
              connector_position += step * basisVector(axis);
              if (++connector_index < manhattan_distance)
                writeTwoSidedSurfaceVoxel(
                  container, connector_position, connector_density, connector_normal, output_idx++, voxels);
            }
          }
        }
      }
    }

    template <typename T>
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::tudfSurfaceVoxels(
      const int3& owner_chunk, const float3& chunk_extents) {
      flatAndReduceHashTable();
      if (current_occupied_blocks_ == 0)
        return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(0, 5);

      for (int completion_pass = 0; completion_pass < 3; ++completion_pass)
        completeTudfFieldKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(d_instance_);
      TudfSurfaceCache* d_surface_cache = nullptr;
      uint* d_block_to_compact = nullptr;
      CUDA_CHECK(cudaMalloc(
        (void**) &d_surface_cache,
        sizeof(TudfSurfaceCache) * current_occupied_blocks_ * total_sdf_block_size));
      CUDA_CHECK(cudaMalloc((void**) &d_block_to_compact, sizeof(uint) * num_sdf_blocks_));
      CUDA_CHECK(cudaMemset(d_block_to_compact, 0xff, sizeof(uint) * num_sdf_blocks_));
      initializeTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact);
      markNonminimalTudfSurfacesKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact);
      removeMarkedTudfSurfacesKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_surface_cache);
      smoothUnderobservedTudfSurfacesKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact);
      fillTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, 1, false);
      fillTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, 2, true);
      fillTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, 3, true);
      fillTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, 4, true);
      fillTudfSurfaceCacheKernel<<<current_occupied_blocks_, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, 5, true);
      uint* d_owner_blocks = nullptr;
      uint* d_owner_block_count = nullptr;
      CUDA_CHECK(cudaMalloc((void**) &d_owner_blocks, sizeof(uint) * current_occupied_blocks_));
      CUDA_CHECK(cudaMalloc((void**) &d_owner_block_count, sizeof(uint)));
      CUDA_CHECK(cudaMemset(d_owner_block_count, 0, sizeof(uint)));
      constexpr uint selection_threads = 256;
      const uint selection_blocks = (current_occupied_blocks_ + selection_threads - 1) / selection_threads;
      selectOwnerBlocksKernel<<<selection_blocks, selection_threads>>>(
        d_instance_, owner_chunk, chunk_extents, d_owner_blocks, d_owner_block_count);
      uint owner_block_count = 0;
      CUDA_CHECK(cudaMemcpy(&owner_block_count, d_owner_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_owner_block_count));
      if (owner_block_count == 0) {
        CUDA_CHECK(cudaFree(d_owner_blocks));
        CUDA_CHECK(cudaFree(d_block_to_compact));
        CUDA_CHECK(cudaFree(d_surface_cache));
        return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(0, 5);
      }

      uint* d_counts = nullptr;
      uint* d_offsets = nullptr;
      CUDA_CHECK(cudaMalloc((void**) &d_counts, sizeof(uint) * owner_block_count));
      CUDA_CHECK(cudaMalloc((void**) &d_offsets, sizeof(uint) * (owner_block_count + 1)));
      CUDA_CHECK(cudaMemset(d_offsets, 0, sizeof(uint)));

      tudfSurfaceVoxelCountsKernel<<<owner_block_count, tudf_extraction_threads>>>(
        d_instance_, d_surface_cache, d_block_to_compact, d_owner_blocks, d_counts);
      CUDA_CHECK(cudaDeviceSynchronize());

      thrust::device_ptr<uint> counts_ptr(d_counts);
      thrust::device_ptr<uint> offsets_ptr(d_offsets);
      thrust::inclusive_scan(thrust::device, counts_ptr, counts_ptr + owner_block_count, offsets_ptr + 1);

      uint surface_count = 0;
      CUDA_CHECK(cudaMemcpy(&surface_count, d_offsets + owner_block_count, sizeof(uint), cudaMemcpyDeviceToHost));
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> surface_voxels(surface_count, 5);
      std::vector<uint> offsets(owner_block_count + 1);
      CUDA_CHECK(cudaMemcpy(
        offsets.data(), d_offsets, sizeof(uint) * offsets.size(), cudaMemcpyDeviceToHost));
      size_t free_bytes = 0;
      size_t total_bytes = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      const uint batch_sample_capacity = free_bytes / (2 * sizeof(float) * 5);
      for (uint start_owner_block = 0; start_owner_block < owner_block_count;) {
        uint end_owner_block = start_owner_block + 1;
        while (end_owner_block < owner_block_count &&
               offsets[end_owner_block + 1] - offsets[start_owner_block] <= batch_sample_capacity)
          ++end_owner_block;
        const uint batch_count = offsets[end_owner_block] - offsets[start_owner_block];
        float* d_voxels = nullptr;
        CUDA_CHECK(cudaMalloc((void**) &d_voxels, sizeof(float) * batch_count * 5));
        tudfSurfaceVoxelFillKernel<<<end_owner_block - start_owner_block, tudf_extraction_threads>>>(
          d_instance_,
          d_surface_cache,
          d_block_to_compact,
          d_offsets,
          d_owner_blocks,
          start_owner_block,
          d_voxels);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(
          surface_voxels.data() + offsets[start_owner_block] * 5,
          d_voxels,
          sizeof(float) * batch_count * 5,
          cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_voxels));
        start_owner_block = end_owner_block;
      }

      CUDA_CHECK(cudaFree(d_counts));
      CUDA_CHECK(cudaFree(d_offsets));
      CUDA_CHECK(cudaFree(d_owner_blocks));
      CUDA_CHECK(cudaFree(d_block_to_compact));
      CUDA_CHECK(cudaFree(d_surface_cache));
      return surface_voxels;
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlock(const int3& pos,
                                                                                               const int resolution) {
      uint h        = calculateHash(pos);    // hash bucket
      const uint hp = h * hash_bucket_size_; // hash position

      int first_empty = -1;
      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i                = hp + j;
        const HashEntry& curr = d_hashTable_[i];
        // in that case the SDF-block is already allocated and corresponds to the current position exit thread
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1;
        }

        // store the first FREE_ENTRY hash entry
        if (first_empty == -1 && curr.ptr == FREE_ENTRY) {
          first_empty = i;
        }
      }

#ifdef RESOLVE_COLLISION

      // handling collisions
      // updated variables as after the loop
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;
      // traverse list until end: memorize idx at list end save offset from last element of
      // bucket to list end int k = 0;

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr = d_hashTable_[i];
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1; // Block already allocated
        }
        if (curr.offset == 0) { // we have found the end of the list
          break;
        }
        i = idx_last_entry_in_bucket + curr.offset;   // go to next element in the list
        i %= (hash_bucket_size_ * hash_num_buckets_); // check for overflow

        max_iter++;
      }

#endif

      // if there is an empty entry and we haven't allocated the current entry before
      if (first_empty != -1) {
        int prev_val = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
        if (prev_val != LOCK_ENTRY) { // only proceed if the bucket has been locked
          HashEntry& entry = d_hashTable_[first_empty];
          entry.pos        = pos;
          entry.offset     = NO_OFFSET;
          entry.resolution = resolution;
          int ptr_idx      = -1;
          if (entry.resolution == 0)
            ptr_idx = consumeHeapHigh();
          else if (entry.resolution == 1)
            ptr_idx = consumeHeapLow();
          if (ptr_idx < 0) {
            printf("allocBlock |  mem size exceed, not inserting hash entry!\n");
            return -1;
          }
          const int voxel_block_volume = getNumVoxels(entry);
          entry.ptr                    = ptr_idx * voxel_block_volume;
        }
        return first_empty;
      }

#ifdef RESOLVE_COLLISION
      // handling collisions
      int offset = 0;
      // linear search for free entry
      max_iter = 0;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        offset++;
        // go to next hash element
        i = (idx_last_entry_in_bucket + offset) % (total_size_);
        if ((offset % hash_bucket_size_) == 0)
          continue; // cannot insert into a last bucket element (would conflict with other linked
                    // lists)
        curr = d_hashTable_[i];
        if (curr.ptr == FREE_ENTRY) { // this is the first free entry
          int prev_value = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
          if (prev_value != LOCK_ENTRY) {
            HashEntry last_entry_in_bucket = d_hashTable_[idx_last_entry_in_bucket];
            h                              = i / hash_bucket_size_;
            prev_value                     = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
            if (prev_value != LOCK_ENTRY) { // only proceed if the bucket has been locked
              HashEntry& entry = d_hashTable_[i];
              entry.pos        = pos;
              entry.offset     = last_entry_in_bucket.offset;
              entry.resolution = resolution;
              int ptr_idx      = -1;
              if (entry.resolution == 0)
                ptr_idx = consumeHeapHigh();
              else if (entry.resolution == 1)
                ptr_idx = consumeHeapLow();
              if (ptr_idx < 0) {
                printf("allocBlock |  mem size exceed, not inserting hash entry!\n");
                return -1;
              }
              const int voxel_block_volume = getNumVoxels(entry);
              entry.ptr                    = ptr_idx * voxel_block_volume;

              last_entry_in_bucket.offset            = offset;
              d_hashTable_[idx_last_entry_in_bucket] = last_entry_in_bucket;
            }
          }
          return -1; // bucket was already locked
        }

        max_iter++;
      }
#endif
      return -1;
    }

    template <typename T>
    __device__ int VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reallocBlock(const int3& pos,
                                                                                                 const int resolution) {
      uint h        = calculateHash(pos);    // hash bucket
      const uint hp = h * hash_bucket_size_; // hash position

      int first_empty = -1;
      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i                = hp + j;
        const HashEntry& curr = d_hashTable_[i];
        // in that case the SDF-block is already allocated and corresponds to the current position exit thread
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1;
        }

        // store the first FREE_ENTRY hash entry
        if (first_empty == -1 && curr.ptr == FREE_ENTRY) {
          first_empty = i;
        }
      }

#ifdef RESOLVE_COLLISION

      // handling collisions
      // updated variables as after the loop
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      uint i                              = idx_last_entry_in_bucket;        // start with the last entry of the current bucket

      HashEntry curr;
      curr.offset = 0;
      // traverse list until end: memorize idx at list end save offset from last element of
      // bucket to list end int k = 0;

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr = d_hashTable_[i];
        if (curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z && curr.ptr != FREE_ENTRY) {
          return -1; // Block already allocated
        }
        if (curr.offset == 0) { // we have found the end of the list
          break;
        }
        i = idx_last_entry_in_bucket + curr.offset;   // go to next element in the list
        i %= (hash_bucket_size_ * hash_num_buckets_); // check for overflow

        max_iter++;
      }

#endif

      // if there is an empty entry and we haven't allocated the current entry before
      if (first_empty != -1) {
        int prev_val = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
        if (prev_val != LOCK_ENTRY) { // only proceed if the bucket has been locked
          HashEntry& entry             = d_hashTable_[first_empty];
          entry.pos                    = pos;
          entry.offset                 = NO_OFFSET;
          entry.resolution             = resolution;
          const int voxel_block_volume = getNumVoxels(entry);
          int ptr_idx                  = -1;
          if (entry.resolution == 0)
            ptr_idx = consumeHeapHigh();
          else if (entry.resolution == 1)
            ptr_idx = consumeHeapLow();
          if (ptr_idx < 0) {
            printf("reallocBlock |  %d mem size exceed (heapCounterHigh: %d | heapCounterLow: %d), not "
                   "inserting hash entry!\n",
                   ptr_idx,
                   *d_heapCounterHigh_,
                   *d_heapCounterLow_);
            return -1;
          }
          entry.ptr = ptr_idx * voxel_block_volume;
          return first_empty;
        } else {
          return LOCK_ENTRY;
        }
      }

#ifdef RESOLVE_COLLISION
      // handling collisions
      int offset = 0;
      // linear search for free entry
      max_iter = 0;
#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        offset++;
        // go to next hash element
        i = (idx_last_entry_in_bucket + offset) % (total_size_);
        if ((offset % hash_bucket_size_) == 0)
          continue; // cannot insert into a last bucket element (would conflict with other linked
                    // lists)
        curr = d_hashTable_[i];
        if (curr.ptr == FREE_ENTRY) { // this is the first free entry
          int prev_value = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
          if (prev_value != LOCK_ENTRY) {
            HashEntry last_entry_in_bucket = d_hashTable_[idx_last_entry_in_bucket];
            h                              = i / hash_bucket_size_;
            prev_value                     = atomicExch(&d_hashTableBucketMutex_[h], LOCK_ENTRY);
            if (prev_value != LOCK_ENTRY) { // only proceed if the bucket has been locked
              HashEntry& entry             = d_hashTable_[i];
              entry.pos                    = pos;
              entry.offset                 = last_entry_in_bucket.offset;
              entry.resolution             = resolution;
              const int voxel_block_volume = getNumVoxels(entry);
              int ptr_idx                  = -1;
              if (entry.resolution == 0)
                ptr_idx = consumeHeapHigh();
              else if (entry.resolution == 1)
                ptr_idx = consumeHeapLow();
              if (ptr_idx < 0) {
                printf("reallocBlock |  mem size exceed, not inserting hash entry!\n");
                return LOCK_ENTRY;
              }
              entry.ptr = ptr_idx * voxel_block_volume;

              last_entry_in_bucket.offset            = offset;
              d_hashTable_[idx_last_entry_in_bucket] = last_entry_in_bucket;
            }
          }
          return LOCK_ENTRY; // bucket was already locked
        }

        max_iter++;
      }
#endif
      return -1;
    }

    template <typename T>
    __global__ void allocBlocksKernel(const CUDAMatrixf* depth_img,
                                      const Camera* camera,
                                      const float max_integration_distance,
                                      const float sdf_truncation,
                                      const float sdf_truncation_scale,
                                      VoxelContainer<T>* container) {
      int row = blockDim.y * blockIdx.y + threadIdx.y;
      int col = blockDim.x * blockIdx.x + threadIdx.x;

      if (!depth_img->inside(row, col))
        return;

      const float depth = depth_img->at<1>(row, col);

      if (depth <= camera->minDepth() || depth > camera->maxDepth())
        return;

      const float t         = getTruncation(depth, sdf_truncation, sdf_truncation_scale);
      const float min_depth = min(max_integration_distance, depth - t);
      const float max_depth = min(max_integration_distance, depth + t);

      if (min_depth >= max_depth)
        return;

      // clang-format off
    float3 pcam_min = camera->inverseProjection(row, col, min_depth);
    float3 pcam_max = camera->inverseProjection(row, col, max_depth);


    float3 pw_min = camera->camInWorld() * pcam_min;
    float3 pw_max = camera->camInWorld() * pcam_max;
    
    float3 dir = normalize(pw_max - pw_min);

    int3 id_current_voxel = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_min);
    int3 id_end = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_max);

    float3 step = make_float3(sign(dir));

    float3 boundary_pos = SDFBlockToWorldPoint(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) - 0.5f * container->virtual_voxel_size_;
    float3 t_max   = (boundary_pos - pw_min) / dir;
    float3 t_delta = (step * sdf_block_size * container->virtual_voxel_size_) / dir;
    int3 id_bound  = make_int3(make_float3(id_end) + step);

    if (fabsf(dir.x) < FLOAT_EPSILON) {
      t_max.x   = numeric_limits<float>::max();
      t_delta.x = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
      t_max.x   = numeric_limits<float>::max();
      t_delta.x = numeric_limits<float>::max();
    }

    if (fabsf(dir.y) < FLOAT_EPSILON) {
      t_max.y   = numeric_limits<float>::max();
      t_delta.y = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
      t_max.y   = numeric_limits<float>::max();
      t_delta.y = numeric_limits<float>::max();
    }

    if (fabsf(dir.z) < FLOAT_EPSILON) {
      t_max.z   = numeric_limits<float>::max();
      t_delta.z = numeric_limits<float>::max();
    }
    if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
      t_max.z   = numeric_limits<float>::max();
      t_delta.z = numeric_limits<float>::max();
    }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        if (container->isSDFBlockInCameraFrustumApprox(camera, id_current_voxel)) {
          const int is_allocated = container->allocBlock(id_current_voxel);
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }

        iter++;
      }
    }

    template <typename T>
    __global__ void allocateMemoryLow(VoxelContainer<T>* container) {
      __shared__ int addr_high;
      __shared__ int addr_low;
      if (threadIdx.x == 0) {
        addr_high = atomicSub(&container->d_heapCounterHigh_[0], 1);
        addr_low  = atomicAdd(&container->d_heapCounterLow_[0], octree_branching_factor);
      }
      __syncthreads();
      int idx = threadIdx.x + 1;
      container->d_heap_low_[addr_low + idx] =
        container->d_heap_high_[addr_high] * octree_branching_factor + octree_branching_factor - idx;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlocks(const CUDAMatrixf& depth_img,
                                                                                      const Camera& camera) {
      // fast version, does not guarantee that all blocks are allocated (staggers alloc to the next frame)

      int prev_free_blocks = getHeapHighFreeCount();
      prev_free_blocks += getHeapLowFreeCount();

      resetHashBucketMutex();

      if (sdf_var_threshold_ > 0.f && getHeapLowFreeCount() < low_blocks_to_allocate_) {
        std::cerr << "allocBlocks | need to allocate additional memory in low-res heap" << std::endl;
        const dim3 threads_per_block_lowalloc(octree_branching_factor, 1);
        const dim3 n_blocks(low_blocks_to_allocate_, 1);
        allocateMemoryLow<<<n_blocks, threads_per_block_lowalloc>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      allocBlocksKernel<<<camera.blocks(), camera.threads()>>>(depth_img.deviceInstance(),
                                                               camera.deviceInstance(),
                                                               max_integration_distance_,
                                                               sdf_truncation_,
                                                               sdf_truncation_scale_,
                                                               d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

      while (1) {
        resetHashBucketMutex();
        allocBlocksKernel<<<camera.blocks(), camera.threads()>>>(depth_img.deviceInstance(),
                                                                 camera.deviceInstance(),
                                                                 max_integration_distance_,
                                                                 sdf_truncation_,
                                                                 sdf_truncation_scale_,
                                                                 d_instance_);

        CUDA_CHECK(cudaDeviceSynchronize());
        int curr_free_blocks = getHeapHighFreeCount();
        curr_free_blocks += getHeapLowFreeCount();
        if (prev_free_blocks == curr_free_blocks) {
          break;
        }
        prev_free_blocks = curr_free_blocks;
      }

#endif
    }

    template <typename T>
    __global__ void allocBlocks3DKernel(const CUDAVectorf3* point_cloud,
                                        const CUDAVectorf3* normals,
                                        const CUDAVectorf* weights,
                                        const Camera* camera,
                                        const float max_integration_distance,
                                        const float sdf_truncation,
                                        const float sdf_truncation_scale,
                                        VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3& pcam       = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);

      if (range == 0.f)
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float t         = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth = min(max_integration_distance, range - t);
      const float max_depth = min(max_integration_distance, range + t);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min, pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam + cam_dir * (min_depth - range);
        pcam_max = pcam + cam_dir * (max_depth - range);
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_min);
      int3 id_end           = worldPointToSDFBlock(container->virtual_voxel_size_, container->voxel_extents_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        SDFBlockToWorldPoint(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * sdf_block_size * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0;
#pragma unroll
      while (iter < max_dda_iteration_count) {
        const int& is_allocated = container->allocBlock(id_current_voxel);

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }

        iter++;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocBlocks3D(const CUDAVectorf3& point_cloud,
                                                                                        const CUDAVectorf3& normals,
                                                                                        const CUDAVectorf weights,
                                                                                        const Camera& camera) {
      // fast version, does not guarantee that all blocks are allocated (staggers alloc to the next frame)

      int prev_free_blocks = getHeapHighFreeCount();
      prev_free_blocks += getHeapLowFreeCount();

      resetHashBucketMutex();

      if (sdf_var_threshold_ > 0.f && getHeapLowFreeCount() < low_blocks_to_allocate_) {
        std::cerr << "allocBlocks | need to allocate additional memory in low-res heap" << std::endl;
        const dim3 threads_per_block_lowalloc(octree_branching_factor, 1);
        const dim3 n_blocks(low_blocks_to_allocate_, 1);
        allocateMemoryLow<<<n_blocks, threads_per_block_lowalloc>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      allocBlocks3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           max_integration_distance_,
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

      while (1) {
        resetHashBucketMutex();
        allocBlocks3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                             normals.deviceInstance(),
                                                             weights.deviceInstance(),
                                                             camera.deviceInstance(),
                                                             max_integration_distance_,
                                                             sdf_truncation_,
                                                             sdf_truncation_scale_,
                                                             d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
        int curr_free_blocks = getHeapHighFreeCount();
        curr_free_blocks += getHeapLowFreeCount();
        if (prev_free_blocks == curr_free_blocks) {
          break;
        }
        prev_free_blocks = curr_free_blocks;
      }

#endif
    }

    template <typename T>
    __global__ void integrateDepthMapKernel(const CUDAMatrixf* depth_img,
                                            const CUDAMatrixuc3* rgb_img,
                                            const Camera* camera,
                                            const float sdf_truncation,
                                            const float sdf_truncation_scale,
                                            const float max_integration_distance,
                                            const uchar integration_weight_sample,
                                            const uchar integration_weight_max,
                                            VoxelContainer<T>* container) {
      // we can access this linearly, we have compact representation
      const uint entry_idx   = blockIdx.x * blockDim.x + threadIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[entry_idx];
      if (entry.ptr == FREE_ENTRY)
        return;
      const int num_voxels = container->getNumVoxels(entry);
      const uint voxel_idx = blockIdx.y * blockDim.y + threadIdx.y;
      if (voxel_idx >= num_voxels)
        return;

      const int scaling_factor = 1 << entry.resolution;

      const int3 pi_base      = SDFBlockToVirtualVoxelPos(entry.pos);
      const int3 voxel_coords = scaling_factor * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scaling_factor));
      const int3 pi           = pi_base + voxel_coords;
      const float3 pf         = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi);

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // if depth is good
      const float depth = depth_img->at<1>(row, col);
      if (depth <= camera->minDepth() || depth > max_integration_distance)
        return;

      const float depth_normalized = camera->normalizeDepth(depth);

      float sdf              = depth - camera->getDepth(pcam);
      const float truncation = getTruncation(depth, sdf_truncation, sdf_truncation_scale);

      if (container->two_sided_surface_field_) {
        if (fabsf(sdf) > truncation)
          return;
        sdf = fabsf(sdf);
      } else {
        if (sdf <= -truncation)
          return;
        sdf = fminf(truncation, sdf);
      }

      // float weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - depth_normalized), 1);
      float weight_update = integration_weight_sample;

      // construct current voxel
      T curr;
      if (container->two_sided_surface_field_) {
        curr.sdf = sdf;
        curr.sum_squared = sdf * sdf;
        curr.weight = weight_update;
      } else {
        curr.sdf = sdf;
        curr.weight = weight_update;
        curr.rgb = rgb_img->at<1>(row, col);
      }

      // integrate
      const uint volume_idx = entry.ptr + voxel_idx;
      if (container->two_sided_surface_field_) {
        T merged_voxel;
        combineTwoSidedSurfaceVoxel(
          container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
        container->d_SDFBlocks_[volume_idx] = merged_voxel;
        return;
      }
      float curr_mean   = 0.f;
      float curr_sum_sq = 0.f;
      if (container->d_SDFBlocks_[volume_idx].weight > 0) {
        curr_mean   = container->d_SDFBlocks_[volume_idx].sdf;
        curr_sum_sq = container->d_SDFBlocks_[volume_idx].sum_squared;
      } else {
        curr_mean = sdf;
      }
      float delta = (sdf - curr_mean) / (container->virtual_voxel_size_ / 2);

      T merged_voxel;
      if (container->d_SDFBlocks_[volume_idx].weight == 0) {
        container->d_SDFBlocks_[volume_idx].rgb = curr.rgb;
      }
      combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
      container->d_SDFBlocks_[volume_idx] = merged_voxel;
      float delta2 = (sdf - container->d_SDFBlocks_[volume_idx].sdf) / (container->virtual_voxel_size_ / 2);
      atomicAdd(&container->d_SDFBlocks_[volume_idx].sum_squared, delta * delta2);
    }

    template <typename T>
    __global__ void checkVoxelsKernel(VoxelContainer<T>* container) {
      const HashEntry& entry = container->d_compactHashTable_[blockIdx.x];
      const int num_voxels   = container->getNumVoxels(entry);
      if (threadIdx.x >= num_voxels)
        return;
      const auto& voxel = container->d_SDFBlocks_[entry.ptr + threadIdx.x];
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateDepthMap(const CUDAMatrixf& depth_img,
                                                                                            const CUDAMatrixuc3& rgb_img,
                                                                                            const Camera& camera) {
      const dim3 threads_per_block(n_threads, n_threads, 1);
      const dim3 n_blocks((current_occupied_blocks_ + threads_per_block.x - 1) / threads_per_block.x,
                          (voxel_block_volume_ + threads_per_block.y - 1) / threads_per_block.y,
                          1);
      if (current_occupied_blocks_ > 0) {
        integrateDepthMapKernel<<<n_blocks, threads_per_block>>>(depth_img.deviceInstance(),
                                                                 rgb_img.deviceInstance(),
                                                                 camera.deviceInstance(),
                                                                 sdf_truncation_,
                                                                 sdf_truncation_scale_,
                                                                 max_integration_distance_,
                                                                 integration_weight_sample_,
                                                                 integration_weight_max_,
                                                                 d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void integrate3DKernel(const CUDAVectorf3* point_cloud,
                                      const CUDAVectorf3* normals,
                                      const CUDAVectorf* weights,
                                      const Camera* camera,
                                      const float sdf_truncation,
                                      const float sdf_truncation_scale,
                                      const float max_integration_distance,
                                      const uchar integration_weight_sample,
                                      const uchar integration_weight_max,
                                      VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3 pcam        = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float3 pw          = camera->camInWorld() * pcam;
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);
      if (range < 1e-6 || range > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float truncation = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth  = min(max_integration_distance, range - truncation);
      const float max_depth  = min(max_integration_distance, range + truncation);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min;
      float3 pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam - cam_dir * truncation;
        pcam_max = pcam + cam_dir * truncation;
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_min);
      int3 id_end           = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        // update weight and sdf value of id_current_voxel
        const int3 sdf_block_pos =
          virtualVoxelPosToSDFBlock(id_current_voxel, container->virtual_voxel_size_, container->voxel_extents_);
        const HashEntry& entry = container->getHashEntry(sdf_block_pos);
        if (entry.ptr != FREE_ENTRY) {
          const float3 actual_voxel_pos = virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel);
          const int scale               = 1 << entry.resolution;
          const int3 voxel_pos_aprox =
            make_int3(id_current_voxel.x / scale, id_current_voxel.y / scale, id_current_voxel.z / scale);
          const float3 voxel_pos        = virtualVoxelPosToWorld(container->getVoxelSize(entry), voxel_pos_aprox);
          const float3 voxel_pos_camera = camera->camInWorld().inverse() * voxel_pos;
          const float voxel_range       = norm3df(voxel_pos_camera.x, voxel_pos_camera.y, voxel_pos_camera.z);
          float sdf;
          if (container->projective_sdf_) {
            sdf = range - voxel_range;
          } else {
            sdf = dot((voxel_pos_camera - pcam), norm_dir);
          }
          if (sdf <= -truncation)
            break;
          if (sdf >= 0.f)
            sdf = fminf(truncation, sdf);
          else
            sdf = fmaxf(-truncation, sdf);

          const float range_normalized = camera->normalizeDepth(range);
          float weight_update;

          if (container->projective_sdf_) {
            // weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - range_normalized), 1);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          } else {
            // weight_update = fmaxf(1.5 * point_weight * integration_weight_sample, 1.f);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          }

          T curr;
          curr.sdf    = sdf;
          curr.weight = weight_update;

          // integrate
          const uint volume_idx = entry.ptr + virtualVoxelPosToSDFBlockIndex(id_current_voxel, sdf_block_size / scale);
          float curr_mean       = 0.f;
          float curr_sum_sq     = 0.f;
          if (container->d_SDFBlocks_[volume_idx].weight > 0) {
            curr_mean   = container->d_SDFBlocks_[volume_idx].sdf;
            curr_sum_sq = container->d_SDFBlocks_[volume_idx].sum_squared;
          }
          float delta = (sdf - curr_mean) / (container->virtual_voxel_size_ / 2);
          T merged_voxel;
          combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
          container->d_SDFBlocks_[volume_idx] = merged_voxel;
          float delta2 = (sdf - container->d_SDFBlocks_[volume_idx].sdf) / (container->virtual_voxel_size_ / 2);
          atomicAdd(&container->d_SDFBlocks_[volume_idx].sum_squared, delta * delta2);
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }
        iter++;
      }
    }

    template <typename T>
    __host__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrate3D(const CUDAVectorf3& point_cloud,
                                                                                               const CUDAVectorf3& normals,
                                                                                               const CUDAVectorf& weights,
                                                                                               const Camera& camera) {
      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      if (current_occupied_blocks_ > 0) {
        integrate3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void reintegrate3DKernel(const CUDAVectorf3* point_cloud,
                                        const CUDAVectorf3* normals,
                                        const CUDAVectorf* weights,
                                        const Camera* camera,
                                        const float sdf_truncation,
                                        const float sdf_truncation_scale,
                                        const float max_integration_distance,
                                        const uchar integration_weight_sample,
                                        const uchar integration_weight_max,
                                        VoxelContainer<T>* container) {
      const uint point_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (point_idx >= point_cloud->size())
        return;
      const float3 pcam        = point_cloud->at<1>(point_idx);
      const float3 normal      = normals->at<1>(3 * point_idx);
      const float point_weight = weights->at<1>(point_idx);
      const float3 pw          = camera->camInWorld() * pcam;
      const float range        = norm3df(pcam.x, pcam.y, pcam.z);
      if (range == 0.f || range > max_integration_distance) // from matrix initialization, emtpy is 0
        return;

      const float3 cam_dir  = normalize(pcam);
      const float3 norm_dir = normalize(normal);

      const float truncation = getTruncation(range, sdf_truncation, sdf_truncation_scale);
      const float min_depth  = min(max_integration_distance, range - truncation);
      const float max_depth  = min(max_integration_distance, range + truncation);

      if (min_depth >= max_depth)
        return;

      float3 pcam_min;
      float3 pcam_max;
      if (container->projective_sdf_) {
        pcam_min = pcam - cam_dir * truncation;
        pcam_max = pcam + cam_dir * truncation;
      } else {
        pcam_min = pcam + norm_dir * (min_depth - range);
        pcam_max = pcam + norm_dir * (max_depth - range);
      }

      float3 pw_min = camera->camInWorld() * pcam_min;
      float3 pw_max = camera->camInWorld() * pcam_max;

      float3 dir = normalize(pw_max - pw_min);

      int3 id_current_voxel = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_min);
      int3 id_end           = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, pw_max);

      float3 step = make_float3(sign(dir));

      float3 boundary_pos =
        virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel + make_int3(clamp(step, 0.0, 1.f))) -
        0.5f * container->virtual_voxel_size_;
      float3 t_max   = (boundary_pos - pw_min) / dir;
      float3 t_delta = (step * container->virtual_voxel_size_) / dir;
      int3 id_bound  = make_int3(make_float3(id_end) + step);

      if (fabsf(dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.x - dir.x) < FLOAT_EPSILON) {
        t_max.x   = numeric_limits<float>::max();
        t_delta.x = numeric_limits<float>::max();
      }

      if (fabsf(dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.y - dir.y) < FLOAT_EPSILON) {
        t_max.y   = numeric_limits<float>::max();
        t_delta.y = numeric_limits<float>::max();
      }

      if (fabsf(dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      if (fabsf(boundary_pos.z - dir.z) < FLOAT_EPSILON) {
        t_max.z   = numeric_limits<float>::max();
        t_delta.z = numeric_limits<float>::max();
      }
      // clang-format on

      uint iter = 0; // iter < max_loop_iter_count
#pragma unroll
      while (iter < max_dda_iteration_count) {
        const int3 sdf_block_pos =
          virtualVoxelPosToSDFBlock(id_current_voxel, container->virtual_voxel_size_, container->voxel_extents_);
        const HashEntry& entry = container->getHashEntryReintegrate(sdf_block_pos);
        if (entry.ptr != FREE_ENTRY) {
          const float3 actual_voxel_pos = virtualVoxelPosToWorld(container->virtual_voxel_size_, id_current_voxel);
          const int scale               = 1 << entry.resolution;
          const int3 voxel_pos_aprox =
            make_int3(id_current_voxel.x / scale, id_current_voxel.y / scale, id_current_voxel.z / scale);
          const float3 voxel_pos        = virtualVoxelPosToWorld(container->getVoxelSize(entry), voxel_pos_aprox);
          const float3 voxel_pos_camera = camera->camInWorld().inverse() * voxel_pos;
          const float voxel_range       = norm3df(voxel_pos_camera.x, voxel_pos_camera.y, voxel_pos_camera.z);
          float sdf;
          if (container->projective_sdf_) {
            sdf = range - voxel_range;
          } else {
            sdf = dot((voxel_pos_camera - pcam), norm_dir);
          }
          if (sdf <= -truncation)
            break;
          if (sdf >= 0.f)
            sdf = fminf(truncation, sdf);
          else
            sdf = fmaxf(-truncation, sdf);

          const float range_normalized = camera->normalizeDepth(range);
          float weight_update;

          if (container->projective_sdf_) {
            // weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - range_normalized), 1);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          } else {
            // weight_update = fmaxf(1.5 * point_weight * integration_weight_sample, 1.f);
            weight_update = integration_weight_sample; // Note: Could implement adaptive weighting
          }

          T curr;
          curr.sdf    = sdf;
          curr.weight = weight_update;

          // integrate
          const uint volume_idx = entry.ptr + virtualVoxelPosToSDFBlockIndex(id_current_voxel, sdf_block_size / scale);
          T merged_voxel;
          combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
          container->d_SDFBlocks_[volume_idx] = merged_voxel;
        }

        // traverse voxel grid
        if (t_max.x < t_max.y && t_max.x < t_max.z) {
          id_current_voxel.x += step.x;
          if (id_current_voxel.x == id_bound.x)
            return;
          t_max.x += t_delta.x;
        } else if (t_max.z < t_max.y) {
          id_current_voxel.z += step.z;
          if (id_current_voxel.z == id_bound.z)
            return;
          t_max.z += t_delta.z;
        } else {
          id_current_voxel.y += step.y;
          if (id_current_voxel.y == id_bound.y)
            return;
          t_max.y += t_delta.y;
        }
        iter++;
      }
    }

    template <typename T>
    __host__ void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reintegrate3D(const CUDAVectorf3& point_cloud,
                                                                                                 const CUDAVectorf3& normals,
                                                                                                 const CUDAVectorf& weights,
                                                                                                 const Camera& camera) {
      const dim3 threads_per_block(n_threads * n_threads, 1);
      const dim3 n_blocks((point_cloud.size() + threads_per_block.x - 1) / threads_per_block.x, 1);
      if (current_occupied_blocks_ > 0) {
        integrate3DKernel<<<n_blocks, threads_per_block>>>(point_cloud.deviceInstance(),
                                                           normals.deviceInstance(),
                                                           weights.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    // ! pack int and float into a single uint64
    __device__ __forceinline__ unsigned long long pack(int a, float b) {
      return (((unsigned long long) (*(reinterpret_cast<unsigned*>(&b)))) << 32) + *(reinterpret_cast<unsigned*>(&a));
    }

    // ! unpack an uint64 into an int and a float
    __device__ __forceinline__ void unpack(int& a, float& b, unsigned long long val) {
      unsigned ma = (unsigned) (val & 0x0FFFFFFFFULL);
      a           = *(reinterpret_cast<int*>(&ma));
      unsigned mb = (unsigned) (val >> 32);
      b           = *(reinterpret_cast<float*>(&mb));
    }

    // ! invalidate voxel kernel, if element is noisy (weight low) we set the weight to zero
    template <typename T>
    __global__ void starveVoxelsKernel(VoxelContainer<T>* container,
                                       const Camera* camera,
                                       const bool is_depth_buffer,
                                       CUDAMatrixu64* depth_buff) {
      const uint idx         = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[idx];

      const int3 pi_base = SDFBlockToVirtualVoxelPos(entry.pos);
      const uint i       = threadIdx.x; // inside an SDF block
      const int3 pi      = pi_base + make_int3(delinearizeVoxelPos(i));
      const float3 pf    = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi); // sdf block in world coordinates

      uint volume_idx = entry.ptr + i;
      const T& vox    = container->d_SDFBlocks_[volume_idx];

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      float depth = camera->getDepth(pcam);
      if (depth < camera->minDepth())
        return;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // z-buffer stuff
      const int unique_tid = blockDim.x * blockIdx.x + threadIdx.x;

      if (is_depth_buffer) {
        // unsigned long long candidate_depth_idx = pack(tid, depth);
        // depth buffer implemented as comparison between two uint64
        // packing depth and tid, with depth on the first 32 bits
        // this is required to make reproducible experiment
        // in this way, even if depth is the same, the one with lower tid is preferred
        atomicMin(&(depth_buff->at<1>(row, col)), pack(unique_tid, depth));
        return;
      }

      unsigned long long candidate_depth_idx = pack(unique_tid, depth);
      if (candidate_depth_idx != depth_buff->at<1>(row, col))
        return;

      // should be typically exectued only every n'th frame
      uchar weight                                            = container->d_SDFBlocks_[entry.ptr + threadIdx.x].weight;
      weight                                                  = max(0, weight - 1);
      container->d_SDFBlocks_[entry.ptr + threadIdx.x].weight = weight;
    }

    // ! invalidate voxel if weight is low, noisy elements
    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::starveVoxels(const Camera& camera) {
      if (init_buff) {
        depth_buff = CUDAMatrixu64(camera.rows(), camera.cols());
        init_buff  = false;
      }
      depth_buff.fill(numeric_limits<unsigned long long>::max(), true);
      if (current_occupied_blocks_ > 0) {
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const dim3 threads_per_block(voxel_block_volume_, 1);

        starveVoxelsKernel<<<n_blocks, threads_per_block>>>(
          d_instance_, camera.deviceInstance(), true, depth_buff.deviceInstance());
        CUDA_CHECK(cudaDeviceSynchronize());

        starveVoxelsKernel<<<n_blocks, threads_per_block>>>(
          d_instance_, camera.deviceInstance(), false, depth_buff.deviceInstance());
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void garbageCollectIdentifyKernel(const VoxelContainer<T>* container, const float truncation_threshold) {
      __shared__ float shared_minSDF[total_sdf_block_size / 2];
      __shared__ float shared_max_weight[total_sdf_block_size / 2];
      const uint idx         = blockIdx.x;
      const HashEntry& entry = container->d_compactHashTable_[idx];
      const int actual_pairs = container->getNumVoxels(entry) / 2;

      if (threadIdx.x >= actual_pairs)
        return;

      const uint idx0 = entry.ptr + 2 * threadIdx.x;
      const uint idx1 = entry.ptr + 2 * threadIdx.x + 1;

      T& v0 = container->d_SDFBlocks_[idx0];
      T& v1 = container->d_SDFBlocks_[idx1];

      float sdf0 = (v0.weight == 0) ? FLT_MAX : fabsf(v0.sdf);
      float sdf1 = (v1.weight == 0) ? FLT_MAX : fabsf(v1.sdf);

      shared_minSDF[threadIdx.x]     = fminf(sdf0, sdf1);
      shared_max_weight[threadIdx.x] = max(v0.weight, v1.weight);

      __syncthreads();

      // Binary tree reduction only over valid threads
      for (int stride = 1; stride < actual_pairs; stride <<= 1) {
        int idx = (threadIdx.x + 1) * (stride * 2) - 1;
        if (idx < actual_pairs && idx - stride >= 0) {
          shared_minSDF[idx]     = fminf(shared_minSDF[idx], shared_minSDF[idx - stride]);
          shared_max_weight[idx] = max(shared_max_weight[idx], shared_max_weight[idx - stride]);
        }
        __syncthreads();
      }

      if (threadIdx.x == actual_pairs - 1) {
        float minSDF                    = shared_minSDF[threadIdx.x];
        uchar maxWeight                 = shared_max_weight[threadIdx.x];
        container->d_hashDecision_[idx] = (minSDF >= truncation_threshold || maxWeight == 0) ? 1 : 0;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::garbageCollectIdentify(const Camera& camera) {
      if (current_occupied_blocks_ > 0) {
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const dim3 threads_per_block(voxel_block_volume_ / 2, 1);
        const float largest_truncation = getTruncation(camera.maxDepth(), sdf_truncation_, sdf_truncation_scale_);
        garbageCollectIdentifyKernel<<<n_blocks, threads_per_block>>>(d_instance_, largest_truncation);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __device__ bool
    VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::deleteHashEntryElement(const int3& sdf_block) {
      uint hash      = calculateHash(sdf_block); // hash bucket
      uint start_idx = hash * hash_bucket_size_; // hash position

      for (uint j = 0; j < hash_bucket_size_; j++) {
        uint i                       = start_idx + j;
        const HashEntry& curr        = d_hashTable_[i];
        const int voxel_per_side     = 1 << finest_block_log2_dim - curr.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
#ifndef RESOLVE_COLLISION
          if (curr.resolution == 0)
            appendHeapHigh(curr.ptr / voxel_block_volume);
          if (curr.resolution == 1)
            appendHeapLow(curr.ptr / voxel_block_volume);
          deleteHashEntry(d_hashTable_[i]);
          return true;
#endif
#ifdef RESOLVE_COLLISION
          if (curr.offset != 0) { // if there was a pointer set it to the next list element
            // InterlockedExchange(bucketMutex[h], LOCK_ENTRY, prevValue);	//lock the hash bucket
            int prev_value = atomicExch(&d_hashTableBucketMutex_[hash], LOCK_ENTRY);
            if (prev_value == LOCK_ENTRY) {
              return false;
            }

            if (prev_value != LOCK_ENTRY) {
              if (curr.resolution == 0)
                appendHeapHigh(curr.ptr / voxel_block_volume);
              if (curr.resolution == 1)
                appendHeapLow(curr.ptr / voxel_block_volume);
              int next_idx    = (i + curr.offset) % total_size_;
              d_hashTable_[i] = d_hashTable_[next_idx];
              deleteHashEntry(d_hashTable_[next_idx]);
              return true;
            }
          } else {
            if (curr.resolution == 0)
              appendHeapHigh(curr.ptr / voxel_block_volume);
            if (curr.resolution == 1)
              appendHeapLow(curr.ptr / voxel_block_volume);
            deleteHashEntry(d_hashTable_[i]);
            return true;
          }
#endif
        }
      }
#ifdef RESOLVE_COLLISION
      const uint idx_last_entry_in_bucket = (hash + 1) * hash_bucket_size_ - 1;
      int i                               = idx_last_entry_in_bucket;
      HashEntry curr;
      curr         = d_hashTable_[i];
      int prev_idx = i;
      i            = idx_last_entry_in_bucket + curr.offset; // go to next element in the list
      i %= total_size_;                                      // check for overflow

      uint max_iter            = 0;
      uint max_loop_iter_count = linked_list_size_;

#pragma unroll 1
      while (max_iter < max_loop_iter_count) {
        curr                         = d_hashTable_[i];
        const int voxel_per_side     = 1 << finest_block_log2_dim - curr.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;
        // found that dude that we need/want to delete
        if (curr.pos.x == sdf_block.x && curr.pos.y == sdf_block.y && curr.pos.z == sdf_block.z && curr.ptr != FREE_ENTRY) {
          int prev_value = atomicExch(&d_hashTableBucketMutex_[hash], LOCK_ENTRY);
          if (prev_value == LOCK_ENTRY) {
            return false;
          }
          if (prev_value != LOCK_ENTRY) {
            if (curr.resolution == 0)
              appendHeapHigh(curr.ptr / voxel_block_volume);
            if (curr.resolution == 1)
              appendHeapLow(curr.ptr / voxel_block_volume);
            deleteHashEntry(d_hashTable_[i]);
            HashEntry prev         = d_hashTable_[prev_idx];
            prev.offset            = curr.offset;
            d_hashTable_[prev_idx] = prev;
            return true;
          }
        }
        // we have found the end of the list
        // should actually never happen because we need to find that guy before
        if (curr.offset == 0) {
          return false;
        }

        prev_idx = i;
        i        = idx_last_entry_in_bucket + curr.offset; // go to next element in the list
        i %= total_size_;                                  // check for overflow

        max_iter++;
      }
#endif
      return false;
    }

    template <typename T>
    __global__ void garbageCollectFreeKernel(VoxelContainer<T>* container) {
      // const uint hashIdx = blockIdx.x;
      const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
      // if in bound and decision to delete the hash entry
      if (idx < container->current_occupied_blocks_ && container->d_hashDecision_[idx] != 0) {
        const HashEntry& entry       = container->d_compactHashTable_[idx];
        const int voxel_per_side     = 1 << finest_block_log2_dim - entry.resolution;
        const int voxel_block_volume = voxel_per_side * voxel_per_side * voxel_per_side;

        // delete hash entry from hash (and performs heap append)
        if (container->deleteHashEntryElement(entry.pos)) {
#pragma unroll 1
          for (uint i = 0; i < voxel_block_volume; ++i) {
            deleteVoxel<T>(container->d_SDFBlocks_[entry.ptr + i]);
          }
        }
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::garbageCollectFree() {
      if (current_occupied_blocks_ > 0) {
        const dim3 threads_per_block(n_threads_cam * n_threads_cam, 1);
        const dim3 n_blocks((current_occupied_blocks_ + threads_per_block.x - 1) / threads_per_block.x, 1);
        garbageCollectFreeKernel<<<n_blocks, threads_per_block>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void checkVarSDFKernel(VoxelContainer<T>* container) {
      const uint block_idx = blockIdx.x;
      if (block_idx >= container->current_occupied_blocks_) {
        printf("block_idx>=current_occupied_blocks_");
        return;
      }

      const HashEntry& entry = container->d_compactHashTable_[block_idx];
      if (entry.resolution >= 1)
        return;

      int tid = threadIdx.x;

      const int resolution_factor  = 1 << (finest_block_log2_dim - entry.resolution);
      const int voxel_block_volume = resolution_factor * resolution_factor * resolution_factor;
      if (tid >= voxel_block_volume) {
        return;
      }

      extern __shared__ float shared[];
      float* sum_sq_shared = shared;
      float* weight_shared = shared + blockDim.x;

      float local_sum_sq = 0.f;
      float local_weight = 0.f;

      int gx = (tid % 4) * 2;
      int gy = ((tid / 4) % 4) * 2;
      int gz = (tid / 16) * 2;

      for (int dz = 0; dz < 2; ++dz)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx) {
            int x = gx + dx;
            int y = gy + dy;
            int z = gz + dz;

            int local_idx   = z * octree_branching_factor * octree_branching_factor + y * octree_branching_factor + x;
            uint volume_idx = entry.ptr + local_idx;

            const T& voxel = container->d_SDFBlocks_[volume_idx];

            if (voxel.weight > 0) {
              local_sum_sq += voxel.sum_squared;
              local_weight += voxel.weight;
            }
          }

      sum_sq_shared[threadIdx.x] = local_sum_sq;
      weight_shared[threadIdx.x] = local_weight;

      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
          sum_sq_shared[threadIdx.x] += sum_sq_shared[threadIdx.x + stride];
          weight_shared[threadIdx.x] += weight_shared[threadIdx.x + stride];
        }
        __syncthreads();
      }

      __syncthreads();

      if (threadIdx.x == 0) {
        if (weight_shared[0] < 2)
          return;
        double avg_var = sum_sq_shared[0] / (weight_shared[0] - 1);

        if ((weight_shared[0] - 1) > 1e-6f && avg_var > 0.f && avg_var < container->sdf_var_threshold_) {
          const int3 block_pos = entry.pos;
          const int resolution = entry.resolution + 1;
          if (container->deleteHashEntryElement(block_pos)) {
#pragma unroll 1
            for (uint i = 0; i < voxel_block_volume; ++i) {
              deleteVoxel<T>(container->d_SDFBlocks_[entry.ptr + i]);
            }
            const uint reallocate_idx                    = atomicAdd(container->d_num_reallocate_, 1);
            container->d_reallocate_pos_[reallocate_idx] = block_pos;
            container->d_reallocate_res_[reallocate_idx] = resolution;
          }
        }
      }
    }

    template <typename T>
    __global__ void reintegrateDepthMapKernel(const CUDAMatrixf* depth_img,
                                              const CUDAMatrixuc3* rgb_img,
                                              const Camera* camera,
                                              const float sdf_truncation,
                                              const float sdf_truncation_scale,
                                              const float max_integration_distance,
                                              const uchar integration_weight_sample,
                                              const uchar integration_weight_max,
                                              VoxelContainer<T>* container) {
      // we can access this linearly, we have compact representation
      const uint entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (entry_idx >= container->d_num_reintegrate_[0])
        return;
      const uint reintegrate_idx = container->d_reintegrate_[entry_idx];
      const HashEntry& entry     = container->d_hashTable_[reintegrate_idx];
      const int num_voxels       = container->getNumVoxels(entry);
      const uint voxel_idx       = blockIdx.y * blockDim.y + threadIdx.y;
      if (voxel_idx >= num_voxels)
        return;

      const int scaling_factor = 1 << entry.resolution;

      const int3 pi_base      = SDFBlockToVirtualVoxelPos(entry.pos);
      const int3 voxel_coords = scaling_factor * make_int3(delinearizeVoxelPos(voxel_idx, sdf_block_size / scaling_factor));
      const int3 pi           = pi_base + voxel_coords;
      const float3 pf         = virtualVoxelPosToWorld(container->virtual_voxel_size_, pi);

      // get point in screen
      const float3 pcam = camera->camInWorld().inverse() * pf;

      int2 img_point;
      bool is_good = camera->projectPoint(pcam, img_point);

      if (!is_good)
        return;

      const auto& row = img_point.x;
      const auto& col = img_point.y;

      // if depth is good
      const float depth = depth_img->at<1>(row, col);
      if (depth <= camera->minDepth() || depth > max_integration_distance)
        return;

      const float depth_normalized = camera->normalizeDepth(depth);

      float sdf              = depth - camera->getDepth(pcam);
      const float truncation = getTruncation(depth, sdf_truncation, sdf_truncation_scale);

      if (container->two_sided_surface_field_) {
        if (fabsf(sdf) > truncation)
          return;
        sdf = fabsf(sdf);
      } else {
        if (sdf <= -truncation)
          return;
        sdf = fminf(truncation, sdf);
      }

      // float weight_update = fmaxf(integration_weight_sample * 1.5f * (1.f - depth_normalized), 1);
      float weight_update = integration_weight_sample;

      // construct current voxel
      T curr;
      if (container->two_sided_surface_field_) {
        curr.sdf = sdf;
        curr.sum_squared = sdf * sdf;
        curr.weight = weight_update;
      } else {
        curr.sdf = sdf;
        curr.weight = weight_update;
        curr.rgb = rgb_img->at<1>(row, col);
      }

      // integrate
      uint volume_idx = entry.ptr + voxel_idx;

      T merged_voxel;
      if (container->two_sided_surface_field_)
        combineTwoSidedSurfaceVoxel(
          container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
      else {
        if (container->d_SDFBlocks_[volume_idx].weight == 0)
          container->d_SDFBlocks_[volume_idx].rgb = curr.rgb;
        combineVoxel<T>(container->d_SDFBlocks_[volume_idx], curr, integration_weight_max, merged_voxel);
      }
      container->d_SDFBlocks_[volume_idx] = merged_voxel;
    }

    template <typename T>
    __global__ void reallocBlocksKernel(VoxelContainer<T>* container) {
      const uint entry_idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (entry_idx >= container->d_num_reallocate_[0])
        return;

      const int3& entry_pos = container->d_reallocate_pos_[entry_idx];
      const int& resolution = container->d_reallocate_res_[entry_idx];

      const int& allocation_idx = container->reallocBlock(entry_pos, resolution);
      if (allocation_idx >= 0) {
        const uint reintegrate_idx                 = atomicAdd(container->d_num_reintegrate_, 1);
        container->d_reintegrate_[reintegrate_idx] = allocation_idx;
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reallocBlocks() {
      uint num_reallocate = 0;
      CUDA_CHECK(cudaMemcpy(&num_reallocate, d_num_reallocate_, sizeof(uint), cudaMemcpyDeviceToHost));
      if (num_reallocate > 0) {
        CUDA_CHECK(cudaMemset(d_reintegrate_, 0, sizeof(uint) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_num_reintegrate_, 0, sizeof(uint)));
        int prev_free_blocks = getHeapHighFreeCount();
        prev_free_blocks += getHeapLowFreeCount();
        resetHashBucketMutex();
        const dim3 threads_per_block(64, 1);
        const dim3 n_blocks((num_reallocate + threads_per_block.x - 1) / threads_per_block.x);
        reallocBlocksKernel<<<n_blocks, threads_per_block>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());

#ifdef RESOLVE_CONFLICT_ALLOCATIONS

        while (1) {
          resetHashBucketMutex();

          reallocBlocksKernel<<<n_blocks, threads_per_block>>>(d_instance_);

          CUDA_CHECK(cudaDeviceSynchronize());
          int curr_free_blocks = getHeapHighFreeCount();
          curr_free_blocks += getHeapLowFreeCount();
          if (prev_free_blocks == curr_free_blocks) {
            break;
          }
          prev_free_blocks = curr_free_blocks;
        }

#endif
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::checkVarSDF() {
      if (current_occupied_blocks_ > 0) {
        resetHashBucketMutex();
        CUDA_CHECK(cudaMemset(d_reallocate_pos_, 0, sizeof(int3) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_reallocate_res_, 0, sizeof(int) * num_sdf_blocks_));
        CUDA_CHECK(cudaMemset(d_num_reallocate_, 0, sizeof(uint)));
        const dim3 threads_per_block(64, 1);
        const dim3 n_blocks(current_occupied_blocks_, 1);
        const size_t shared_mem_size = 2 * threads_per_block.x * sizeof(float);
        checkVarSDFKernel<<<n_blocks, threads_per_block, shared_mem_size>>>(d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::reintegrateDepthMap(const CUDAMatrixf& depth_img,
                                                                                              const CUDAMatrixuc3& rgb_img,
                                                                                              const Camera& camera) {
      uint num_reintegrate = 0;
      CUDA_CHECK(cudaMemcpy(&num_reintegrate, d_num_reintegrate_, sizeof(uint), cudaMemcpyDeviceToHost));
      if (num_reintegrate > 0) {
        const dim3 threads_per_block(n_threads, n_threads, 1);
        const dim3 n_blocks((num_reintegrate + threads_per_block.x - 1) / threads_per_block.x,
                            (voxel_block_volume_ + threads_per_block.y - 1) / threads_per_block.y,
                            1);
        reintegrateDepthMapKernel<<<n_blocks, n_threads>>>(depth_img.deviceInstance(),
                                                           rgb_img.deviceInstance(),
                                                           camera.deviceInstance(),
                                                           sdf_truncation_,
                                                           sdf_truncation_scale_,
                                                           max_integration_distance_,
                                                           integration_weight_sample_,
                                                           integration_weight_max_,
                                                           d_instance_);
      }
    }

    // ! inserts a hash entry without allocating any memory: used by streaming:
    // ! entry must be modifiable, but just inside the scope of this function, not actually returned
    template <typename T>
    __device__ bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::insertHashEntry(HashEntry entry) {
      uint h  = calculateHash(entry.pos);
      uint hp = h * hash_bucket_size_;

      for (uint j = 0; j < hash_bucket_size_; ++j) {
        uint i          = j + hp;
        int prev_weight = 0;
        prev_weight     = atomicCAS(&d_hashTable_[i].ptr, FREE_ENTRY, LOCK_ENTRY);
        if (prev_weight == FREE_ENTRY) {
          d_hashTable_[i] = entry;
          return true;
        }
      }

#ifdef RESOLVE_COLLISION
      const uint idx_last_entry_in_bucket = (h + 1) * hash_bucket_size_ - 1; // get last index of bucket
      int offset = 0;
#pragma unroll 1
      for (uint attempt = 0; attempt < linked_list_size_; ++attempt) {
        offset++;
        const uint i = (idx_last_entry_in_bucket + offset) % total_size_;
        if ((offset % hash_bucket_size_) == 0)
          continue;

        if (atomicCAS(&d_hashTable_[i].ptr, FREE_ENTRY, LOCK_ENTRY) == FREE_ENTRY) {
          entry.offset = atomicExch(&d_hashTable_[idx_last_entry_in_bucket].offset, offset);
          d_hashTable_[i] = entry;
          return true;
        }
      }
#endif

      printf("insertHashEntry: could not insert entry with hash %d and pos %d %d %d\n", h, entry.pos.x, entry.pos.y, entry.pos.z);

      return false;
    }

  } // namespace cugeoutils
} // namespace cupanutils
