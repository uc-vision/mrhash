#include "streamer.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace cupanutils {
  namespace cugeoutils {

    ///////////////////////////////////////////////////////////////////////////
    // streaming from device to host: copies an entire useless stuff in host //
    ///////////////////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void integrateFromGlobalHashPass1Kernel(const float radius,
                                                       const float3 camera_position,
                                                       const int start_idx,
                                                       uint* d_outputCounter,
                                                       SDFBlockDesc* d_output,
                                                       VoxelContainer<T>* container) {
      const uint bucket_id = start_idx + blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= container->total_size_)
        return;

      HashEntry& entry = container->d_hashTable_[bucket_id];

      float3 pw = SDFBlockToWorldPoint(container->virtual_voxel_size_, entry.pos);
      float d   = length(pw, camera_position);

      if (entry.ptr != FREE_ENTRY && d >= radius) {
        SDFBlockDesc desc(entry);

#ifndef RESOLVE_COLLISION
        uint addr      = atomicAdd(&d_outputCounter[0], 1);
        d_output[addr] = desc;

        if (entry.resolution == 0)
          container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
        if (entry.resolution == 1)
          container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
        deleteHashEntry(bucket_id);
#endif
#ifdef RESOLVE_COLLISION
        if (entry.offset != 0 || container->calculateHash(entry.pos) != bucket_id / container->hash_bucket_size_) {
          if (container->deleteHashEntryElement(entry.pos)) {
            uint addr      = atomicAdd(&d_outputCounter[0], 1);
            d_output[addr] = desc;
          }
        } else {
          uint addr      = atomicAdd(&d_outputCounter[0], 1);
          d_output[addr] = desc;
          if (entry.resolution == 0)
            container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
          else if (entry.resolution == 1)
            container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
          deleteHashEntry(entry);
        }
#endif
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass1(const float radius,
                                                                                                 const float3& camera_position) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      int stream_size = container_->total_size_;
      const dim3 n_blocks((stream_size + threads_per_block.x - 1) / threads_per_block.x, 1);
      int start_idx = 0;

      if (stream_size > 0) {
        integrateFromGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          radius, camera_position, start_idx, d_SDF_block_counter_, d_SDFBlockDescOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass1(const float radius,
                                                                                                 const float3& camera_position,
                                                                                                 const int num_pass) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      int stream_size = max_num_sdf_block_integrate_from_global_hash_;
      const dim3 n_blocks((stream_size + threads_per_block.x - 1) / threads_per_block.x, 1);
      int start_idx = num_pass * max_num_sdf_block_integrate_from_global_hash_;

      if (stream_size > 0) {
        integrateFromGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          radius, camera_position, start_idx, d_SDF_block_counter_, d_SDFBlockDescOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    __global__ void integrateFromGlobalHashPass1Kernel(const int start_idx,
                                                       uint* d_outputCounter,
                                                       SDFBlockDesc* d_output,
                                                       VoxelContainer<T>* container) {
      const uint bucket_id = start_idx + blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= container->total_size_)
        return;

      HashEntry& entry = container->d_hashTable_[bucket_id];

      if (entry.ptr != FREE_ENTRY) {
        SDFBlockDesc desc(entry);

#ifndef RESOLVE_COLLISION
        uint addr      = atomicAdd(&d_outputCounter[0], 1);
        d_output[addr] = desc;

        if (entry.resolution == 0)
          container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
        if (entry.resolution == 1)
          container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
        deleteHashEntry(bucket_id);
#endif
#ifdef RESOLVE_COLLISION
        if (entry.offset != 0 || container->calculateHash(entry.pos) != bucket_id / container->hash_bucket_size_) {
          if (container->deleteHashEntryElement(entry.pos)) {
            uint addr      = atomicAdd(&d_outputCounter[0], 1);
            d_output[addr] = desc;
          }
        } else {
          uint addr      = atomicAdd(&d_outputCounter[0], 1);
          d_output[addr] = desc;
          if (entry.resolution == 0)
            container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
          else if (entry.resolution == 1)
            container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
          deleteHashEntry(entry);
        }
#endif
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass1(const int num_pass) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      int stream_size = max_num_sdf_block_integrate_from_global_hash_;
      const dim3 n_blocks((stream_size + threads_per_block.x - 1) / threads_per_block.x, 1);
      int start_idx = num_pass * max_num_sdf_block_integrate_from_global_hash_;

      if (stream_size > 0) {
        integrateFromGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          start_idx, d_SDF_block_counter_, d_SDFBlockDescOutput_, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    __device__ int worldToChunkCoordinate(const float world, const float extent) {
      const float scaled = world / extent;
      const float sign = static_cast<float>((scaled > 0.f) - (scaled < 0.f));
      return static_cast<int>(scaled + sign * 0.5f);
    }

    template <typename T>
    __device__ bool isDiscardedChunk(const HashEntry& entry,
                                     const int3* chunks,
                                     const uint chunk_count,
                                     const float3 chunk_extents,
                                     VoxelContainer<T>* container) {
      if (entry.ptr == FREE_ENTRY)
        return false;
      const float3 world = SDFBlockToWorldPoint(container->virtual_voxel_size_, entry.pos);
      const int3 chunk = make_int3(worldToChunkCoordinate(world.x, chunk_extents.x),
                                   worldToChunkCoordinate(world.y, chunk_extents.y),
                                   worldToChunkCoordinate(world.z, chunk_extents.z));
      for (uint index = 0; index < chunk_count; ++index) {
        const int3 candidate = chunks[index];
        if (candidate.x == chunk.x && candidate.y == chunk.y && candidate.z == chunk.z)
          return true;
      }
      return false;
    }

    template <typename T>
    __device__ void releaseHashEntry(HashEntry& entry, VoxelContainer<T>* container) {
      if (entry.resolution == 0)
        container->appendHeapHigh(entry.ptr / container->getNumVoxels(entry));
      else if (entry.resolution == 1)
        container->appendHeapLow(entry.ptr / container->getNumVoxels(entry));
    }

    template <typename T>
    __global__ void discardChunksKernel(const int3* chunks,
                                        const uint chunk_count,
                                        const float3 chunk_extents,
                                        VoxelContainer<T>* container) {
      const uint bucket = blockIdx.x * blockDim.x + threadIdx.x;
      if (bucket >= container->hash_num_buckets_)
        return;
      const uint bucket_start = bucket * container->hash_bucket_size_;
      const uint bucket_last = bucket_start + container->hash_bucket_size_ - 1;
      for (uint index = bucket_start; index < bucket_last; ++index) {
        HashEntry& entry = container->d_hashTable_[index];
        if (entry.ptr != FREE_ENTRY && container->calculateHash(entry.pos) == bucket &&
            isDiscardedChunk(entry, chunks, chunk_count, chunk_extents, container)) {
          releaseHashEntry(entry, container);
          deleteHashEntry(entry);
        }
      }

      uint index = bucket_last;
      uint previous_index = bucket_last;
      for (uint iteration = 0; iteration < container->linked_list_size_; ++iteration) {
        HashEntry& entry = container->d_hashTable_[index];
        const int next_offset = entry.offset;
        if (isDiscardedChunk(entry, chunks, chunk_count, chunk_extents, container)) {
          releaseHashEntry(entry, container);
          if (index == bucket_last && next_offset != 0) {
            const uint next_index = (bucket_last + next_offset) % container->total_size_;
            entry = container->d_hashTable_[next_index];
            deleteHashEntry(container->d_hashTable_[next_index]);
            continue;
          }
          container->d_hashTable_[previous_index].offset = next_offset;
          deleteHashEntry(entry);
        }
        if (next_offset == 0)
          break;
        if (entry.ptr != FREE_ENTRY)
          previous_index = index;
        index = (bucket_last + next_offset) % container->total_size_;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::discardChunks(
      const std::vector<Eigen::Vector3i>& chunks) {
      std::vector<int3> host_chunks;
      host_chunks.reserve(chunks.size());
      for (const Eigen::Vector3i& chunk : chunks)
        host_chunks.push_back(Eig2CUDA(chunk));
      int3* device_chunks = nullptr;
      CUDA_CHECK(cudaMalloc(&device_chunks, sizeof(int3) * host_chunks.size()));
      CUDA_CHECK(cudaMemcpy(
        device_chunks, host_chunks.data(), sizeof(int3) * host_chunks.size(), cudaMemcpyHostToDevice));
      const dim3 threads(n_threads_reduce_hashtable, 1);
      const dim3 blocks((container_->hash_num_buckets_ + threads.x - 1) / threads.x, 1);
      discardChunksKernel<<<blocks, threads>>>(
        device_chunks, host_chunks.size(), Eig2CUDA(voxel_extents_), container_->d_instance_);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaFree(device_chunks));
    }

    template <typename T>
    __global__ void prepareIntegrateFromGlobalHashPass2Kernel(uint start_idx,
                                                              const uint num_SDF_block_desc,
                                                              const SDFBlockDesc* d_SDFBlockDescs,
                                                              uint* num_voxels_vec,
                                                              VoxelContainer<T>* container) {
      const uint idx_block = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
      if (idx_block >= num_SDF_block_desc)
        return;

      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[idx_block];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;

      num_voxels_vec[idx_block] = (uint) num_voxels;
    }

    template <typename T>
    __global__ void integrateFromGlobalHashPass2Kernel(const int start_idx,
                                                       const uint num_SDF_block_desc,
                                                       const SDFBlockDesc* d_SDFBlockDescs,
                                                       const uint* voxel_offsets,
                                                       const bool fixed_block_size,
                                                       T* d_output,
                                                       VoxelContainer<T>* container) {
      const uint idx_block = start_idx + blockIdx.x;

      if (idx_block >= num_SDF_block_desc)
        return;

      const SDFBlockDesc& desc = d_SDFBlockDescs[idx_block];
      const int scale          = 1 << (finest_block_log2_dim - desc.resolution);
      const int num_voxels     = scale * scale * scale;

      if (threadIdx.x >= num_voxels)
        return;

      const uint output_offset = fixed_block_size ? idx_block * container->voxel_block_volume_ : voxel_offsets[idx_block];

      // Copy SDF block to CPU
      d_output[output_offset + threadIdx.x] = container->d_SDFBlocks_[desc.ptr + threadIdx.x];

      // reset SDF block
      deleteVoxel<T>(container->d_SDFBlocks_[desc.ptr + threadIdx.x]);
    }

    template <typename T>
    __global__ void integrateFromGlobalHashPass2CompactKernel(const uint num_sdf_blocks,
                                                              const SDFBlockDesc* block_descs,
                                                              CompactVoxel* output,
                                                              VoxelContainer<T>* container) {
      const uint block_index = blockIdx.x;
      if (block_index >= num_sdf_blocks)
        return;
      const SDFBlockDesc& desc = block_descs[block_index];
      const int scale = 1 << (finest_block_log2_dim - desc.resolution);
      const int num_voxels = scale * scale * scale;
      if (threadIdx.x >= num_voxels)
        return;
      T& voxel = container->d_SDFBlocks_[desc.ptr + threadIdx.x];
      const ushort sdf = __half_as_ushort(__float2half(voxel.sdf));
      const ushort secondary = __half_as_ushort(__float2half(voxel.sum_squared));
      output[block_index * container->voxel_block_volume_ + threadIdx.x] =
        CompactVoxel{static_cast<uchar>(sdf),
                     static_cast<uchar>(sdf >> 8),
                     voxel.weight,
                     static_cast<uchar>(secondary),
                     static_cast<uchar>(secondary >> 8),
                     voxel.rgb.x};
      deleteVoxel<T>(voxel);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass2(const uint num_SDF_block_desc) {
      const dim3 n_blocks_prepare((num_SDF_block_desc + n_threads - 1) / n_threads, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);
      const dim3 n_blocks(num_SDF_block_desc, 1);
      int start_idx = 0;

      if (num_SDF_block_desc > 0) {
        if (!compact_host_voxels_) {
          prepareIntegrateFromGlobalHashPass2Kernel<<<n_blocks_prepare, n_threads>>>(
            start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, d_voxel_offsets_, container_->d_instance_);
          thrust::device_ptr<uint> offsets(d_voxel_offsets_);
          thrust::exclusive_scan(thrust::device, offsets, offsets + num_SDF_block_desc, offsets);
        }
        if (compact_host_voxels_) {
          integrateFromGlobalHashPass2CompactKernel<<<n_blocks, threads_per_block>>>(
            num_SDF_block_desc, d_SDFBlockDescOutput_, d_compact_voxel_output_, container_->d_instance_);
        } else {
          integrateFromGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
            start_idx,
            num_SDF_block_desc,
            d_SDFBlockDescOutput_,
            d_voxel_offsets_,
            false,
            d_SDFBlockOutput_,
            container_->d_instance_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateFromGlobalHashPass2(const uint num_SDF_block_desc,
                                                                                                 const int num_pass) {
      const dim3 n_blocks_prepare((num_SDF_block_desc + n_threads - 1) / n_threads, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);
      int stream_size = num_SDF_block_desc;
      const dim3 n_blocks(stream_size, 1);
      int start_idx = 0;

      if (stream_size > 0) {
        if (!compact_host_voxels_) {
          prepareIntegrateFromGlobalHashPass2Kernel<<<n_blocks_prepare, n_threads>>>(
            start_idx, num_SDF_block_desc, d_SDFBlockDescOutput_, d_voxel_offsets_, container_->d_instance_);
          thrust::device_ptr<uint> offsets(d_voxel_offsets_);
          thrust::exclusive_scan(thrust::device, offsets, offsets + num_SDF_block_desc, offsets);
        }
        if (compact_host_voxels_) {
          integrateFromGlobalHashPass2CompactKernel<<<n_blocks, threads_per_block>>>(
            num_SDF_block_desc, d_SDFBlockDescOutput_, d_compact_voxel_output_, container_->d_instance_);
        } else {
          integrateFromGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
            start_idx,
            num_SDF_block_desc,
            d_SDFBlockDescOutput_,
            d_voxel_offsets_,
            false,
            d_SDFBlockOutput_,
            container_->d_instance_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    //////////////////////////////////////////////////////////////////////////
    // streaming from host to device: copies an entire chunk back to device //
    //////////////////////////////////////////////////////////////////////////

    template <typename T>
    __global__ void
    prepareChunkToGlobalHashPass1Kernel(const SDFBlockDesc* d_SDFBlockDescs, uint* num_blocks_vec, VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_blocks               = scale * scale * scale;
      num_blocks_vec[blockIdx.x]         = (uint) num_blocks;
    }

    //-------------------------------------------------------
    // Pass 1: Allocate memory
    //-------------------------------------------------------

    template <typename T>
    __global__ void chunkToGlobalHashPass1Kernel(const uint num_sdf_blocks_descs,
                                                 const uint heap_count_prev,
                                                 const SDFBlockDesc* d_SDFBlockDescs,
                                                 uint* d_blocks_ptr,
                                                 uchar* d_merge_blocks,
                                                 VoxelContainer<T>* container) {
      const unsigned int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;

      if (bucket_id >= num_sdf_blocks_descs)
        return;

      HashEntry entry;
      entry.pos            = d_SDFBlockDescs[bucket_id].pos;
      entry.offset         = 0;
      entry.resolution     = d_SDFBlockDescs[bucket_id].resolution;
      const HashEntry existing = container->getHashEntry(entry.pos);
      if (existing.ptr != FREE_ENTRY) {
        d_blocks_ptr[bucket_id] = existing.ptr;
        d_merge_blocks[bucket_id] = 1;
        return;
      }
      const int scale      = 1 << (finest_block_log2_dim - entry.resolution);
      const int num_voxels = scale * scale * scale;
      int ptr              = -1;
      if (entry.resolution == 0)
        ptr = container->consumeHeapHigh() * num_voxels;
      else if (entry.resolution == 1)
        ptr = container->consumeHeapLow() * num_voxels;
      d_blocks_ptr[bucket_id] = ptr;
      d_merge_blocks[bucket_id] = 0;
      entry.ptr               = ptr;

      // next kernel will randomly fill memory)
      bool is_inserted = container->insertHashEntry(entry);
      if (!is_inserted && entry.pos.x != 0 && entry.pos.y != 0 && entry.pos.z != 0)
        printf("WARNING entry [ %d %d %d ] not inserted, possible memory leak!\n", entry.pos.x, entry.pos.y, entry.pos.z);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunkToGlobalHashPass1(const uint num_sdf_blocks_descs,
                                                                                           const uint heap_count_prev,
                                                                                           const SDFBlockDesc* d_SDFBlockDescs,
                                                                                           uint* d_blocks_ptr,
                                                                                           uchar* d_merge_blocks) {
      const dim3 threads_per_block((n_threads * n_threads), 1);
      const dim3 n_blocks((num_sdf_blocks_descs + threads_per_block.x - 1) / threads_per_block.x, 1);

      if (num_sdf_blocks_descs > 0) {
        chunkToGlobalHashPass1Kernel<<<n_blocks, threads_per_block>>>(
          num_sdf_blocks_descs, heap_count_prev, d_SDFBlockDescs, d_blocks_ptr, d_merge_blocks, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    //-------------------------------------------------------
    // Pass 2: Copy input to SDFBlocks
    //-------------------------------------------------------

    template <typename T>
    __global__ void
    prepareChunkToGlobalHashPass2Kernel(const SDFBlockDesc* d_SDFBlockDescs, uint* num_voxels_vec, VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;
      num_voxels_vec[blockIdx.x]         = (uint) num_voxels;
    }

    template <typename T>
    __global__ void chunkToGlobalHashPass2Kernel(const uint heap_count_prev,
                                                 const SDFBlockDesc* d_SDFBlockDescs,
                                                 const uint* voxel_offsets,
                                                 const bool fixed_block_size,
                                                 const T* d_SDFBlocks,
                                                 const uint* d_blocks_ptr,
                                                 const uchar* merge_blocks,
                                                 VoxelContainer<T>* container) {
      const SDFBlockDesc& sdf_block_desc = d_SDFBlockDescs[blockIdx.x];
      const int scale                    = 1 << (finest_block_log2_dim - sdf_block_desc.resolution);
      const int num_voxels               = scale * scale * scale;

      if (threadIdx.x >= num_voxels)
        return;

      const uint input_offset = fixed_block_size ? blockIdx.x * container->voxel_block_volume_ : voxel_offsets[blockIdx.x];
      const uint ptr                             = d_blocks_ptr[blockIdx.x];
      const T& input = d_SDFBlocks[input_offset + threadIdx.x];
      T& output = container->d_SDFBlocks_[ptr + threadIdx.x];
      const bool input_observed = input.weight > 0 ||
                                  (container->two_sided_surface_field_ && input.rgb.x > 0);
      const bool output_observed = output.weight > 0 ||
                                   (container->two_sided_surface_field_ && output.rgb.x > 0);
      if (merge_blocks[blockIdx.x] && input_observed) {
        if (output_observed) {
          T merged;
          if (container->two_sided_surface_field_)
            combineTwoSidedSurfaceVoxel(output, input, container->integration_weight_max_, merged);
          else
            combineVoxel(output, input, container->integration_weight_max_, merged);
          output = merged;
        } else {
          output = input;
        }
      } else if (!merge_blocks[blockIdx.x]) {
        output = input;
      }
    }

    template <typename T>
    __global__ void chunkToGlobalHashPass2CompactKernel(const SDFBlockDesc* block_descs,
                                                        const CompactVoxel* voxels,
                                                        const uint* block_pointers,
                                                        const uchar* merge_blocks,
                                                        VoxelContainer<T>* container) {
      const SDFBlockDesc& desc = block_descs[blockIdx.x];
      const int scale = 1 << (finest_block_log2_dim - desc.resolution);
      const int num_voxels = scale * scale * scale;
      if (threadIdx.x >= num_voxels)
        return;
      const CompactVoxel& compact = voxels[blockIdx.x * container->voxel_block_volume_ + threadIdx.x];
      const ushort sdf = compact.sdf_low | (compact.sdf_high << 8);
      const ushort secondary = compact.secondary_low | (compact.secondary_high << 8);
      T voxel;
      voxel.sdf = __half2float(__ushort_as_half(sdf));
      voxel.weight = compact.weight;
      voxel.sum_squared = __half2float(__ushort_as_half(secondary));
      voxel.rgb.x = compact.secondary_weight;
      T& output = container->d_SDFBlocks_[block_pointers[blockIdx.x] + threadIdx.x];
      const bool voxel_observed = voxel.weight > 0 ||
                                  (container->two_sided_surface_field_ && voxel.rgb.x > 0);
      const bool output_observed = output.weight > 0 ||
                                   (container->two_sided_surface_field_ && output.rgb.x > 0);
      if (merge_blocks[blockIdx.x] && voxel_observed) {
        if (output_observed) {
          T merged;
          if (container->two_sided_surface_field_)
            combineTwoSidedSurfaceVoxel(output, voxel, container->integration_weight_max_, merged);
          else
            combineVoxel(output, voxel, container->integration_weight_max_, merged);
          output = merged;
        } else {
          output = voxel;
        }
      } else if (!merge_blocks[blockIdx.x]) {
        output = voxel;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunkToGlobalHashPass2(const uint num_sdf_blocks_descs,
                                                                                           const uint heap_count_prev,
                                                                                           const SDFBlockDesc* d_SDFBlockDescs,
                                                                                           const T* d_SDFBlocks,
                                                                                           uint* d_heap_ptr,
                                                                                           const uchar* merge_blocks) {
      const dim3 n_blocks(num_sdf_blocks_descs, 1);
      const dim3 threads_per_block(container_->voxel_block_volume_, 1);

      if (num_sdf_blocks_descs > 0) {
        if (!compact_host_voxels_) {
          prepareChunkToGlobalHashPass2Kernel<<<n_blocks, 1>>>(
            d_SDFBlockDescs, d_voxel_offsets_, container_->d_instance_);
          thrust::device_ptr<uint> offsets(d_voxel_offsets_);
          thrust::exclusive_scan(thrust::device, offsets, offsets + num_sdf_blocks_descs, offsets);
        }
        chunkToGlobalHashPass2Kernel<<<n_blocks, threads_per_block>>>(
          heap_count_prev,
          d_SDFBlockDescs,
          d_voxel_offsets_,
          compact_host_voxels_,
          d_SDFBlocks,
          d_heap_ptr,
          merge_blocks,
          container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunkToGlobalHashPass2Compact(
      const uint num_sdf_blocks_descs,
      const SDFBlockDesc* block_descs,
      const CompactVoxel* voxels,
      uint* block_pointers,
      const uchar* merge_blocks) {
      const dim3 blocks(num_sdf_blocks_descs, 1);
      const dim3 threads(container_->voxel_block_volume_, 1);
      if (num_sdf_blocks_descs > 0) {
        chunkToGlobalHashPass2CompactKernel<<<blocks, threads>>>(
          block_descs, voxels, block_pointers, merge_blocks, container_->d_instance_);
        CUDA_CHECK(cudaDeviceSynchronize());
      }
    }

  } // namespace cugeoutils
} // namespace cupanutils
