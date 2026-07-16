#include "streamer.cuh"
#include "utils/point_cloud_serializer.h"
#include <unordered_set>

namespace cupanutils {
  namespace cugeoutils {

    template <typename T>
    void
    Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::create(const Eigen::Vector3f& voxel_extents,
                                                                      const uint max_num_sdf_block_integrate_from_global_hash,
                                                                      unsigned int initial_chunk_list_size,
                                                                      bool compact_host_voxels) {
      container_->voxel_extents_ = Eig2CUDA(voxel_extents);
      voxel_extents_             = voxel_extents;
      const float max_chunk_ext_ = std::max(std::max(voxel_extents_.x(), voxel_extents_.y()), voxel_extents_.z());
      chunk_radius_              = 0.5f * max_chunk_ext_ * sqrt(3.f);
      initial_chunk_list_size_   = initial_chunk_list_size;
      compact_host_voxels_       = compact_host_voxels;
      max_num_sdf_block_integrate_from_global_hash_ = max_num_sdf_block_integrate_from_global_hash;
      stream_in_done_                               = true;

      // use cuda host allocation, allocate page-lock memory on host,
      // this is required for asynchronous stream, parallel copy GPU-CPU

      calculateMemoryUsage();

      // desc hash entry, other voxel
      CUDA_CHECK(cudaMallocHost(&h_SDFBlockDescOutput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(cudaMalloc(&d_SDFBlockDescOutput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(cudaMalloc(&d_SDFBlockDescInput_, sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_));
      const size_t voxel_count = total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;
      if (compact_host_voxels_) {
        CUDA_CHECK(cudaMallocHost(&h_compact_voxel_output_, sizeof(CompactVoxel) * voxel_count));
        CUDA_CHECK(cudaMalloc(&d_compact_voxel_output_, sizeof(CompactVoxel) * voxel_count));
        CUDA_CHECK(cudaMalloc(&d_compact_voxel_input_, sizeof(CompactVoxel) * voxel_count));
      } else {
        CUDA_CHECK(cudaMallocHost(&h_SDFBlockOutput_, sizeof(T) * voxel_count));
        CUDA_CHECK(cudaMalloc(&d_SDFBlockOutput_, sizeof(T) * voxel_count));
        CUDA_CHECK(cudaMalloc(&d_SDFBlockInput_, sizeof(T) * voxel_count));
      }

      CUDA_CHECK(cudaMalloc(&d_SDF_block_counter_, sizeof(uint)));
      CUDA_CHECK(cudaMalloc(&d_voxel_offsets_, sizeof(uint) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(cudaMalloc(&d_blocks_ptr_, sizeof(uint) * max_num_sdf_block_integrate_from_global_hash_));
      CUDA_CHECK(cudaMalloc(&d_merge_blocks_, sizeof(uchar) * max_num_sdf_block_integrate_from_global_hash_));

      container_->updateFieldsDevice(); // update this gets called after constructors

      CUDA_CHECK(cudaEventCreate(&start_event_));
      CUDA_CHECK(cudaEventCreate(&stop_event_));
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::clearGrid() {
      for (auto& [chunk_pos, chunk_ptr] : grid_) {
        chunk_ptr->clear();
        chunk_ptr.reset();
      }
      grid_.clear();
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::rechunk(const float scale) {
      auto source_grid = std::move(grid_);
      voxel_extents_ *= scale;
      chunk_radius_ = 0.5f * voxel_extents_.maxCoeff() * sqrt(3.f);
      for (auto& [source_chunk, source] : source_grid) {
        const uint block_count = source->getNElements();
        for (uint block_index = 0; block_index < block_count; ++block_index) {
          const SDFBlockDesc& desc = source->getSDFBlockDesc(block_index);
          const Eigen::Vector3f world = CUDA2Eig(desc.pos).cast<float>() * sdf_block_size * container_->virtual_voxel_size_;
          const Eigen::Vector3i chunk = worldToChunks(world);
          grid_.try_emplace(
            chunk, std::make_unique<ChunkDesc<T>>(initial_chunk_list_size_, compact_host_voxels_));
          source->moveSDFBlock(block_index, *grid_.at(chunk));
        }
        source->clear();
      }
    }
    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::destroy() {
      // Cleanup CUDA events
      if (start_event_) {
        CUDA_CHECK(cudaEventDestroy(start_event_));
        start_event_ = cudaEvent_t{};
      }
      if (stop_event_) {
        CUDA_CHECK(cudaEventDestroy(stop_event_));
        stop_event_ = cudaEvent_t{};
      }

      clearGrid();

      // Cleanup host memory
      if (h_SDFBlockDescOutput_) {
        CUDA_CHECK(cudaFreeHost(h_SDFBlockDescOutput_));
        h_SDFBlockDescOutput_ = nullptr;
      }
      if (h_SDFBlockOutput_) {
        CUDA_CHECK(cudaFreeHost(h_SDFBlockOutput_));
        h_SDFBlockOutput_ = nullptr;
      }
      if (h_compact_voxel_output_) {
        CUDA_CHECK(cudaFreeHost(h_compact_voxel_output_));
        h_compact_voxel_output_ = nullptr;
      }

      // Cleanup device memory
      if (d_SDFBlockDescOutput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockDescOutput_));
        d_SDFBlockDescOutput_ = nullptr;
      }
      if (d_SDFBlockOutput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockOutput_));
        d_SDFBlockOutput_ = nullptr;
      }
      if (d_SDFBlockDescInput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockDescInput_));
        d_SDFBlockDescInput_ = nullptr;
      }
      if (d_SDFBlockInput_) {
        CUDA_CHECK(cudaFree(d_SDFBlockInput_));
        d_SDFBlockInput_ = nullptr;
      }
      if (d_compact_voxel_output_) {
        CUDA_CHECK(cudaFree(d_compact_voxel_output_));
        d_compact_voxel_output_ = nullptr;
      }
      if (d_compact_voxel_input_) {
        CUDA_CHECK(cudaFree(d_compact_voxel_input_));
        d_compact_voxel_input_ = nullptr;
      }
      if (d_SDF_block_counter_) {
        CUDA_CHECK(cudaFree(d_SDF_block_counter_));
        d_SDF_block_counter_ = nullptr;
      }
      if (d_voxel_offsets_) {
        CUDA_CHECK(cudaFree(d_voxel_offsets_));
        d_voxel_offsets_ = nullptr;
      }
      if (d_blocks_ptr_) {
        CUDA_CHECK(cudaFree(d_blocks_ptr_));
        d_blocks_ptr_ = nullptr;
      }
      if (d_merge_blocks_) {
        CUDA_CHECK(cudaFree(d_merge_blocks_));
        d_merge_blocks_ = nullptr;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::serializeData(const std::string& filename_hash,
                                                                                  const std::string& filename_voxel) const {
      uint hash_count  = 0;
      uint voxel_count = 0;
      for (const auto& [chunk_pos, chunk_ptr] : grid_) {
        const data::vector<SDFBlockDesc>& descs = chunk_ptr->getSDFBlockDescs();
        for (uint k = 0; k < descs.size(); ++k) {
          uint valid_voxels        = 0;
          const int scale          = 1 << (finest_block_log2_dim - descs[k].resolution);
          const int num_voxels     = scale * scale * scale;
          for (uint l = 0; l < num_voxels; ++l) {
            if (chunk_ptr->getVoxel(k, l).weight > 0) {
              valid_voxels++;
            }
          }
          voxel_count += valid_voxels;
          hash_count += valid_voxels > 0 ? 1 : 0;
        }
      }

      std::ofstream hash_file(filename_hash, std::ios::binary);
      std::ofstream voxel_file(filename_voxel, std::ios::binary);
      hash_file << "ply\nformat binary_little_endian 1.0\nelement vertex " << hash_count
                << "\nproperty float x\nproperty float y\nproperty float z\nproperty float weight\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n";
      voxel_file << "ply\nformat binary_little_endian 1.0\nelement vertex " << voxel_count
                 << "\nproperty float x\nproperty float y\nproperty float z\nproperty float sdf\nproperty float weight\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nend_header\n";

      for (const auto& [chunk_pos, chunk_ptr] : grid_) {
        const data::vector<SDFBlockDesc>& descs = chunk_ptr->getSDFBlockDescs();
        for (uint k = 0; k < descs.size(); ++k) {
          const Eigen::Vector3i pos      = CUDA2Eig(descs[k].pos);
          const Eigen::Vector3f block_pw = (pos * sdf_block_size).cast<float>() * container_->virtual_voxel_size_;
          Eigen::Vector4f block_color    = Eigen::Vector4f::Zero();
          uint valid_voxels              = 0;
          float block_weight_sum         = 0.f;
          const int scale                = 1 << (finest_block_log2_dim - descs[k].resolution);
          const int num_voxels           = scale * scale * scale;
          const int scaling_factor       = 1 << descs[k].resolution;
          for (uint l = 0; l < num_voxels; ++l) {
            const T voxel = chunk_ptr->getVoxel(k, l);
            if (voxel.weight > 0) {
              const Eigen::Vector3i dl =
                CUDA2Eig(SDFBlock<T>::delinearizeVoxelIndex(l, sdf_block_size / scaling_factor)) * scaling_factor;
              const Eigen::Vector3f voxel_pw = block_pw + dl.cast<float>() * container_->virtual_voxel_size_;
              Eigen::Vector4f voxel_color;
              if (descs[k].resolution == 0) {
                voxel_color = Eigen::Vector4f(1.f, 0.f, 0.f, 1.f);
              } else {
                voxel_color = Eigen::Vector4f(0.f, 1.f, 0.f, 1.f);
              }
              const float voxel_weight = static_cast<float>(voxel.weight);
              const float voxel_sdf    = voxel.sdf;
              unsigned char color[4]   = {(unsigned char) (voxel_color(0) * 255),
                                           (unsigned char) (voxel_color(1) * 255),
                                           (unsigned char) (voxel_color(2) * 255),
                                           (unsigned char) (voxel_color(3) * 255)};
              voxel_file.write((const char*) voxel_pw.data(), sizeof(float) * 3);
              voxel_file.write((const char*) &voxel_sdf, sizeof(float));
              voxel_file.write((const char*) &voxel_weight, sizeof(float));
              voxel_file.write((const char*) color, sizeof(unsigned char) * 4);
              block_color += voxel_color;
              block_weight_sum += voxel_weight;
              valid_voxels++;
            }
          }
          if (valid_voxels > 0) {
            const Eigen::Vector4f avg_color = block_color / valid_voxels;
            const float avg_weight          = block_weight_sum / static_cast<float>(valid_voxels);
            unsigned char color[4]          = {(unsigned char) (avg_color(0) * 255),
                                                (unsigned char) (avg_color(1) * 255),
                                                (unsigned char) (avg_color(2) * 255),
                                                255};
            hash_file.write((const char*) block_pw.data(), sizeof(float) * 3);
            hash_file.write((const char*) &avg_weight, sizeof(float));
            hash_file.write((const char*) color, sizeof(unsigned char) * 4);
          }
        }
      }
      std::cout << "Streamer::serializeData | written " << hash_count << " hash points and " << voxel_count
                << " voxels to " << filename_hash << " and " << filename_voxel << std::endl;
    }

    /**
     *
     * STREAM-OUT TO HOST
     *
     **/

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamOutToHostPass0(const float3& camera_position,
                                                                                         const float radius) {
      const uint num_pass = (container_->total_size_ + max_num_sdf_block_integrate_from_global_hash_ - 1) /
                            max_num_sdf_block_integrate_from_global_hash_;
      uint streamed_out_blocks = 0;
      for (int pass = 0; pass < num_pass; ++pass) {
        container_->resetHashBucketMutex();
        clearSDFBlockCounter();
        integrateFromGlobalHashPass1(radius, camera_position, pass);
        curr_stream_out_blocks_ = getSDFBlockCounter();

        if (curr_stream_out_blocks_ > 0) {
          integrateFromGlobalHashPass2(curr_stream_out_blocks_, pass);

          const int stream_size = curr_stream_out_blocks_;
          CUDA_CHECK(cudaMemcpy(
            &h_SDFBlockDescOutput_[0], &d_SDFBlockDescOutput_[0], sizeof(SDFBlockDesc) * stream_size, cudaMemcpyDeviceToHost));
          if (compact_host_voxels_) {
            CUDA_CHECK(cudaMemcpy(h_compact_voxel_output_,
                                  d_compact_voxel_output_,
                                  sizeof(CompactVoxel) * container_->voxel_block_volume_ * stream_size,
                                  cudaMemcpyDeviceToHost));
            integrateCompactInChunkGrid(h_SDFBlockDescOutput_, h_compact_voxel_output_);
          } else {
            CUDA_CHECK(cudaMemcpy(h_SDFBlockOutput_,
                                  d_SDFBlockOutput_,
                                  sizeof(T) * container_->voxel_block_volume_ * stream_size,
                                  cudaMemcpyDeviceToHost));
            integrateInChunkGrid(h_SDFBlockDescOutput_, h_SDFBlockOutput_);
          }
          streamed_out_blocks += curr_stream_out_blocks_;
        }
      }
      curr_stream_out_blocks_ = streamed_out_blocks;
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamOutToCPUPass1CPU() {
      if (curr_stream_out_blocks_ > 0) {
        if (compact_host_voxels_)
          integrateCompactInChunkGrid(h_SDFBlockDescOutput_, h_compact_voxel_output_);
        else
          integrateInChunkGrid(h_SDFBlockDescOutput_, h_SDFBlockOutput_);
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateInChunkGrid(const SDFBlockDesc* h_SDFBlockDescOutput,
                                                                                         const T* h_SDFBlockOutput) {
      int start_idx = 0;
      for (uint i = 0; i < curr_stream_out_blocks_; ++i) {
        const SDFBlockDesc& desc = h_SDFBlockDescOutput[i];
        const int scale          = 1 << (finest_block_log2_dim - desc.resolution);
        const int num_voxels     = scale * scale * scale;
        SDFBlock<T> block(num_voxels);

        for (int j = 0; j < num_voxels; j++) {
          const auto& voxel = h_SDFBlockOutput[start_idx + j];
          block.data.push_back(voxel);
        }

        Eigen::Vector3i pos       = CUDA2Eig(h_SDFBlockDescOutput[i].pos);
        Eigen::Vector3f pw        = pos.cast<float>() * sdf_block_size * container_->virtual_voxel_size_;
        Eigen::Vector3i chunk_pos = worldToChunks(pw);

        const float r               = 0.f;
        const float g               = float(i) / float(curr_stream_out_blocks_);
        const float b               = 1.f - float(i) / float(curr_stream_out_blocks_);
        const Eigen::Vector4f color = Eigen::Vector4f(r, g, b, 1);

        grid_.try_emplace(
          chunk_pos, std::make_unique<ChunkDesc<T>>(initial_chunk_list_size_, compact_host_voxels_));

        // add element to host list
        // if this element is in frustum cannot be accessed by the gpu anyway
        // it lives in host from now
        grid_.at(chunk_pos)->addSDFBlock(desc, block);
        start_idx += num_voxels;
      }
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateCompactInChunkGrid(
      const SDFBlockDesc* block_descs, const CompactVoxel* voxels) {
      for (uint block_index = 0; block_index < curr_stream_out_blocks_; ++block_index) {
        const SDFBlockDesc& desc = block_descs[block_index];
        const Eigen::Vector3f world =
          CUDA2Eig(desc.pos).cast<float>() * sdf_block_size * container_->virtual_voxel_size_;
        const Eigen::Vector3i chunk = worldToChunks(world);
        grid_.try_emplace(
          chunk, std::make_unique<ChunkDesc<T>>(initial_chunk_list_size_, compact_host_voxels_));
        grid_.at(chunk)->addCompactSDFBlock(desc, voxels + block_index * total_sdf_block_size);
      }
    }

    template <typename T>
    std::vector<Eigen::Vector3i> Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::chunks() {
      std::unordered_set<Eigen::Vector3i, Vector3iHash> unique_chunks;
      for (const auto& [chunk, contents] : grid_) {
        if (contents->getNElements() > 0)
          unique_chunks.insert(chunk);
      }
      container_->flatAndReduceHashTable();
      std::vector<HashEntry> active_blocks(container_->current_occupied_blocks_);
      CUDA_CHECK(cudaMemcpy(active_blocks.data(),
                            container_->d_compactHashTable_,
                            sizeof(HashEntry) * active_blocks.size(),
                            cudaMemcpyDeviceToHost));
      for (const HashEntry& entry : active_blocks) {
        const Eigen::Vector3f world =
          CUDA2Eig(entry.pos).cast<float>() * sdf_block_size * container_->virtual_voxel_size_;
        unique_chunks.insert(worldToChunks(world));
      }
      return std::vector<Eigen::Vector3i>(unique_chunks.begin(), unique_chunks.end());
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamAllOut() {
      const uint num_pass = (container_->total_size_ + max_num_sdf_block_integrate_from_global_hash_ - 1) /
                            max_num_sdf_block_integrate_from_global_hash_;
      uint streamed_out_blocks = 0;
      for (int pass = 0; pass < num_pass; ++pass) {
        container_->resetHashBucketMutex();
        clearSDFBlockCounter();
        CUDA_CHECK(cudaEventRecord(start_event_, 0));
        integrateFromGlobalHashPass1(pass);
        curr_stream_out_blocks_ = getSDFBlockCounter();

        if (curr_stream_out_blocks_ > 0) {
          integrateFromGlobalHashPass2(curr_stream_out_blocks_, pass);

          const int stream_size = curr_stream_out_blocks_;
          CUDA_CHECK(cudaMemcpy(
            &h_SDFBlockDescOutput_[0], &d_SDFBlockDescOutput_[0], sizeof(SDFBlockDesc) * stream_size, cudaMemcpyDeviceToHost));
          if (compact_host_voxels_) {
            CUDA_CHECK(cudaMemcpy(h_compact_voxel_output_,
                                  d_compact_voxel_output_,
                                  sizeof(CompactVoxel) * container_->voxel_block_volume_ * stream_size,
                                  cudaMemcpyDeviceToHost));
          } else {
            CUDA_CHECK(cudaMemcpy(h_SDFBlockOutput_,
                                  d_SDFBlockOutput_,
                                  sizeof(T) * container_->voxel_block_volume_ * stream_size,
                                  cudaMemcpyDeviceToHost));
          }
          CUDA_CHECK(cudaEventRecord(stop_event_, 0));
          CUDA_CHECK(cudaEventSynchronize(stop_event_));
          CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));
          if (compact_host_voxels_)
            integrateCompactInChunkGrid(h_SDFBlockDescOutput_, h_compact_voxel_output_);
          else
            integrateInChunkGrid(h_SDFBlockDescOutput_, h_SDFBlockOutput_);
          streamed_out_blocks += curr_stream_out_blocks_;
        }
      }
    }

    /**
     *
     * stream-IN TO DEVICE
     *
     * */

    template <typename T>
    uint Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateInHash(const Eigen::Vector3f& camera_pose,
                                                                                    float radius) {
      Eigen::Vector3i camera_chunk = worldToChunks(camera_pose);
      Eigen::Vector3i chunk_radius = meterToNumberOfChunksCeil(radius);
      Eigen::Vector3i start_chunk  = camera_chunk - chunk_radius;
      Eigen::Vector3i end_chunk    = camera_chunk + chunk_radius;

      uint num_SDF_blocks = 0;
      uint copied_voxels  = 0;
      stream_in_done_     = true;

      for (auto& [chunk_pos, chunk_ptr] : grid_) {
        if (!isChunkInSphere(chunk_pos, camera_pose, radius))
          continue;

        const uint num_blocks = chunk_ptr->getNElements();
        const uint available = max_num_sdf_block_integrate_from_global_hash_ - num_SDF_blocks;
        const uint copy_count = std::min(num_blocks, available);
        for (uint i = 0; i < copy_count; ++i) {
          const uint chunk_index = num_blocks - copy_count + i;
          const SDFBlockDesc& desc = chunk_ptr->getSDFBlockDesc(chunk_index);
          const int resolution     = desc.resolution;
          const int scale          = 1 << (finest_block_log2_dim - resolution);
          const int num_voxels     = scale * scale * scale;

          h_SDFBlockDescOutput_[num_SDF_blocks + i] = desc;
          if (compact_host_voxels_) {
            chunk_ptr->copyCompactSDFBlock(
              chunk_index, h_compact_voxel_output_ + (num_SDF_blocks + i) * total_sdf_block_size);
          } else {
            chunk_ptr->copySDFBlock(chunk_index, h_SDFBlockOutput_ + copied_voxels);
          }
          copied_voxels += num_voxels;
        }
        chunk_ptr->removeLastSDFBlocks(copy_count);
        num_SDF_blocks += copy_count;
        if (num_SDF_blocks == max_num_sdf_block_integrate_from_global_hash_) {
          stream_in_done_ = false;
          break;
        }
      }

      CUDA_CHECK(cudaMemcpy(d_SDFBlockDescInput_,
                            h_SDFBlockDescOutput_,
                            sizeof(SDFBlockDesc) * num_SDF_blocks,
                            cudaMemcpyHostToDevice));
      if (compact_host_voxels_) {
        CUDA_CHECK(cudaMemcpy(d_compact_voxel_input_,
                              h_compact_voxel_output_,
                              sizeof(CompactVoxel) * total_sdf_block_size * num_SDF_blocks,
                              cudaMemcpyHostToDevice));
      } else {
        CUDA_CHECK(cudaMemcpy(
          d_SDFBlockInput_, h_SDFBlockOutput_, sizeof(T) * copied_voxels, cudaMemcpyHostToDevice));
      }

      return num_SDF_blocks;
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::stream(const Eigen::Vector3f& camera_position,
                                                                           const float radius) {
      {
        CUDAProfiler::CUDAEvent event(streaming_profiler_);
        CUDA_CHECK(cudaEventRecord(start_event_, 0));
        // stream - out in RAM
        streamOutToHostPass0(Eig2CUDA(camera_position), radius);
        // blocks host until event terminate, here we are
        // copy host middle layer structure to main grid
        CUDA_CHECK(cudaEventRecord(stop_event_, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_event_));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start_event_, stop_event_));

        // stream - in in GPU
        streamInToGPU(camera_position, radius);
      }
      streaming_profiler_.write(curr_stream_in_blocks_ + curr_stream_out_blocks_);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamInToGPU(const Eigen::Vector3f& camera_position,
                                                                                  const float radius) {
      do {
        const uint n_sdf_block_descs = integrateInHash(camera_position, radius);
        curr_stream_in_blocks_       = n_sdf_block_descs;
        if (curr_stream_in_blocks_ > 0) {
          // ! alloc memory for chunks
          uint heap_count_prev; // ptr to the first free block

          CUDA_CHECK(cudaMemcpy(&heap_count_prev, container_->d_heapCounterHigh_, sizeof(uint), cudaMemcpyDeviceToHost));

          chunkToGlobalHashPass1(
            curr_stream_in_blocks_, heap_count_prev, d_SDFBlockDescInput_, d_blocks_ptr_, d_merge_blocks_);
          if (compact_host_voxels_) {
            chunkToGlobalHashPass2Compact(
              curr_stream_in_blocks_, d_SDFBlockDescInput_, d_compact_voxel_input_, d_blocks_ptr_, d_merge_blocks_);
          } else {
            chunkToGlobalHashPass2(
              curr_stream_in_blocks_, heap_count_prev, d_SDFBlockDescInput_, d_SDFBlockInput_, d_blocks_ptr_, d_merge_blocks_);
          }
        }
      } while (!stream_in_done_);
    }

    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::streamInToGPUChunkNeighborhood(const Eigen::Vector3i& chunk,
                                                                                                   const int radius) {
      const Eigen::Vector3i start_chunk = chunk - Eigen::Vector3i(radius);
      const Eigen::Vector3i end_chunk   = chunk + Eigen::Vector3i(radius);

      for (int x = start_chunk.x(); x < end_chunk.x(); ++x) {
        for (int y = start_chunk.y(); y < end_chunk.y(); ++y) {
          for (int z = start_chunk.z(); z < end_chunk.z(); ++z) {
            const Eigen::Vector3i curr_chunk = Eigen::Vector3i(x, y, z);
            streamInToGPU(chunkToWorld(chunk), 1.1f * getChunkRadiusInMeter());
          }
        }
      }
    }

    /////////////////////////////////////////////////////////////
    ////////////////////////// DEBUG ////////////////////////////
    /////////////////////////////////////////////////////////////

    template <typename T>
    double Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::debugCheckForDuplicates() const {
      uint duplicates_count = 0;
      uint total_grid_count = 0;
      std::unordered_set<SDFBlockDesc, SDFBlockDesc::HashSDFBlockDesc> desc_hash;
      std::cerr << "debugCheckForDuplicates | HashTable\n";
      HashEntry* h_hashTable = new HashEntry[container_->total_size_];
      CUDA_CHECK(
        cudaMemcpy(h_hashTable, container_->d_hashTable_, sizeof(HashEntry) * container_->total_size_, cudaMemcpyDeviceToHost));
      for (uint i = 0; i < container_->total_size_; ++i) {
        if (h_hashTable[i].ptr != FREE_ENTRY) {
          total_grid_count++;
          SDFBlockDesc curr(h_hashTable[i]);
          if (desc_hash.find(curr) == desc_hash.end())
            desc_hash.insert(curr);
          else {
            duplicates_count++;
            // std::cerr << "debugCheckForDuplicates | ptr: " << curr.ptr << " pos: " << curr.pos.x << " " << curr.pos.y << " "
            // << curr.pos.z << std::endl; throw std::runtime_error("debugCheckForDuplicates | duplicate found in streaming hash
            // data (in hash)");
          }
        }
      }

      delete[] h_hashTable;

      std::cerr << "debugCheckForDuplicates | Grid\n";
      for (const auto& [chunk_pos, chunk_ptr] : grid_) {
        const data::vector<SDFBlockDesc>& descs_copy = chunk_ptr->getSDFBlockDescs();
        for (unsigned int k = 0; k < descs_copy.size(); ++k) {
          total_grid_count++;
          if (desc_hash.find(descs_copy[k]) == desc_hash.end())
            desc_hash.insert(descs_copy[k]);
          else {
            duplicates_count++;
            // std::cerr << "debugCheckForDuplicates | ptr: " << descs_copy[k].ptr << " pos: " << descs_copy[k].pos.x << " " <<
            // descs_copy[k].pos.y << " " << descs_copy[k].pos.z << std::endl; throw std::runtime_error("debugCheckForDuplicates
            // | duplicate found in streaming hash data (in grid)");
          }
        }
      }

      // stream
      const double duplicates_ratio = (double) duplicates_count / (double) total_grid_count * 100;
      std::cerr << "debugCheckForDuplicates | duplicates ratio: " << duplicates_ratio << std::endl;
      return duplicates_ratio;
    }

    /// @brief DEBUG calculate and output total memory usage by the Streamer class
    template <typename T>
    void Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>>::calculateMemoryUsage() const {
      double toMB = 1e-6;
      if (memory_allocation_filepath_.empty())
        return;

      // host buffers
      const uint64_t size_h_SDFBlockDescOutput = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_h_SDFBlockOutput     = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;

      // get grid size
      const uint64_t size_h_grid = sizeof(std::vector<ChunkDesc<T>*>) * grid_.size() *
                                   (sizeof(std::vector<SDFBlock<T>>) + initial_chunk_list_size_ * sizeof(SDFBlock<T>) +
                                    sizeof(std::vector<SDFBlockDesc>) + initial_chunk_list_size_ * sizeof(SDFBlockDesc));

      std::ofstream out_file(memory_allocation_filepath_);
      if (!out_file.is_open()) {
        std::cerr << "Streamer::calculateMemoryUsage | Failed to open file for writing voxel memory usage." << std::endl;
        return;
      }
      out_file << "Streamer | size_h_SDFBlockDescOutput : " << size_h_SDFBlockDescOutput << std::endl;
      out_file << "Streamer | size_h_SDFBlockOutput : " << size_h_SDFBlockOutput << std::endl;
      out_file << "Streamer | size_h_grid : " << size_h_grid << std::endl;
      const uint64_t tot_size_host = size_h_SDFBlockDescOutput + size_h_SDFBlockOutput + size_h_grid;
      out_file << "Streamer | total h_size: " << tot_size_host << " B || " << (double) tot_size_host * toMB << " MiB"
               << std::endl;
      out_file << "=========================================================" << std::endl;

      // device buffers
      const uint64_t size_d_SDFBlockDescOutput = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockOutput     = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockDescInput  = sizeof(SDFBlockDesc) * max_num_sdf_block_integrate_from_global_hash_;
      const uint64_t size_d_SDFBlockInput      = sizeof(T) * total_sdf_block_size * max_num_sdf_block_integrate_from_global_hash_;

      out_file << "Streamer | size_d_SDFBlockDescOutput : " << size_d_SDFBlockDescOutput << std::endl;
      out_file << "Streamer | size_d_SDFBlockOutput : " << size_d_SDFBlockOutput << std::endl;
      out_file << "Streamer | size_d_SDFBlockDescInput : " << size_d_SDFBlockDescInput << std::endl;
      out_file << "Streamer | size_d_SDFBlockInput : " << size_d_SDFBlockInput << std::endl;

      const uint64_t tot_size_device =
        size_d_SDFBlockDescOutput + size_d_SDFBlockOutput + size_d_SDFBlockDescInput + size_d_SDFBlockInput;
      out_file << "Streamer | total d_size: " << tot_size_device << " B || " << (double) tot_size_device * toMB << " MiB"
               << std::endl;
      out_file << "=========================================================" << std::endl;
    }

  } // namespace cugeoutils
} // namespace cupanutils
