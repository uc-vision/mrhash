#pragma once

#include "utils/cista.h"
#include "voxel_data_structures.cuh"
#include <cuda_fp16.h>
#include <unordered_map>
#include <vector>

namespace data = cista::offset;

namespace cupanutils {
  namespace cugeoutils {

    struct Vector3iHash {
      std::size_t operator()(const Eigen::Vector3i& v) const noexcept {
        size_t h1 = std::hash<int>()(v.x());
        size_t h2 = std::hash<int>()(v.y());
        size_t h3 = std::hash<int>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2); // Combine hashes
      }
    };

    template <typename T>
    struct SDFBlock {
      data::vector<T> data;

      SDFBlock(const int num_voxels) {
        data.reserve(num_voxels);
      }

      static uint3 delinearizeVoxelIndex(uint idx, const int block_size = sdf_block_size) {
        uint x = idx % block_size;
        uint y = (idx % (block_size * block_size)) / block_size;
        uint z = idx / (block_size * block_size);
        return make_uint3(x, y, z);
      }

      auto cista_members() {
        return std::tie(data);
      }
    };

    struct CompactVoxel {
      uchar sdf_low;
      uchar sdf_high;
      uchar weight;
      uchar secondary_low;
      uchar secondary_high;
      uchar secondary_weight;

      auto cista_members() {
        return std::tie(sdf_low, sdf_high, weight, secondary_low, secondary_high, secondary_weight);
      }
    };
    static_assert(sizeof(CompactVoxel) == 6);

    class SDFBlockDesc {
    public:
      __host__ __device__ SDFBlockDesc() :
        pos(make_int3(0, 0, 0)),
        ptr(-1),
        resolution(0),
        direction(static_cast<signed char>(TSDFDirection::none)) {
      }

      __host__ __device__ SDFBlockDesc(const HashEntry& entry) {
        pos        = entry.pos;
        ptr        = entry.ptr;
        resolution = entry.resolution;
        direction  = entry.direction;
      }

      bool operator<(const SDFBlockDesc& other) const {
        if (pos.x == other.pos.x) {
          if (pos.y == other.pos.y) {
            if (pos.z == other.pos.z)
              return direction < other.direction;
            return pos.z < other.pos.z;
          }
          return pos.y < other.pos.y;
        }
        return pos.x < other.pos.x;
      }

      bool operator==(const SDFBlockDesc& other) const {
        return pos.x == other.pos.x && pos.y == other.pos.y && pos.z == other.pos.z && direction == other.direction;
      }

      struct HashSDFBlockDesc {
        size_t operator()(const SDFBlockDesc& other) const {
          const int3& v    = other.pos;
          const size_t res = ((size_t) v.x * p0) ^ ((size_t) v.y * p1) ^ ((size_t) v.z * p2) ^
                             (static_cast<size_t>(other.direction) * 83492791u);
          return res;
        }
      };

      auto cista_members() {
        return std::tie(pos, ptr, resolution, direction);
      }

      int3 pos;
      int ptr;
      int resolution;
      signed char direction;
    } __align__(16);

    template <typename T>
    class ChunkDesc {
    public:
      ChunkDesc(const uint& initial_chunk_list_size, const bool compact_voxels) : compact_voxels_(compact_voxels) {
        vecSDFBlock_ = data::vector<SDFBlock<T>>();
        vecSDFBlock_.reserve(initial_chunk_list_size);
        vecCompactVoxels_ = data::vector<CompactVoxel>();
        vecCompactVoxels_.reserve(initial_chunk_list_size * total_sdf_block_size);
        vecChunkDesc_ = data::vector<SDFBlockDesc>();
        vecChunkDesc_.reserve(initial_chunk_list_size);
      }

      ChunkDesc(const ChunkDesc& chunk_desc) {
        vecSDFBlock_  = chunk_desc.vecSDFBlock_;
        vecCompactVoxels_ = chunk_desc.vecCompactVoxels_;
        vecChunkDesc_ = chunk_desc.vecChunkDesc_;
        block_indices_ = chunk_desc.block_indices_;
        compact_voxels_ = chunk_desc.compact_voxels_;
      }

      // Copy assignment operator
      ChunkDesc& operator=(const ChunkDesc& chunk_desc) {
        if (this != &chunk_desc) {
          vecSDFBlock_  = chunk_desc.vecSDFBlock_;
          vecCompactVoxels_ = chunk_desc.vecCompactVoxels_;
          vecChunkDesc_ = chunk_desc.vecChunkDesc_;
          block_indices_ = chunk_desc.block_indices_;
          compact_voxels_ = chunk_desc.compact_voxels_;
        }
        return *this;
      }

      // Move constructor
      ChunkDesc(ChunkDesc&& chunk_desc) noexcept {
        vecSDFBlock_  = std::move(chunk_desc.vecSDFBlock_);
        vecCompactVoxels_ = std::move(chunk_desc.vecCompactVoxels_);
        vecChunkDesc_ = std::move(chunk_desc.vecChunkDesc_);
        block_indices_ = std::move(chunk_desc.block_indices_);
        compact_voxels_ = chunk_desc.compact_voxels_;
      }

      // Move assignment operator
      ChunkDesc& operator=(ChunkDesc&& chunk_desc) noexcept {
        if (this != &chunk_desc) {
          vecSDFBlock_  = std::move(chunk_desc.vecSDFBlock_);
          vecCompactVoxels_ = std::move(chunk_desc.vecCompactVoxels_);
          vecChunkDesc_ = std::move(chunk_desc.vecChunkDesc_);
          block_indices_ = std::move(chunk_desc.block_indices_);
          compact_voxels_ = chunk_desc.compact_voxels_;
        }
        return *this;
      }

      // Destructor (default is fine, but explicit for completeness)
      ~ChunkDesc() = default;

      void addSDFBlock(const SDFBlockDesc& desc, const SDFBlock<T>& data) {
        if (!compact_voxels_ && desc.direction != static_cast<signed char>(TSDFDirection::none)) {
          const auto [existing, inserted] = block_indices_.try_emplace(desc, vecChunkDesc_.size());
          if (!inserted) {
            SDFBlock<T>& target = vecSDFBlock_[existing->second];
            for (uint voxel_index = 0; voxel_index < data.data.size(); ++voxel_index) {
              const T& input = data.data[voxel_index];
              if (input.sum_squared == 0.f)
                continue;
              T& output = target.data[voxel_index];
              if (output.sum_squared == 0.f)
                output = input;
              else {
                output.sdf += input.sdf;
                output.sum_squared += input.sum_squared;
                output.weight = 1;
              }
            }
            return;
          }
        }
        vecChunkDesc_.push_back(desc);
        if (compact_voxels_) {
          const size_t offset = vecCompactVoxels_.size();
          vecCompactVoxels_.resize(offset + total_sdf_block_size);
          CompactVoxel* output = vecCompactVoxels_.data() + offset;
          for (const T& voxel : data.data) {
            const ushort sdf = __half_as_ushort(__float2half(voxel.sdf));
            const ushort secondary = __half_as_ushort(__float2half(voxel.sum_squared));
            *output++ = CompactVoxel{static_cast<uchar>(sdf),
                                     static_cast<uchar>(sdf >> 8),
                                     voxel.weight,
                                     static_cast<uchar>(secondary),
                                     static_cast<uchar>(secondary >> 8),
                                     voxel.rgb.x};
          }
        } else {
          vecSDFBlock_.push_back(data);
        }
      }

      void addCompactSDFBlock(const SDFBlockDesc& desc, const CompactVoxel* data) {
        const auto [existing, inserted] = block_indices_.try_emplace(desc, vecChunkDesc_.size());
        if (!inserted) {
          CompactVoxel* target = vecCompactVoxels_.data() + existing->second * total_sdf_block_size;
          for (uint voxel_index = 0; voxel_index < total_sdf_block_size; ++voxel_index) {
            const CompactVoxel& input = data[voxel_index];
            CompactVoxel& output = target[voxel_index];
            if (input.weight > 0) {
              if (output.weight == 0) {
                output.sdf_low = input.sdf_low;
                output.sdf_high = input.sdf_high;
                output.weight = input.weight;
              } else {
                const ushort input_sdf = input.sdf_low | (input.sdf_high << 8);
                const ushort output_sdf = output.sdf_low | (output.sdf_high << 8);
                const uint weight = output.weight + input.weight;
                const float sdf = (__half2float(__ushort_as_half(output_sdf)) * output.weight +
                                   __half2float(__ushort_as_half(input_sdf)) * input.weight) /
                                  weight;
                const ushort merged_sdf = __half_as_ushort(__float2half(sdf));
                output.sdf_low = static_cast<uchar>(merged_sdf);
                output.sdf_high = static_cast<uchar>(merged_sdf >> 8);
                output.weight = static_cast<uchar>(std::min(weight, integration_weight_max));
              }
            }
            if (input.secondary_weight == 0)
              continue;
            if (output.secondary_weight == 0) {
              output.secondary_low = input.secondary_low;
              output.secondary_high = input.secondary_high;
              output.secondary_weight = input.secondary_weight;
              continue;
            }
            const ushort input_secondary = input.secondary_low | (input.secondary_high << 8);
            const ushort output_secondary = output.secondary_low | (output.secondary_high << 8);
            const uint secondary_weight = output.secondary_weight + input.secondary_weight;
            const float secondary =
              (__half2float(__ushort_as_half(output_secondary)) * output.secondary_weight +
               __half2float(__ushort_as_half(input_secondary)) * input.secondary_weight) /
              secondary_weight;
            const ushort merged_secondary = __half_as_ushort(__float2half(secondary));
            output.secondary_low = static_cast<uchar>(merged_secondary);
            output.secondary_high = static_cast<uchar>(merged_secondary >> 8);
            output.secondary_weight =
              static_cast<uchar>(std::min(secondary_weight, integration_weight_max));
          }
          return;
        }
        vecChunkDesc_.push_back(desc);
        const size_t offset = vecCompactVoxels_.size();
        vecCompactVoxels_.resize(offset + total_sdf_block_size);
        std::copy_n(data, total_sdf_block_size, vecCompactVoxels_.data() + offset);
      }

      uint getNElements() const {
        return (uint) vecChunkDesc_.size();
      }

      const SDFBlockDesc& getSDFBlockDesc(uint i) const {
        return vecChunkDesc_[i];
      }

      int findSDFBlock(const SDFBlockDesc& description) const {
        const auto found = block_indices_.find(description);
        return found == block_indices_.end() ? -1 : static_cast<int>(found->second);
      }

      void copySDFBlock(const uint i, T* output) const {
        if (compact_voxels_) {
          const CompactVoxel* block = vecCompactVoxels_.data() + i * total_sdf_block_size;
          for (uint voxel_index = 0; voxel_index < total_sdf_block_size; ++voxel_index) {
            const ushort sdf = block[voxel_index].sdf_low | (block[voxel_index].sdf_high << 8);
            const ushort secondary =
              block[voxel_index].secondary_low | (block[voxel_index].secondary_high << 8);
            output[voxel_index] = T();
            output[voxel_index].sdf = __half2float(__ushort_as_half(sdf));
            output[voxel_index].weight = block[voxel_index].weight;
            output[voxel_index].sum_squared = __half2float(__ushort_as_half(secondary));
            output[voxel_index].rgb.x = block[voxel_index].secondary_weight;
          }
        } else {
          std::copy(vecSDFBlock_[i].data.begin(), vecSDFBlock_[i].data.end(), output);
        }
      }

      void copyCompactSDFBlock(const uint index, CompactVoxel* output) const {
        const CompactVoxel* input = vecCompactVoxels_.data() + index * total_sdf_block_size;
        std::copy_n(input, total_sdf_block_size, output);
      }

      T getVoxel(const uint block_index, const uint voxel_index) const {
        if (!compact_voxels_)
          return vecSDFBlock_[block_index].data[voxel_index];
        const CompactVoxel& compact = vecCompactVoxels_[block_index * total_sdf_block_size + voxel_index];
        const ushort sdf = compact.sdf_low | (compact.sdf_high << 8);
        const ushort secondary = compact.secondary_low | (compact.secondary_high << 8);
        T voxel;
        voxel.sdf = __half2float(__ushort_as_half(sdf));
        voxel.weight = compact.weight;
        voxel.sum_squared = __half2float(__ushort_as_half(secondary));
        voxel.rgb.x = compact.secondary_weight;
        return voxel;
      }

      void moveSDFBlock(const uint index, ChunkDesc<T>& target) {
        if (compact_voxels_) {
          target.addCompactSDFBlock(
            vecChunkDesc_[index], vecCompactVoxels_.data() + index * total_sdf_block_size);
        } else if (vecChunkDesc_[index].direction != static_cast<signed char>(TSDFDirection::none)) {
          target.addSDFBlock(vecChunkDesc_[index], vecSDFBlock_[index]);
        } else {
          target.vecChunkDesc_.push_back(vecChunkDesc_[index]);
          target.vecSDFBlock_.push_back(std::move(vecSDFBlock_[index]));
        }
      }

      void clear() {
        vecChunkDesc_.clear();
        vecSDFBlock_.clear();
        vecCompactVoxels_.clear();
        block_indices_.clear();
      }

      void removeLastSDFBlocks(const uint count) {
        const size_t remaining = vecChunkDesc_.size() - count;
        for (size_t index = remaining; index < vecChunkDesc_.size(); ++index)
          block_indices_.erase(vecChunkDesc_[index]);
        vecChunkDesc_.resize(remaining);
        if (compact_voxels_)
          vecCompactVoxels_.resize(remaining * total_sdf_block_size);
        else {
          while (vecSDFBlock_.size() > remaining)
            vecSDFBlock_.pop_back();
        }
      }

      void removeSDFBlock(const uint index) {
        const uint last = getNElements() - 1;
        block_indices_.erase(vecChunkDesc_[index]);
        if (index != last) {
          vecChunkDesc_[index] = vecChunkDesc_[last];
          if (compact_voxels_) {
            CompactVoxel* first = vecCompactVoxels_.data() + index * total_sdf_block_size;
            CompactVoxel* second = vecCompactVoxels_.data() + last * total_sdf_block_size;
            std::swap_ranges(first, first + total_sdf_block_size, second);
          } else {
            vecSDFBlock_[index] = std::move(vecSDFBlock_[last]);
          }
          if (block_indices_.find(vecChunkDesc_[index]) != block_indices_.end())
            block_indices_[vecChunkDesc_[index]] = index;
        }
        vecChunkDesc_.pop_back();
        if (compact_voxels_)
          vecCompactVoxels_.resize(last * total_sdf_block_size);
        else
          vecSDFBlock_.pop_back();
      }

      bool isStreamedOut() const {
        return vecChunkDesc_.size() > 0;
      }

      const data::vector<SDFBlockDesc>& getSDFBlockDescs() const {
        return vecChunkDesc_;
      }

      const data::vector<SDFBlock<T>>& getSDFBlocks() const {
        return vecSDFBlock_;
      }

      auto cista_members() {
        return std::tie(vecSDFBlock_, vecCompactVoxels_, vecChunkDesc_, compact_voxels_);
      }

      data::vector<SDFBlock<T>> vecSDFBlock_;
      data::vector<CompactVoxel> vecCompactVoxels_;
      data::vector<SDFBlockDesc> vecChunkDesc_;
      std::unordered_map<SDFBlockDesc, uint, SDFBlockDesc::HashSDFBlockDesc> block_indices_;
      bool compact_voxels_ = false;
    };

    template <typename T, typename Enable = void>
    class Streamer {
      Streamer() {
        std::cerr << "Streamer is not implemented for this type" << std::endl;
      }
    };

    template <typename T>
    class Streamer<T, std::enable_if_t<is_voxel_derived<T>::value>> {
    public:
      Streamer() {
      }
      Streamer(VoxelContainer<T>* c,
               const bool write_timings,
               const std::string& memory_allocation_filepath,
               const std::string& profiler_name) :
        container_(c),
        memory_allocation_filepath_(memory_allocation_filepath),
        streaming_profiler_(profiler_name, write_timings) {
      }

      ~Streamer() {
        destroy();
      }

      // Copy constructor - deleted because of raw CUDA pointers
      // Deep copying CUDA memory requires explicit handling
      Streamer(const Streamer&) = delete;

      // Copy assignment operator - deleted
      Streamer& operator=(const Streamer&) = delete;

      //  debugging
      void calculateMemoryUsage() const;
      double debugCheckForDuplicates() const;

      void create(const Eigen::Vector3f& voxel_extents,
                  const uint max_num_sdf_block_integrate_from_global_hash,
                  unsigned int initial_chunk_list_size,
                  bool compact_host_voxels);

      void clearGrid();
      void rechunk(float scale);
      void eraseChunk(const Eigen::Vector3i& chunk) {
        grid_.erase(chunk);
      }
      void destroy();

      // ! stream out - in, wrapper for both
      void stream(const Eigen::Vector3f& camera_position, const float radius);
      void stream(const Camera& camera, const Eigen::Isometry3f& camera_in_world);
      void stream(const Camera& camera,
                  const Eigen::Isometry3f& camera_in_world,
                  const CUDAMatrixf& depth);
      void stream(const Camera& camera,
                  const Eigen::Isometry3f& camera_in_world,
                  const CUDAMatrixf& depth,
                  int row_begin,
                  int row_end);

      // ! stream out - to host
      void streamOutToHostPass0(const float3& camera_pose, const float radius);
      void streamOutToHostPass0(const Camera& camera);
      void streamOutToHostPass0(const Camera& camera, const CUDAMatrixf& depth);
      void streamOutToHostPass0(const Camera& camera,
                                const CUDAMatrixf& depth,
                                int row_begin,
                                int row_end);
      void integrateFromGlobalHashPass1(const float radius, const float3& camera_position);
      void integrateFromGlobalHashPass1(const float radius, const float3& camera_position, const int num_pass);
      void integrateFromGlobalHashPass1(const Camera& camera, const int num_pass);
      void integrateFromGlobalHashPass1(const Camera& camera,
                                        const CUDAMatrixf& depth,
                                        const int num_pass);
      void integrateFromGlobalHashPass1(const Camera& camera,
                                        const CUDAMatrixf& depth,
                                        int row_begin,
                                        int row_end,
                                        int num_pass);
      void integrateFromGlobalHashPass2(const uint num_SDF_block_desc);

      void streamAllOut(); // usually at the end
      void integrateFromGlobalHashPass1(const int num_pass);
      void integrateFromGlobalHashPass2(const uint num_SDF_block_desc, const int num_pass);

      void streamOutToCPUPass1CPU();
      void integrateInChunkGrid(const SDFBlockDesc* h_SDFBlockDescOutput_, const T* h_SDFBlockOutput_);
      void integrateCompactInChunkGrid(const SDFBlockDesc* block_descs, const CompactVoxel* voxels);
      std::vector<Eigen::Vector3i> chunks();
      void discardChunks(const std::vector<Eigen::Vector3i>& chunks);

      // ! stream in - to device
      uint integrateInHash(const Eigen::Vector3f& camera_pose, float radius);
      uint integrateInHash(const Camera& camera, const Eigen::Isometry3f& camera_in_world);
      uint integrateInHash(const Camera& camera,
                           const Eigen::Isometry3f& camera_in_world,
                           const std::vector<float>& depth);
      uint integrateInHash(const Camera& camera,
                           const Eigen::Isometry3f& camera_in_world,
                           const std::vector<float>& depth,
                           int row_begin,
                           int row_end);
      void streamInToGPU(const Eigen::Vector3f& camera_position, const float radius);
      void streamInToGPU(const Camera& camera, const Eigen::Isometry3f& camera_in_world);
      void streamInToGPU(const Camera& camera,
                         const Eigen::Isometry3f& camera_in_world,
                         const std::vector<float>& depth);
      void streamInToGPU(const Camera& camera,
                         const Eigen::Isometry3f& camera_in_world,
                         const std::vector<float>& depth,
                         int row_begin,
                         int row_end);
      bool streamInDone() const {
        return stream_in_done_;
      }
      void mergeResidentBlocksFromHost();
      void streamInToGPUChunk(const Eigen::Vector3i& chunk);
      void streamInToGPUChunkNeighborhood(const Eigen::Vector3i& chunk, const int radius);
      void chunkToGlobalHashPass1(const uint num_sdf_blocks_descs,
                                  const uint heap_count_prev,
                                  const SDFBlockDesc* d_SDFBlockDescs,
                                  uint* d_blocks_ptr,
                                  uchar* d_merge_blocks);
      void chunkToGlobalHashPass2(const uint num_sdf_blocks_descs,
                                  const uint heap_count_prev,
                                  const SDFBlockDesc* d_SDFBlockDescs,
                                  const T* d_SDFBlocks,
                                  uint* d_blocks_ptr,
                                  const uchar* d_merge_blocks);
      void chunkToGlobalHashPass2Compact(const uint num_sdf_blocks_descs,
                                         const SDFBlockDesc* block_descs,
                                         const CompactVoxel* voxels,
                                         uint* block_pointers,
                                         const uchar* merge_blocks);

      // ! some other utils
      inline void clearSDFBlockCounter() {
        const uint value = 0;
        CUDA_CHECK(cudaMemcpy(d_SDF_block_counter_, &value, sizeof(uint), cudaMemcpyHostToDevice));
      }

      inline uint getSDFBlockCounter() const {
        uint dest;
        CUDA_CHECK(cudaMemcpy(&dest, d_SDF_block_counter_, sizeof(uint), cudaMemcpyDeviceToHost));
        return dest;
      }

      inline Eigen::Vector3i worldToChunks(const Eigen::Vector3f& pw) const {
        Eigen::Vector3f p;

        p.x() = pw.x() / voxel_extents_.x();
        p.y() = pw.y() / voxel_extents_.y();
        p.z() = pw.z() / voxel_extents_.z();

        Eigen::Vector3f s = (Eigen::Vector3f) p.array().sign();
        return (p + s * 0.5f).cast<int>();
      }

      inline Eigen::Vector3i worldPointToVirtualVoxelPos(const Eigen::Vector3f& pw) {
        const Eigen::Vector3f p = pw / container_->virtual_voxel_size_;

        const Eigen::Vector3f s = (Eigen::Vector3f) p.array().sign();
        return (p + s * 0.5f).cast<int>();
      }

      inline Eigen::Vector3i worldPointToSDFBlockPos(const Eigen::Vector3f& pw) {
        return virtualVoxelPosToSDFBlock(worldPointToVirtualVoxelPos(pw));
      }

      inline uint linearizeVoxelPos(const Eigen::Vector3i& pos) {
        return pos.z() * sdf_block_size * sdf_block_size + pos.y() * sdf_block_size + pos.x();
      }

      inline uint worldPointToSDFBlockIndex(const Eigen::Vector3f& pw, const int block_size = sdf_block_size) {
        Eigen::Vector3i virtual_voxel_pos = worldPointToVirtualVoxelPos(pw);
        Eigen::Vector3i local_voxel_pos;
        local_voxel_pos.x() = virtual_voxel_pos.x() % block_size;
        local_voxel_pos.y() = virtual_voxel_pos.y() % block_size;
        local_voxel_pos.z() = virtual_voxel_pos.z() % block_size;
        if (local_voxel_pos.x() < 0)
          local_voxel_pos.x() += block_size;
        if (local_voxel_pos.y() < 0)
          local_voxel_pos.y() += block_size;
        if (local_voxel_pos.z() < 0)
          local_voxel_pos.z() += block_size;

        return linearizeVoxelPos(local_voxel_pos);
      }

      inline Eigen::Vector3i virtualVoxelPosToSDFBlock(Eigen::Vector3i virtual_voxel_pos) {
        if (virtual_voxel_pos.x() < 0)
          virtual_voxel_pos.x() -= sdf_block_size - 1; // i.e voxelBlock virtual_voxel_size -1
        if (virtual_voxel_pos.y() < 0)
          virtual_voxel_pos.y() -= sdf_block_size - 1;
        if (virtual_voxel_pos.z() < 0)
          virtual_voxel_pos.z() -= sdf_block_size - 1;
        return Eigen::Vector3i(
          virtual_voxel_pos.x() / sdf_block_size, virtual_voxel_pos.y() / sdf_block_size, virtual_voxel_pos.z() / sdf_block_size);
      }

      inline Eigen::Vector3f chunkToWorld(const Eigen::Vector3i& chunk_pose) const {
        Eigen::Vector3f res;
        res << chunk_pose.x() * voxel_extents_.x(), chunk_pose.y() * voxel_extents_.y(), chunk_pose.z() * voxel_extents_.z();
        return res;
      }

      inline const Eigen::Vector3i meterToNumberOfChunksCeil(const float f) const {
        return Eigen::Vector3i(
          (int) ceil(f / voxel_extents_.x()), (int) ceil(f / voxel_extents_.y()), (int) ceil(f / voxel_extents_.z()));
      }

      float getChunkRadiusInMeter() const {
        return voxel_extents_.norm() / 2.0f;
      }

      const Eigen::Vector3f& getChunkExtents() const {
        return voxel_extents_;
      }

      bool containSDFBlocksChunk(const Eigen::Vector3i& chunk) const {
        auto it = grid_.find(chunk);
        return ((it != grid_.end()) && (grid_.at(chunk)->isStreamedOut()));
      }

      bool containSDFBlocksChunkNeighborhood(const Eigen::Vector3i& chunk,
                                             const Eigen::Vector3i& min_grid_pos,
                                             const Eigen::Vector3i& max_grid_pos,
                                             const int radius) const {
        const Eigen::Vector3i start_chunk = Eigen::Vector3i(std::max(chunk.x() - radius, min_grid_pos.x()),
                                                            std::max(chunk.y() - radius, min_grid_pos.y()),
                                                            std::max(chunk.z() - radius, min_grid_pos.z()));
        const Eigen::Vector3i end_chunk   = Eigen::Vector3i(std::min(chunk.x() + radius, max_grid_pos.x()),
                                                          std::min(chunk.y() + radius, max_grid_pos.y()),
                                                          std::min(chunk.z() + radius, max_grid_pos.z()));

        for (int x = start_chunk.x(); x < end_chunk.x(); ++x) {
          for (int y = start_chunk.y(); y < end_chunk.y(); ++y) {
            for (int z = start_chunk.z(); z < end_chunk.z(); ++z) {
              if (containSDFBlocksChunk(Eigen::Vector3i(x, y, z)))
                return true;
            }
          }
        }
        return false;
      }

      bool isChunkInSphere(const Eigen::Vector3i& chunk, const Eigen::Vector3f& center, float radius) const {
        const Eigen::Vector3f world_pose = chunkToWorld(chunk); // chunk center
        const float l                    = (world_pose - center).norm();
        if (l <= std::abs(radius - chunk_radius_))
          return true;
        return false;
      }

      void serializeData(const std::string& filename_hash  = "./data/hash_points.ply",
                         const std::string& filename_voxel = "./data/voxel_points.ply") const;

      std::pair<Eigen::Vector3i, Eigen::Vector3i> computeBounds() {
        Eigen::Vector3i min_grid_pos(
          std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
        Eigen::Vector3i max_grid_pos(
          std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest(), std::numeric_limits<int>::lowest());
        for (const auto& [chunk_pos, _] : grid_) {
          min_grid_pos = min_grid_pos.cwiseMin(chunk_pos);
          max_grid_pos = max_grid_pos.cwiseMax(chunk_pos);
        }
        return {min_grid_pos, max_grid_pos};
      }

      const std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash>& getGrid() const {
        return grid_;
      }

      inline void printStatistics() const {
        uint num_SDF_blocks = 0;
        for (const auto& [chunk_pos, chunk_ptr] : grid_) {
          num_SDF_blocks += chunk_ptr->getNElements();
        }
        std::cout << "Streamer | total number of blocks in RAM: " << num_SDF_blocks << std::endl;
      }

      // protected:
      VoxelContainer<T>* container_ = nullptr;

      std::unordered_map<Eigen::Vector3i, std::unique_ptr<ChunkDesc<T>>, Vector3iHash> grid_;

      // output host and device
      SDFBlockDesc* h_SDFBlockDescOutput_ = nullptr;
      T* h_SDFBlockOutput_                = nullptr;
      SDFBlockDesc* d_SDFBlockDescOutput_ = nullptr;
      T* d_SDFBlockOutput_                = nullptr;

      // input
      SDFBlockDesc* d_SDFBlockDescInput_ = nullptr;
      T* d_SDFBlockInput_                = nullptr;
      CompactVoxel* h_compact_voxel_output_ = nullptr;
      CompactVoxel* d_compact_voxel_output_ = nullptr;
      CompactVoxel* d_compact_voxel_input_ = nullptr;

      uint* d_SDF_block_counter_ = nullptr;
      uint* d_voxel_offsets_ = nullptr;
      uint* d_blocks_ptr_ = nullptr;
      uchar* d_merge_blocks_ = nullptr;

      uint curr_stream_out_blocks_                       = 0;
      uint curr_stream_in_blocks_                        = 0;
      uint max_num_sdf_block_integrate_from_global_hash_ = 0;

      uint initial_chunk_list_size_;
      bool compact_host_voxels_;

      Eigen::Vector3f voxel_extents_;

      float chunk_radius_;
      bool stream_in_done_;

      // asynchronous streams gpu
      cudaEvent_t start_event_, stop_event_;

      std::string memory_allocation_filepath_;
      CUDAProfiler streaming_profiler_;
      float elapsed_time = 0.f;
    };

    template class Streamer<Voxel>;
    using GeometricStreamer = Streamer<Voxel>;

  } // namespace cugeoutils
} // namespace cupanutils
