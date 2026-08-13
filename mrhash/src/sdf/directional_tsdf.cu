#include "voxel_data_structures.cuh"

#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <vector>

namespace cupanutils {
  namespace cugeoutils {

    namespace {

      constexpr float direction_threshold = 0.3826834323650898f;
      constexpr float cluster_coherence_threshold = 0.9238795325112867f;
      constexpr int normal_filter_radius = 5;
      constexpr float normal_filter_sigma_distance = 2.5f;
      constexpr float normal_filter_sigma_range = 5.f;

      struct VoxelKey {
        int x;
        int y;
        int z;

        __host__ __device__ bool operator==(const VoxelKey& other) const {
          return x == other.x && y == other.y && z == other.z;
        }
      };

      struct CandidateKey {
        VoxelKey voxel;
        signed char cluster;
      };

      struct CandidateValue {
        float3 point_sum;
        float3 normal_sum;
        float weight;
      };

      struct SurfaceValue {
        float3 point;
        float3 normal;
        float confidence;
      };


      struct CandidateKeyLess {
        __host__ __device__ bool operator()(const CandidateKey& left, const CandidateKey& right) const {
          if (left.voxel.x != right.voxel.x)
            return left.voxel.x < right.voxel.x;
          if (left.voxel.y != right.voxel.y)
            return left.voxel.y < right.voxel.y;
          if (left.voxel.z != right.voxel.z)
            return left.voxel.z < right.voxel.z;
          return left.cluster < right.cluster;
        }
      };

      struct CandidateKeyEqual {
        __host__ __device__ bool operator()(const CandidateKey& left, const CandidateKey& right) const {
          return left.voxel == right.voxel && left.cluster == right.cluster;
        }
      };

      struct VoxelKeyLess {
        __host__ __device__ bool operator()(const VoxelKey& left, const VoxelKey& right) const {
          if (left.x != right.x)
            return left.x < right.x;
          if (left.y != right.y)
            return left.y < right.y;
          return left.z < right.z;
        }
      };

      struct CandidateValueAdd {
        __host__ __device__ CandidateValue operator()(const CandidateValue& left, const CandidateValue& right) const {
          return CandidateValue{
            left.point_sum + right.point_sum,
            left.normal_sum + right.normal_sum,
            left.weight + right.weight};
        }
      };

      struct MaximumConfidence {
        __host__ __device__ SurfaceValue operator()(const SurfaceValue& left, const SurfaceValue& right) const {
          return left.confidence >= right.confidence ? left : right;
        }
      };

      __device__ signed char normalCluster(const float3 normal) {
        const float3 magnitude = make_float3(fabsf(normal.x), fabsf(normal.y), fabsf(normal.z));
        if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z)
          return normal.x >= 0.f ? 0 : 1;
        if (magnitude.y >= magnitude.z)
          return normal.y >= 0.f ? 2 : 3;
        return normal.z >= 0.f ? 4 : 5;
      }

      __device__ bool validDepth(const float depth, const Camera* camera) {
        return depth > camera->minDepth() && depth <= camera->maxDepth();
      }

      __global__ void estimateNormalsKernel(
        const CUDAMatrixf* depth, const Camera* camera, CUDAMatrixf3* normals) {
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int column = blockDim.x * blockIdx.x + threadIdx.x;
        if (!depth->inside(row, column))
          return;
        normals->at<1>(row, column) = make_float3(0.f);
        const float center_depth = depth->at<1>(row, column);
        if (!validDepth(center_depth, camera))
          return;
        const int row_offsets[4] = {0, 1, 0, -1};
        const int column_offsets[4] = {1, 0, -1, 0};
        const float3 center = camera->inverseProjection(row, column, center_depth);
        float3 normal_sum = make_float3(0.f);
        int triangle_count = 0;
        for (int triangle = 0; triangle < 4; ++triangle) {
          const int next = (triangle + 1) % 4;
          const int first_row = row + row_offsets[triangle];
          const int first_column = column + column_offsets[triangle];
          const int second_row = row + row_offsets[next];
          const int second_column = column + column_offsets[next];
          if (!depth->inside(first_row, first_column) || !depth->inside(second_row, second_column))
            continue;
          const float first_depth = depth->at<1>(first_row, first_column);
          const float second_depth = depth->at<1>(second_row, second_column);
          if (!validDepth(first_depth, camera) || !validDepth(second_depth, camera))
            continue;
          const float3 first = camera->inverseProjection(
            first_row, first_column, first_depth);
          const float3 second = camera->inverseProjection(
            second_row, second_column, second_depth);
          float3 normal = -1.f * normalize(cross(first - center, second - center));
          if (dot(normal, -1.f * normalize(center)) < 0.f)
            normal = -normal;
          normal_sum += normal;
          ++triangle_count;
        }
        if (triangle_count > 0)
          normals->at<1>(row, column) = normalize(normal_sum);
      }

      __global__ void filterNormalsKernel(
        const CUDAMatrixf3* input, CUDAMatrixf3* output) {
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int column = blockDim.x * blockIdx.x + threadIdx.x;
        if (!input->inside(row, column))
          return;
        const float3 center = input->at<1>(row, column);
        if (dot(center, center) == 0.f) {
          output->at<1>(row, column) = make_float3(0.f);
          return;
        }
        const float spatial_factor = 1.f /
                                     (2.f * normal_filter_sigma_distance * normal_filter_sigma_distance);
        const float range_factor = 1.f /
                                   (2.f * normal_filter_sigma_range * normal_filter_sigma_range);
        float3 sum = make_float3(0.f);
        float weight_sum = 0.f;
        for (int row_offset = -normal_filter_radius; row_offset <= normal_filter_radius; ++row_offset) {
          for (int column_offset = -normal_filter_radius; column_offset <= normal_filter_radius; ++column_offset) {
            const int neighbor_row = row + row_offset;
            const int neighbor_column = column + column_offset;
            if (neighbor_row < 0 || neighbor_column < 0 || neighbor_row >= static_cast<int>(input->rows()) ||
                neighbor_column >= static_cast<int>(input->cols()))
              continue;
            const float3 value = input->at<1>(neighbor_row, neighbor_column);
            if (dot(value, value) == 0.f)
              continue;
            const float3 difference = value - center;
            const float spatial_distance = row_offset * row_offset + column_offset * column_offset;
            const float weight = expf(-spatial_distance * spatial_factor - dot(difference, difference) * range_factor);
            sum += weight * value;
            weight_sum += weight;
          }
        }
        output->at<1>(row, column) = normalize(sum / weight_sum);
      }

      template <typename T>
      __device__ void allocateDirectionalVoxel(
        VoxelContainer<T>* container,
        const int3 voxel_index,
        const TSDFDirection direction,
        uint* allocation_failed) {
        const int3 block = virtualVoxelPosToSDFBlock(
          voxel_index, container->virtual_voxel_size_, container->voxel_extents_);
        if (container->allocDirectionalBlock(block, direction) == -2)
          atomicExch(allocation_failed, 1);
      }

      template <typename T>
      __device__ void traverseDirectionalBand(
        const int row,
        const int column,
        const CUDAMatrixf* depth,
        const CUDAMatrixf3* normals,
        const Camera* camera,
        VoxelContainer<T>* container,
        uint* allocation_failed,
        const int row_begin,
        const int row_end) {
        if (row < row_begin || row >= row_end)
          return;
        const float depth_value = depth->at<1>(row, column);
        const float3 normal_camera = normals->at<1>(row, column);
        if (!validDepth(depth_value, camera) || dot(normal_camera, normal_camera) == 0.f)
          return;
        const float3 surface_camera = camera->inverseProjection(row, column, depth_value);
        const float3 view_ray_camera = normalize(surface_camera);
        const float view_weight = dot(normal_camera, -1.f * view_ray_camera);
        if (view_weight <= 0.f)
          return;
        const float3 surface_world = camera->camInWorld() * surface_camera;
        const float3 normal_world = normalize(camera->camInWorld().rotation * normal_camera);
        const float truncation = directional_truncation_voxels * container->virtual_voxel_size_;
        const float3 segment_start = surface_world + truncation * normal_world;
        const float3 segment_end = surface_world - truncation * normal_world;
        const float3 segment_direction = normalize(segment_end - segment_start);
        for (int direction_index = 0; direction_index < directional_tsdf_count; ++direction_index) {
          const TSDFDirection direction = static_cast<TSDFDirection>(direction_index);
          const float direction_weight = dot(normal_world, directionVector(direction));
          if (direction_weight <= direction_threshold)
            continue;
          int3 voxel_index = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, segment_start);
          const int3 end_voxel = worldPointToVirtualVoxelPos(container->virtual_voxel_size_, segment_end);
          const int3 step = sign(segment_direction);
          const float maximum = 1e30f;
          float3 next_boundary = virtualVoxelPosToWorld(container->virtual_voxel_size_, voxel_index);
          next_boundary += 0.5f * container->virtual_voxel_size_ * make_float3(step);
          float3 distance_to_boundary = (next_boundary - segment_start) / segment_direction;
          float3 distance_per_voxel = container->virtual_voxel_size_ / make_float3(
            fabsf(segment_direction.x), fabsf(segment_direction.y), fabsf(segment_direction.z));
          if (step.x == 0) {
            distance_to_boundary.x = maximum;
            distance_per_voxel.x = maximum;
          }
          if (step.y == 0) {
            distance_to_boundary.y = maximum;
            distance_per_voxel.y = maximum;
          }
          if (step.z == 0) {
            distance_to_boundary.z = maximum;
            distance_per_voxel.z = maximum;
          }
          const int traversal_count = abs(end_voxel.x - voxel_index.x) +
                                      abs(end_voxel.y - voxel_index.y) +
                                      abs(end_voxel.z - voxel_index.z) + 1;
          for (int traversal = 0; traversal < traversal_count; ++traversal) {
            allocateDirectionalVoxel(container, voxel_index, direction, allocation_failed);
            if (traversal + 1 == traversal_count)
              break;
            if (distance_to_boundary.x < distance_to_boundary.y &&
                distance_to_boundary.x < distance_to_boundary.z) {
              voxel_index.x += step.x;
              distance_to_boundary.x += distance_per_voxel.x;
            } else if (distance_to_boundary.z < distance_to_boundary.y) {
              voxel_index.z += step.z;
              distance_to_boundary.z += distance_per_voxel.z;
            } else {
              voxel_index.y += step.y;
              distance_to_boundary.y += distance_per_voxel.y;
            }
          }
        }
      }

      template <typename T>
      __global__ void allocateDirectionalBlocksKernel(
        const CUDAMatrixf* depth,
        const CUDAMatrixf3* normals,
        const Camera* camera,
        VoxelContainer<T>* container,
        uint* allocation_failed,
        const int row_begin,
        const int row_end) {
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int column = blockDim.x * blockIdx.x + threadIdx.x;
        if (depth->inside(row, column))
          traverseDirectionalBand(
            row, column, depth, normals, camera, container, allocation_failed, row_begin, row_end);
      }

      template <typename T>
      __global__ void integrateDirectionalVisibleBlocksKernel(
        const CUDAMatrixf* depth,
        const CUDAMatrixf3* normals,
        const Camera* camera,
        VoxelContainer<T>* container,
        const int row_begin,
        const int row_end) {
        const HashEntry entry = container->d_compactHashTable_[blockIdx.x];
        const TSDFDirection direction = static_cast<TSDFDirection>(entry.direction);
        const int3 voxel_index = SDFBlockToVirtualVoxelPos(entry.pos) +
                                 make_int3(delinearizeVoxelPos(threadIdx.x));
        const float3 voxel_world = virtualVoxelPosToWorld(container->virtual_voxel_size_, voxel_index);
        const float3 voxel_camera = camera->camInWorld().inverse() * voxel_world;
        int2 pixel;
        if (!camera->projectPoint(voxel_camera, pixel))
          return;
        if (pixel.x < row_begin || pixel.x >= row_end)
          return;
        const float depth_value = depth->at<1>(pixel.x, pixel.y);
        const float3 normal_camera = normals->at<1>(pixel.x, pixel.y);
        if (!validDepth(depth_value, camera) || dot(normal_camera, normal_camera) == 0.f)
          return;
        const float3 surface_camera = camera->inverseProjection(pixel.x, pixel.y, depth_value);
        const float3 surface_world = camera->camInWorld() * surface_camera;
        const float3 normal_world = normalize(camera->camInWorld().rotation * normal_camera);
        const float direction_weight = dot(normal_world, directionVector(direction));
        if (direction_weight <= direction_threshold)
          return;
        const float3 difference = voxel_world - surface_world;
        const float truncation = directional_truncation_voxels * container->virtual_voxel_size_;
        const float distance = dot(difference, normal_world);
        if (fabsf(distance) > truncation)
          return;
        const float view_weight = dot(normal_camera, -1.f * normalize(surface_camera));
        if (view_weight <= 0.f)
          return;
        const float depth_ratio = camera->minDepth() / depth_value;
        T& voxel = container->d_SDFBlocks_[entry.ptr + threadIdx.x];
        float fusion_weight = depth_ratio * depth_ratio * view_weight * direction_weight;
        if (voxel.sum_squared > 0.f) {
          const float residual = distance - voxel.sdf / voxel.sum_squared;
          const float huber_scale = 2.f * container->virtual_voxel_size_;
          if (fabsf(residual) > huber_scale)
            fusion_weight *= huber_scale / fabsf(residual);
        }
        voxel.sdf += fusion_weight * distance;
        voxel.sum_squared += fusion_weight;
      }

      template <typename T>
      __device__ bool directionalDistance(
        const VoxelContainer<T>* container,
        const int3 position,
        const TSDFDirection direction,
        float& distance,
        float& weight) {
        const T voxel = container->getVoxel(position, direction);
        weight = voxel.sum_squared;
        if (weight <= 0.f)
          return false;
        distance = voxel.sdf / weight;
        return true;
      }

      template <typename T>
      __device__ bool directionalGradient(
        const VoxelContainer<T>* container,
        const int3 position,
        const TSDFDirection direction,
        float3& gradient) {
        float center_distance;
        float center_weight;
        if (!directionalDistance(container, position, direction, center_distance, center_weight))
          return false;
        float components[3];
        for (int axis = 0; axis < 3; ++axis) {
          int3 negative = position;
          int3 positive = position;
          (&negative.x)[axis] -= 1;
          (&positive.x)[axis] += 1;
          float negative_distance;
          float negative_weight;
          float positive_distance;
          float positive_weight;
          const bool has_negative = directionalDistance(
            container, negative, direction, negative_distance, negative_weight);
          const bool has_positive = directionalDistance(
            container, positive, direction, positive_distance, positive_weight);
          if (has_negative && has_positive)
            components[axis] = 0.5f * (positive_distance - negative_distance);
          else if (has_positive)
            components[axis] = positive_distance - center_distance;
          else if (has_negative)
            components[axis] = center_distance - negative_distance;
          else
            components[axis] = 0.f;
        }
        gradient = make_float3(components[0], components[1], components[2]);
        const float squared_length = dot(gradient, gradient);
        if (squared_length == 0.f)
          return false;
        gradient /= sqrtf(squared_length);
        return true;
      }

      template <typename T>
      __device__ uint ownedCrossingCount(
        const VoxelContainer<T>* container,
        const int3 position,
        const TSDFDirection direction,
        const float3 owner_minimum,
        const float3 owner_maximum) {
        float center_distance;
        float center_weight;
        if (!directionalDistance(container, position, direction, center_distance, center_weight))
          return 0;
        uint count = 0;
        for (int axis = 0; axis < 3; ++axis) {
          int3 neighbor = position;
          (&neighbor.x)[axis] += 1;
          float neighbor_distance;
          float neighbor_weight;
          if (!directionalDistance(container, neighbor, direction, neighbor_distance, neighbor_weight) ||
              !((center_distance <= 0.f && neighbor_distance > 0.f) ||
                (center_distance > 0.f && neighbor_distance <= 0.f)))
            continue;
          const float interpolation = center_distance / (center_distance - neighbor_distance);
          int3 surface_voxel = position;
          if (interpolation >= 0.5f)
            (&surface_voxel.x)[axis] += 1;
          const float3 world = virtualVoxelPosToWorld(container->virtual_voxel_size_, surface_voxel);
          if (world.x >= owner_minimum.x && world.y >= owner_minimum.y && world.z >= owner_minimum.z &&
              world.x < owner_maximum.x && world.y < owner_maximum.y && world.z < owner_maximum.z)
            ++count;
        }
        return count;
      }

      template <typename T>
      __global__ void countCrossingsKernel(
        const VoxelContainer<T>* container,
        const int3 owner_chunk,
        const float3 chunk_extents,
        uint* crossing_count) {
        const uint compact_block = blockIdx.x;
        const uint local_index = threadIdx.x;
        const HashEntry entry = container->d_compactHashTable_[compact_block];
        const int3 position = SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(delinearizeVoxelPos(local_index));
        const float3 owner_center = make_float3(owner_chunk) * chunk_extents;
        const float3 owner_minimum = owner_center - 0.5f * chunk_extents;
        const float3 owner_maximum = owner_center + 0.5f * chunk_extents;
        const TSDFDirection direction = static_cast<TSDFDirection>(entry.direction);
        atomicAdd(
          crossing_count,
          ownedCrossingCount(container, position, direction, owner_minimum, owner_maximum));
      }

      template <typename T>
      __global__ void fillCrossingsKernel(
        const VoxelContainer<T>* container,
        CandidateKey* keys,
        CandidateValue* values,
        uint* output_count,
        const int3 owner_chunk,
        const float3 chunk_extents) {
        const uint compact_block = blockIdx.x;
        const uint local_index = threadIdx.x;
        const HashEntry entry = container->d_compactHashTable_[compact_block];
        const TSDFDirection direction = static_cast<TSDFDirection>(entry.direction);
        const int3 position = SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(delinearizeVoxelPos(local_index));
        float center_distance;
        float center_weight;
        if (!directionalDistance(container, position, direction, center_distance, center_weight))
          return;
        for (int axis = 0; axis < 3; ++axis) {
          int3 neighbor = position;
          (&neighbor.x)[axis] += 1;
          float neighbor_distance;
          float neighbor_weight;
          if (!directionalDistance(container, neighbor, direction, neighbor_distance, neighbor_weight) ||
              !((center_distance <= 0.f && neighbor_distance > 0.f) ||
                (center_distance > 0.f && neighbor_distance <= 0.f)))
            continue;
          const float interpolation = center_distance / (center_distance - neighbor_distance);
          float3 center_gradient;
          float3 neighbor_gradient;
          if (!directionalGradient(container, position, direction, center_gradient) ||
              !directionalGradient(container, neighbor, direction, neighbor_gradient))
            continue;
          const float3 normal = normalize((1.f - interpolation) * center_gradient + interpolation * neighbor_gradient);
          if (dot(normal, directionVector(direction)) <= direction_threshold)
            continue;
          const float confidence = fminf(center_weight, neighbor_weight);
          float3 surface_point = virtualVoxelPosToWorld(container->virtual_voxel_size_, position);
          (&surface_point.x)[axis] += interpolation * container->virtual_voxel_size_;
          int3 surface_voxel = position;
          if (interpolation >= 0.5f)
            (&surface_voxel.x)[axis] += 1;
          const float3 owner_center = make_float3(owner_chunk) * chunk_extents;
          const float3 owner_minimum = owner_center - 0.5f * chunk_extents;
          const float3 owner_maximum = owner_center + 0.5f * chunk_extents;
          const float3 world = virtualVoxelPosToWorld(container->virtual_voxel_size_, surface_voxel);
          if (world.x < owner_minimum.x || world.y < owner_minimum.y || world.z < owner_minimum.z ||
              world.x >= owner_maximum.x || world.y >= owner_maximum.y || world.z >= owner_maximum.z)
            continue;
          const uint output = atomicAdd(output_count, 1u);
          keys[output] = CandidateKey{
            VoxelKey{surface_voxel.x, surface_voxel.y, surface_voxel.z}, normalCluster(normal)};
          values[output] = CandidateValue{
            confidence * surface_point, confidence * normal, confidence};
        }
      }

      template <typename T>
      __global__ void voteClustersKernel(
        const VoxelContainer<T>*,
        const CandidateKey* keys,
        const CandidateValue* values,
        const uint cluster_count,
        VoxelKey* output_keys,
        SurfaceValue* output_values,
        uint* output_count) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= cluster_count)
          return;
        const CandidateValue candidate = values[index];
        const float normal_sum_length = length(candidate.normal_sum);
        if (normal_sum_length < cluster_coherence_threshold * candidate.weight)
          return;
        const float3 normal = candidate.normal_sum / normal_sum_length;
        const VoxelKey key = keys[index].voxel;
        const uint output = atomicAdd(output_count, 1u);
        output_keys[output] = key;
        output_values[output] = SurfaceValue{
          candidate.point_sum / candidate.weight, normal, candidate.weight};
      }

      struct DeviceTriangle {
        float3 vertices[3];
      };

      template <typename T>
      __device__ bool directionalRawCombinedDistance(
        const VoxelContainer<T>* container,
        const int3 position,
        float& distance) {
        float distances[directional_tsdf_count];
        float weights[directional_tsdf_count];
        int strongest = -1;
        float strongest_weight = 0.f;
        for (int direction_index = 0; direction_index < directional_tsdf_count; ++direction_index) {
          const TSDFDirection direction = static_cast<TSDFDirection>(direction_index);
          if (!directionalDistance(
                container, position, direction, distances[direction_index], weights[direction_index])) {
            weights[direction_index] = 0.f;
            continue;
          }
          if (weights[direction_index] > strongest_weight) {
            strongest = direction_index;
            strongest_weight = weights[direction_index];
          }
        }
        if (strongest < 0)
          return false;
        float distance_sum = 0.f;
        float weight_sum = 0.f;
        for (int direction_index = 0; direction_index < directional_tsdf_count; ++direction_index) {
          if (weights[direction_index] > 0.f &&
              dot(
                directionVector(static_cast<TSDFDirection>(direction_index)),
                directionVector(static_cast<TSDFDirection>(strongest))) >= 0.f) {
            distance_sum += weights[direction_index] * distances[direction_index];
            weight_sum += weights[direction_index];
          }
        }
        distance = distance_sum / weight_sum;
        return true;
      }

      template <typename T>
      __device__ bool directionalRawCombinedSample(
        const VoxelContainer<T>* container,
        const int3 position,
        float& distance,
        float3& gradient) {
        if (!directionalRawCombinedDistance(container, position, distance))
          return false;
        int strongest = -1;
        float strongest_weight = 0.f;
        float weights[directional_tsdf_count];
        for (int direction_index = 0; direction_index < directional_tsdf_count; ++direction_index) {
          float direction_distance;
          const TSDFDirection direction = static_cast<TSDFDirection>(direction_index);
          if (!directionalDistance(
                container, position, direction, direction_distance, weights[direction_index])) {
            weights[direction_index] = 0.f;
            continue;
          }
          if (weights[direction_index] > strongest_weight) {
            strongest = direction_index;
            strongest_weight = weights[direction_index];
          }
        }
        gradient = make_float3(0.f);
        float gradient_weight = 0.f;
        for (int direction_index = 0; direction_index < directional_tsdf_count; ++direction_index) {
          const TSDFDirection direction = static_cast<TSDFDirection>(direction_index);
          float3 direction_gradient;
          if (weights[direction_index] > 0.f &&
              dot(directionVector(direction), directionVector(static_cast<TSDFDirection>(strongest))) >= 0.f &&
              directionalGradient(container, position, direction, direction_gradient)) {
            gradient += weights[direction_index] * direction_gradient;
            gradient_weight += weights[direction_index];
          }
        }
        if (gradient_weight == 0.f)
          return false;
        gradient = normalize(gradient);
        return true;
      }

      template <typename T>
      __device__ bool directionalCombinedDistance(
        const VoxelContainer<T>* container,
        const int3 position,
        float& distance) {
        if (directionalRawCombinedDistance(container, position, distance))
          return true;
        for (int radius = 1; radius <= 2; ++radius) {
          float distance_sum = 0.f;
          int distance_count = 0;
          for (int z = -radius; z <= radius; ++z) {
            for (int y = -radius; y <= radius; ++y) {
              for (int x = -radius; x <= radius; ++x) {
                if (max(max(abs(x), abs(y)), abs(z)) != radius)
                  continue;
                const int3 neighbor = position + make_int3(x, y, z);
                float neighbor_distance;
                float3 neighbor_gradient;
                if (directionalRawCombinedSample(
                      container, neighbor, neighbor_distance, neighbor_gradient)) {
                  const float3 displacement = container->virtual_voxel_size_ * make_float3(position - neighbor);
                  distance_sum += neighbor_distance + dot(displacement, neighbor_gradient);
                  ++distance_count;
                }
              }
            }
          }
          if (distance_count > 0) {
            distance = distance_sum / distance_count;
            return true;
          }
        }
        return false;
      }

      template <typename T>
      __device__ bool canonicalSpatialBlock(const VoxelContainer<T>* container, const HashEntry entry) {
        for (int direction_index = 0; direction_index < entry.direction; ++direction_index) {
          if (container->getHashEntry(entry.pos, static_cast<TSDFDirection>(direction_index)).ptr != FREE_ENTRY)
            return false;
        }
        return true;
      }

      template <typename T>
      __device__ uint directionalCombinedCellTriangles(
        const VoxelContainer<T>* container,
        const int3 cell,
        DeviceTriangle* triangles) {
        float distances[8];
        unsigned char cube_index = 0;
        for (int corner = 0; corner < 8; ++corner) {
          const int3 position = cell + make_int3((corner >> 2) & 1, (corner >> 1) & 1, corner & 1);
          if (!directionalCombinedDistance(container, position, distances[corner]))
            return 0;
          if (distances[corner] <= 0.f)
            cube_index |= 1u << corner;
        }
        if (cube_index == 0 || cube_index == 255)
          return 0;
        const RegularCellData& triangulation = regularCellData[regularCellClass[cube_index]];
        const unsigned short* edge_flags = regularVertexData[cube_index];
        const uint triangle_count = triangulation.getTriangleCount();
        if (triangles == nullptr)
          return triangle_count;
        for (uint triangle = 0; triangle < triangle_count; ++triangle) {
          for (int vertex = 0; vertex < 3; ++vertex) {
            const unsigned char edge_flag = edge_flags[triangulation.vertexIndex[3 * triangle + vertex]] & 0xff;
            const int first_corner = edge_flag >> 4;
            const int second_corner = edge_flag & 0xf;
            int3 first = cell + make_int3(
              (first_corner >> 2) & 1, (first_corner >> 1) & 1, first_corner & 1);
            int3 second = cell + make_int3(
              (second_corner >> 2) & 1, (second_corner >> 1) & 1, second_corner & 1);
            float first_distance = distances[first_corner];
            float second_distance = distances[second_corner];
            const int axis = first.x != second.x ? 0 : first.y != second.y ? 1 : 2;
            if ((&first.x)[axis] > (&second.x)[axis]) {
              const int3 temporary_position = first;
              first = second;
              second = temporary_position;
              const float temporary_distance = first_distance;
              first_distance = second_distance;
              second_distance = temporary_distance;
            }
            const float interpolation = first_distance / (first_distance - second_distance);
            float3 position = virtualVoxelPosToWorld(container->virtual_voxel_size_, first);
            (&position.x)[axis] += interpolation * container->virtual_voxel_size_;
            triangles[triangle].vertices[vertex] = position;
          }
        }
        return triangle_count;
      }

      template <typename T>
      __global__ void countDirectionalMarchingCubesKernel(
        const VoxelContainer<T>* container,
        const uint voxel_count,
        const int3 owner_chunk,
        const float3 chunk_extents,
        uint* total_triangle_count) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= voxel_count)
          return;
        const uint compact_block = index / total_sdf_block_size;
        const uint local_index = index % total_sdf_block_size;
        const HashEntry entry = container->d_compactHashTable_[compact_block];
        if (!canonicalSpatialBlock(container, entry))
          return;
        const int3 cell = SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(delinearizeVoxelPos(local_index));
        const float3 center = virtualVoxelPosToWorld(container->virtual_voxel_size_, cell) +
                              make_float3(0.5f * container->virtual_voxel_size_);
        const float3 owner_center = make_float3(owner_chunk) * chunk_extents;
        const float3 owner_minimum = owner_center - 0.5f * chunk_extents;
        const float3 owner_maximum = owner_center + 0.5f * chunk_extents;
        if (center.x < owner_minimum.x || center.y < owner_minimum.y || center.z < owner_minimum.z ||
            center.x >= owner_maximum.x || center.y >= owner_maximum.y || center.z >= owner_maximum.z)
          return;
        atomicAdd(total_triangle_count, directionalCombinedCellTriangles(container, cell, nullptr));
      }

      template <typename T>
      __global__ void fillDirectionalMarchingCubesKernel(
        const VoxelContainer<T>* container,
        const uint voxel_count,
        const int3 owner_chunk,
        const float3 chunk_extents,
        uint* total_triangle_count,
        DeviceTriangle* triangles) {
        const uint index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= voxel_count)
          return;
        const uint compact_block = index / total_sdf_block_size;
        const uint local_index = index % total_sdf_block_size;
        const HashEntry entry = container->d_compactHashTable_[compact_block];
        if (!canonicalSpatialBlock(container, entry))
          return;
        const int3 cell = SDFBlockToVirtualVoxelPos(entry.pos) + make_int3(delinearizeVoxelPos(local_index));
        const float3 center = virtualVoxelPosToWorld(container->virtual_voxel_size_, cell) +
                              make_float3(0.5f * container->virtual_voxel_size_);
        const float3 owner_center = make_float3(owner_chunk) * chunk_extents;
        const float3 owner_minimum = owner_center - 0.5f * chunk_extents;
        const float3 owner_maximum = owner_center + 0.5f * chunk_extents;
        if (center.x < owner_minimum.x || center.y < owner_minimum.y || center.z < owner_minimum.z ||
            center.x >= owner_maximum.x || center.y >= owner_maximum.y || center.z >= owner_maximum.z)
          return;
        DeviceTriangle cell_triangles[10];
        const uint triangle_count = directionalCombinedCellTriangles(container, cell, cell_triangles);
        const uint offset = atomicAdd(total_triangle_count, triangle_count);
        for (uint triangle = 0; triangle < triangle_count; ++triangle)
          triangles[offset + triangle] = cell_triangles[triangle];
      }

    } // namespace

    template <typename T>
    bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateDirectionalDepthMap(
      const CUDAMatrixf& depth, const Camera& camera) {
      prepareDirectionalDepthMap(depth, camera);
      const bool integrated = integrateDirectionalDepthRows(depth, camera, 0, depth.rows());
      if (integrated)
        completeDirectionalDepthMap();
      return integrated;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::prepareDirectionalDepthMap(
      const CUDAMatrixf& depth, const Camera& camera) {
      CUDAMatrixf3 raw_normals(depth.rows(), depth.cols());
      directional_normals_.resize(depth.rows(), depth.cols());
      estimateNormalsKernel<<<camera.blocks(), camera.threads()>>>(
        depth.deviceInstance(), camera.deviceInstance(), raw_normals.deviceInstance());
      filterNormalsKernel<<<camera.blocks(), camera.threads()>>>(
        raw_normals.deviceInstance(), directional_normals_.deviceInstance());
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T>
    bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::allocateDirectionalDepthRows(
      const CUDAMatrixf& depth, const Camera& camera, const int row_begin, const int row_end) {
      const dim3 directional_threads(8, 8);
      const dim3 directional_blocks(
        (depth.cols() + directional_threads.x - 1) / directional_threads.x,
        (depth.rows() + directional_threads.y - 1) / directional_threads.y);

      int previous_free_blocks = getHeapHighFreeCount();
      while (true) {
        resetHashBucketMutex();
        CUDA_CHECK(cudaMemset(d_directionalAllocationFailed_, 0, sizeof(uint)));
        allocateDirectionalBlocksKernel<<<directional_blocks, directional_threads>>>(
          depth.deviceInstance(),
          directional_normals_.deviceInstance(),
          camera.deviceInstance(),
          d_instance_,
          d_directionalAllocationFailed_,
          row_begin,
          row_end);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint allocation_failed;
        CUDA_CHECK(cudaMemcpy(
          &allocation_failed, d_directionalAllocationFailed_, sizeof(uint), cudaMemcpyDeviceToHost));
        if (allocation_failed != 0)
          return false;
        const int free_blocks = getHeapHighFreeCount();
        if (free_blocks == previous_free_blocks)
          break;
        previous_free_blocks = free_blocks;
      }
      return true;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::fuseDirectionalDepthRows(
      const CUDAMatrixf& depth, const Camera& camera, const int row_begin, const int row_end) {
      flatAndReduceHashTable(camera);
      integrateDirectionalVisibleBlocksKernel<<<current_occupied_blocks_, total_sdf_block_size>>>(
        depth.deviceInstance(),
        directional_normals_.deviceInstance(),
        camera.deviceInstance(),
        d_instance_,
        row_begin,
        row_end);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    template <typename T>
    bool VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::integrateDirectionalDepthRows(
      const CUDAMatrixf& depth, const Camera& camera, const int row_begin, const int row_end) {
      if (!allocateDirectionalDepthRows(depth, camera, row_begin, row_end))
        return false;
      fuseDirectionalDepthRows(depth, camera, row_begin, row_end);
      return true;
    }

    template <typename T>
    void VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::completeDirectionalDepthMap() {
      ++num_integrated_frames_;
      updateFieldsDevice();
    }

    template <typename T>
    SurfaceVoxelData VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::directionalSurfaceVoxels(
      const int3& owner_chunk, const float3& chunk_extents) {
      flatAndReduceHashTable();
      if (current_occupied_blocks_ == 0)
        return SurfaceVoxelData();

      uint* candidate_count_device = nullptr;
      CUDA_CHECK(cudaMalloc(&candidate_count_device, sizeof(uint)));
      CUDA_CHECK(cudaMemset(candidate_count_device, 0, sizeof(uint)));
      countCrossingsKernel<<<current_occupied_blocks_, total_sdf_block_size>>>(
        d_instance_, owner_chunk, chunk_extents, candidate_count_device);
      CUDA_CHECK(cudaDeviceSynchronize());
      uint candidate_capacity = 0;
      CUDA_CHECK(cudaMemcpy(
        &candidate_capacity, candidate_count_device, sizeof(uint), cudaMemcpyDeviceToHost));
      if (candidate_capacity == 0) {
        CUDA_CHECK(cudaFree(candidate_count_device));
        return SurfaceVoxelData();
      }

      CandidateKey* candidate_keys = nullptr;
      CandidateValue* candidate_values = nullptr;
      CUDA_CHECK(cudaMalloc(&candidate_keys, sizeof(CandidateKey) * candidate_capacity));
      CUDA_CHECK(cudaMalloc(&candidate_values, sizeof(CandidateValue) * candidate_capacity));
      CUDA_CHECK(cudaMemset(candidate_count_device, 0, sizeof(uint)));
      fillCrossingsKernel<<<current_occupied_blocks_, total_sdf_block_size>>>(
        d_instance_, candidate_keys, candidate_values, candidate_count_device, owner_chunk, chunk_extents);
      CUDA_CHECK(cudaDeviceSynchronize());
      uint candidate_count = 0;
      CUDA_CHECK(cudaMemcpy(&candidate_count, candidate_count_device, sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(candidate_count_device));
      if (candidate_count == 0) {
        CUDA_CHECK(cudaFree(candidate_keys));
        CUDA_CHECK(cudaFree(candidate_values));
        return SurfaceVoxelData();
      }

      thrust::device_ptr<CandidateKey> candidate_key_pointer(candidate_keys);
      thrust::device_ptr<CandidateValue> candidate_value_pointer(candidate_values);
      thrust::sort_by_key(
        thrust::device,
        candidate_key_pointer,
        candidate_key_pointer + candidate_count,
        candidate_value_pointer,
        CandidateKeyLess());
      CandidateKey* cluster_keys = nullptr;
      CandidateValue* cluster_values = nullptr;
      CUDA_CHECK(cudaMalloc(&cluster_keys, sizeof(CandidateKey) * candidate_count));
      CUDA_CHECK(cudaMalloc(&cluster_values, sizeof(CandidateValue) * candidate_count));
      thrust::device_ptr<CandidateKey> cluster_key_pointer(cluster_keys);
      thrust::device_ptr<CandidateValue> cluster_value_pointer(cluster_values);
      const auto cluster_end = thrust::reduce_by_key(
        thrust::device,
        candidate_key_pointer,
        candidate_key_pointer + candidate_count,
        candidate_value_pointer,
        cluster_key_pointer,
        cluster_value_pointer,
        CandidateKeyEqual(),
        CandidateValueAdd());
      const uint cluster_count = cluster_end.first - cluster_key_pointer;
      CUDA_CHECK(cudaFree(candidate_keys));
      CUDA_CHECK(cudaFree(candidate_values));

      VoxelKey* voted_keys = nullptr;
      SurfaceValue* voted_values = nullptr;
      uint* voted_count_device = nullptr;
      CUDA_CHECK(cudaMalloc(&voted_keys, sizeof(VoxelKey) * cluster_count));
      CUDA_CHECK(cudaMalloc(&voted_values, sizeof(SurfaceValue) * cluster_count));
      CUDA_CHECK(cudaMalloc(&voted_count_device, sizeof(uint)));
      CUDA_CHECK(cudaMemset(voted_count_device, 0, sizeof(uint)));
      constexpr uint vote_threads = 256;
      voteClustersKernel<<<(cluster_count + vote_threads - 1) / vote_threads, vote_threads>>>(
        d_instance_, cluster_keys, cluster_values, cluster_count, voted_keys, voted_values, voted_count_device);
      CUDA_CHECK(cudaDeviceSynchronize());
      uint voted_count = 0;
      CUDA_CHECK(cudaMemcpy(&voted_count, voted_count_device, sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(voted_count_device));
      CUDA_CHECK(cudaFree(cluster_keys));
      CUDA_CHECK(cudaFree(cluster_values));
      if (voted_count == 0) {
        CUDA_CHECK(cudaFree(voted_keys));
        CUDA_CHECK(cudaFree(voted_values));
        return SurfaceVoxelData();
      }

      thrust::device_ptr<VoxelKey> voted_key_pointer(voted_keys);
      thrust::device_ptr<SurfaceValue> voted_value_pointer(voted_values);
      thrust::sort_by_key(
        thrust::device, voted_key_pointer, voted_key_pointer + voted_count, voted_value_pointer, VoxelKeyLess());
      VoxelKey* surface_keys = nullptr;
      SurfaceValue* surface_values = nullptr;
      CUDA_CHECK(cudaMalloc(&surface_keys, sizeof(VoxelKey) * voted_count));
      CUDA_CHECK(cudaMalloc(&surface_values, sizeof(SurfaceValue) * voted_count));
      thrust::device_ptr<VoxelKey> surface_key_pointer(surface_keys);
      thrust::device_ptr<SurfaceValue> surface_value_pointer(surface_values);
      const auto surface_end = thrust::reduce_by_key(
        thrust::device,
        voted_key_pointer,
        voted_key_pointer + voted_count,
        voted_value_pointer,
        surface_key_pointer,
        surface_value_pointer,
        thrust::equal_to<VoxelKey>(),
        MaximumConfidence());
      const uint surface_count = surface_end.first - surface_key_pointer;
      printf(
        "Directional surface stages: raw crossings %u, geometric candidates %u, normal clusters %u, "
        "voted clusters %u, surface voxels %u\n",
        candidate_capacity,
        candidate_count,
        cluster_count,
        voted_count,
        surface_count);
      CUDA_CHECK(cudaFree(voted_keys));
      CUDA_CHECK(cudaFree(voted_values));

      std::vector<VoxelKey> host_keys(surface_count);
      std::vector<SurfaceValue> host_values(surface_count);
      CUDA_CHECK(cudaMemcpy(
        host_keys.data(), surface_keys, sizeof(VoxelKey) * surface_count, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(
        host_values.data(), surface_values, sizeof(SurfaceValue) * surface_count, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(surface_keys));
      CUDA_CHECK(cudaFree(surface_values));

      SurfaceVoxelData result(surface_count);
      for (uint index = 0; index < surface_count; ++index) {
        result.indices.row(index) << host_keys[index].x, host_keys[index].y, host_keys[index].z;
        result.points.row(index) <<
          host_values[index].point.x, host_values[index].point.y, host_values[index].point.z;
        result.confidence(index) = host_values[index].confidence;
        result.normals.row(index) <<
          host_values[index].normal.x, host_values[index].normal.y, host_values[index].normal.z;
      }
      return result;
    }

    template <typename T>
    TriangleMeshData VoxelContainer<T, std::enable_if_t<is_voxel_derived<T>::value>>::directionalSurfaceMesh(
      const int3& owner_chunk, const float3& chunk_extents) {
      flatAndReduceHashTable();
      if (current_occupied_blocks_ == 0)
        return TriangleMeshData();
      const uint voxel_count = current_occupied_blocks_ * total_sdf_block_size;
      constexpr uint mesh_threads = 64;
      const uint mesh_blocks = (voxel_count + mesh_threads - 1) / mesh_threads;
      uint* count = nullptr;
      CUDA_CHECK(cudaMalloc(&count, sizeof(uint)));
      CUDA_CHECK(cudaMemset(count, 0, sizeof(uint)));
      countDirectionalMarchingCubesKernel<<<mesh_blocks, mesh_threads>>>(
        d_instance_, voxel_count, owner_chunk, chunk_extents, count);
      CUDA_CHECK(cudaDeviceSynchronize());
      uint triangle_count = 0;
      CUDA_CHECK(cudaMemcpy(&triangle_count, count, sizeof(uint), cudaMemcpyDeviceToHost));
      if (triangle_count == 0) {
        CUDA_CHECK(cudaFree(count));
        return TriangleMeshData();
      }
      DeviceTriangle* triangles = nullptr;
      CUDA_CHECK(cudaMalloc(&triangles, sizeof(DeviceTriangle) * triangle_count));
      CUDA_CHECK(cudaMemset(count, 0, sizeof(uint)));
      fillDirectionalMarchingCubesKernel<<<mesh_blocks, mesh_threads>>>(
        d_instance_,
        voxel_count,
        owner_chunk,
        chunk_extents,
        count,
        triangles);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<DeviceTriangle> host_triangles(triangle_count);
      CUDA_CHECK(cudaMemcpy(
        host_triangles.data(), triangles, sizeof(DeviceTriangle) * triangle_count, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(count));
      CUDA_CHECK(cudaFree(triangles));

      TriangleMeshData result(3 * triangle_count, triangle_count);
      for (uint triangle_index = 0; triangle_index < triangle_count; ++triangle_index) {
        for (int corner = 0; corner < 3; ++corner) {
          const uint vertex_index = 3 * triangle_index + corner;
          const float3 position = host_triangles[triangle_index].vertices[corner];
          result.vertices.row(vertex_index) << position.x, position.y, position.z;
          result.faces(triangle_index, corner) = vertex_index;
        }
      }
      return result;
    }

  } // namespace cugeoutils
} // namespace cupanutils
