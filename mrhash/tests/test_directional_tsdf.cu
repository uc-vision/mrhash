#include <gtest/gtest.h>

#include <sdf/camera.cuh>
#include <sdf/streamer.cuh>
#include <sdf/voxel_data_structures.cuh>

#include <array>
#include <map>

using namespace cupanutils::cugeoutils;

__global__ void allocateDirectionPairKernel(GeometricVoxelContainer* container, int* pointers) {
  const int3 position = make_int3(-2, 3, -4);
  container->allocDirectionalBlock(position, TSDFDirection::x_positive);
  container->allocDirectionalBlock(position, TSDFDirection::x_negative);
  pointers[0] = container->getHashEntry(position, TSDFDirection::x_positive).ptr;
  pointers[1] = container->getHashEntry(position, TSDFDirection::x_negative).ptr;
}

__global__ void writeDirectionalStatisticsKernel(
  GeometricVoxelContainer* container, const float sdf, const float weight) {
  const int3 position = make_int3(0);
  container->allocDirectionalBlock(position, TSDFDirection::z_negative);
  const HashEntry entry = container->getHashEntry(position, TSDFDirection::z_negative);
  Voxel& voxel = container->d_SDFBlocks_[entry.ptr];
  voxel.sdf = sdf;
  voxel.sum_squared = weight;
}

__global__ void readDirectionalStatisticsKernel(GeometricVoxelContainer* container, float* statistics) {
  const Voxel voxel = container->getVoxel(make_int3(0), TSDFDirection::z_negative);
  statistics[0] = voxel.sdf;
  statistics[1] = voxel.sum_squared;
}

__global__ void allocateSphereBlocksKernel(GeometricVoxelContainer* container) {
  for (int z = -1; z <= 0; ++z) {
    for (int y = -1; y <= 0; ++y) {
      for (int x = -1; x <= 0; ++x) {
        for (int direction = 0; direction < directional_tsdf_count; ++direction)
          container->allocDirectionalBlock(make_int3(x, y, z), static_cast<TSDFDirection>(direction));
      }
    }
  }
}

__global__ void exhaustHighHeapKernel(GeometricVoxelContainer* container, int* pointers) {
  pointers[threadIdx.x] = container->consumeHeapHigh();
}

__global__ void reclaimHighHeapKernel(GeometricVoxelContainer* container, const int* pointers) {
  if (pointers[threadIdx.x] >= 0)
    container->appendHeapHigh(pointers[threadIdx.x]);
}

__global__ void allocateOpposingBlocksKernel(GeometricVoxelContainer* container) {
  container->allocDirectionalBlock(make_int3(0, 0, 5), TSDFDirection::z_negative);
  container->allocDirectionalBlock(make_int3(0, 0, -5), TSDFDirection::z_positive);
}

__global__ void readOpposingBlockPointersKernel(GeometricVoxelContainer* container, int* pointers) {
  pointers[0] = container->getHashEntry(make_int3(0, 0, 5), TSDFDirection::z_negative).ptr;
  pointers[1] = container->getHashEntry(make_int3(0, 0, -5), TSDFDirection::z_positive).ptr;
}

__global__ void allocateVisibleBlocksKernel(GeometricVoxelContainer* container) {
  for (int y = -1; y <= 0; ++y) {
    for (int x = -1; x <= 0; ++x) {
      container->allocDirectionalBlock(make_int3(x, y, 5), TSDFDirection::z_negative);
      container->allocDirectionalBlock(make_int3(x, y, 5), TSDFDirection::x_positive);
    }
  }
}

__global__ void allocateDepthLayerBlocksKernel(GeometricVoxelContainer* container) {
  container->allocDirectionalBlock(make_int3(0, 0, 5), TSDFDirection::z_negative);
  container->allocDirectionalBlock(make_int3(0, 0, 9), TSDFDirection::z_negative);
}

__global__ void readDepthLayerBlockPointersKernel(GeometricVoxelContainer* container, int* pointers) {
  pointers[0] = container->getHashEntry(make_int3(0, 0, 5), TSDFDirection::z_negative).ptr;
  pointers[1] = container->getHashEntry(make_int3(0, 0, 9), TSDFDirection::z_negative).ptr;
}

__global__ void writeSphereKernel(GeometricVoxelContainer* container, const float radius) {
  const int3 block = make_int3(
    static_cast<int>(blockIdx.x & 1) - 1,
    static_cast<int>((blockIdx.x >> 1) & 1) - 1,
    static_cast<int>((blockIdx.x >> 2) & 1) - 1);
  const int3 position = SDFBlockToVirtualVoxelPos(block) + make_int3(delinearizeVoxelPos(threadIdx.x));
  const float3 world = virtualVoxelPosToWorld(container->virtual_voxel_size_, position);
  const float distance = length(world) - radius;
  for (int direction = 0; direction < directional_tsdf_count; ++direction) {
    const HashEntry entry = container->getHashEntry(block, static_cast<TSDFDirection>(direction));
    Voxel& voxel = container->d_SDFBlocks_[entry.ptr + threadIdx.x];
    voxel.sdf = distance;
    voxel.sum_squared = 1.f;
  }
}

TEST(DirectionalTSDF, DirectionIsPartOfSparseBlockIdentity) {
  GeometricVoxelContainer voxels(
    64, 64, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  voxels.voxel_extents_ = make_float3(1.f);
  voxels.updateFieldsDevice();
  voxels.resetHashBucketMutex();
  int* device_pointers = nullptr;
  ASSERT_EQ(cudaMalloc(&device_pointers, 2 * sizeof(int)), cudaSuccess);
  allocateDirectionPairKernel<<<1, 1>>>(voxels.d_instance_, device_pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int pointers[2];
  ASSERT_EQ(cudaMemcpy(pointers, device_pointers, sizeof(pointers), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_pointers), cudaSuccess);
  EXPECT_GE(pointers[0], 0);
  EXPECT_GE(pointers[1], 0);
  EXPECT_NE(pointers[0], pointers[1]);
}

TEST(DirectionalTSDF, ConcurrentHeapExhaustionPreservesReclamation) {
  constexpr int block_count = 8;
  constexpr int allocation_attempts = 64;
  GeometricVoxelContainer voxels(
    block_count, block_count, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  int* pointers = nullptr;
  ASSERT_EQ(cudaMalloc(&pointers, allocation_attempts * sizeof(int)), cudaSuccess);

  exhaustHighHeapKernel<<<1, allocation_attempts>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(voxels.getHeapHighFreeCount(), 0);

  reclaimHighHeapKernel<<<1, allocation_attempts>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(voxels.getHeapHighFreeCount(), block_count);
  ASSERT_EQ(cudaFree(pointers), cudaSuccess);
}

TEST(DirectionalTSDF, FrontFacingPlaneProducesSignedVoxelCrossings) {
  constexpr float voxel_size = 0.02f;
  GeometricVoxelContainer voxels(
    4096, 4096, 2.f, 4.f * voxel_size, 0.f, voxel_size, 1, 0, 0.f, false, true, false, "", "", "");
  voxels.voxel_extents_ = make_float3(1.f);
  voxels.updateFieldsDevice();

  Eigen::Matrix3f intrinsic;
  intrinsic << 100.f, 0.f, 15.5f, 0.f, 100.f, 15.5f, 0.f, 0.f, 1.f;
  Camera camera(CUDAMat3(intrinsic), 32, 32, 0.1f, 2.f, Pinhole);
  camera.setCamInWorld(Eigen::Matrix4f::Identity());
  CUDAMatrixf depth(32, 32);
  for (uint row = 0; row < depth.rows(); ++row) {
    for (uint column = 0; column < depth.cols(); ++column)
      depth.at(row, column) = 1.f;
  }
  depth.toDevice();
  voxels.integrateDirectionalDepthMap(depth, camera);
  const SurfaceVoxelData surface = voxels.directionalSurfaceVoxels(make_int3(0), make_float3(10.f));
  const TriangleMeshData mesh = voxels.directionalSurfaceMesh(make_int3(0), make_float3(10.f));

  ASSERT_EQ(surface.indices.rows(), 256);
  EXPECT_EQ(surface.indices.col(2).minCoeff(), 50);
  EXPECT_EQ(surface.indices.col(2).maxCoeff(), 50);
  EXPECT_NEAR(surface.normals.col(0).mean(), 0.f, 1e-5f);
  EXPECT_NEAR(surface.normals.col(1).mean(), 0.f, 1e-5f);
  EXPECT_NEAR(surface.normals.col(2).mean(), -1.f, 1e-5f);
  EXPECT_GT(surface.confidence.minCoeff(), 0.f);
  ASSERT_GT(mesh.faces.rows(), 0);
  EXPECT_NEAR(mesh.vertices.col(2).minCoeff(), 1.f, 1e-5f);
  EXPECT_NEAR(mesh.vertices.col(2).maxCoeff(), 1.f, 1e-5f);
  EXPECT_GE(mesh.faces.minCoeff(), 0);
  EXPECT_LT(mesh.faces.maxCoeff(), mesh.vertices.rows());
}

TEST(DirectionalTSDF, DirectMeshOfClosedFieldIsWatertight) {
  constexpr float voxel_size = 0.02f;
  GeometricVoxelContainer voxels(
    128, 128, 2.f, 4.f * voxel_size, 0.f, voxel_size, 1, 0, 0.f, false, true, false, "", "", "");
  voxels.voxel_extents_ = make_float3(1.f);
  voxels.updateFieldsDevice();
  int previous_free_blocks = voxels.getHeapHighFreeCount();
  while (true) {
    voxels.resetHashBucketMutex();
    allocateSphereBlocksKernel<<<1, 1>>>(voxels.d_instance_);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const int free_blocks = voxels.getHeapHighFreeCount();
    if (free_blocks == previous_free_blocks)
      break;
    previous_free_blocks = free_blocks;
  }
  writeSphereKernel<<<8, total_sdf_block_size>>>(voxels.d_instance_, 5.25f * voxel_size);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const TriangleMeshData mesh = voxels.directionalSurfaceMesh(make_int3(0), make_float3(10.f));
  ASSERT_GT(mesh.faces.rows(), 0);

  std::map<std::array<int, 3>, int> vertex_indices;
  std::map<std::array<int, 2>, int> edge_counts;
  for (Eigen::Index face_index = 0; face_index < mesh.faces.rows(); ++face_index) {
    std::array<int, 3> face;
    for (int corner = 0; corner < 3; ++corner) {
      const Eigen::Vector3f vertex = mesh.vertices.row(mesh.faces(face_index, corner));
      const std::array<int, 3> key = {
        static_cast<int>(std::lround(vertex.x() * 1e6f)),
        static_cast<int>(std::lround(vertex.y() * 1e6f)),
        static_cast<int>(std::lround(vertex.z() * 1e6f)),
      };
      face[corner] = vertex_indices.emplace(key, vertex_indices.size()).first->second;
    }
    ASSERT_NE(face[0], face[1]);
    ASSERT_NE(face[1], face[2]);
    ASSERT_NE(face[2], face[0]);
    for (int corner = 0; corner < 3; ++corner) {
      const int first = face[corner];
      const int second = face[(corner + 1) % 3];
      ++edge_counts[{min(first, second), max(first, second)}];
    }
  }
  for (const auto& [edge, count] : edge_counts)
    EXPECT_EQ(count, 2);
}

TEST(DirectionalTSDF, StreamingMergesDuplicateDirectionalBlocksWithoutLoss) {
  GeometricVoxelContainer voxels(
    64, 64, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  GeometricStreamer streamer(&voxels, false, "", "");
  streamer.create(Eigen::Vector3f::Ones(), 64, 1, false);

  voxels.resetHashBucketMutex();
  writeDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, 1.25f, 2.5f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  streamer.streamAllOut();

  voxels.resetHashBucketMutex();
  writeDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, 3.75f, 1.5f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  streamer.streamAllOut();

  const auto& grid = streamer.getGrid();
  ASSERT_EQ(grid.size(), 1);
  const ChunkDesc<Voxel>& chunk = *grid.begin()->second;
  ASSERT_EQ(chunk.getNElements(), 1);
  EXPECT_EQ(chunk.getSDFBlockDesc(0).direction, static_cast<signed char>(TSDFDirection::z_negative));
  EXPECT_FLOAT_EQ(chunk.getVoxel(0, 0).sdf, 5.f);
  EXPECT_FLOAT_EQ(chunk.getVoxel(0, 0).sum_squared, 4.f);

  streamer.streamInToGPU(Eigen::Vector3f::Zero(), 2.f);
  float* device_statistics = nullptr;
  ASSERT_EQ(cudaMalloc(&device_statistics, 2 * sizeof(float)), cudaSuccess);
  readDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, device_statistics);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float statistics[2];
  ASSERT_EQ(
    cudaMemcpy(statistics, device_statistics, sizeof(statistics), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_statistics), cudaSuccess);
  EXPECT_FLOAT_EQ(statistics[0], 5.f);
  EXPECT_FLOAT_EQ(statistics[1], 4.f);
}

TEST(DirectionalTSDF, ResidentBlockReceivesHostHistoryBeforeFusion) {
  GeometricVoxelContainer voxels(
    64, 64, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  GeometricStreamer streamer(&voxels, false, "", "");
  streamer.create(Eigen::Vector3f::Ones(), 64, 1, false);

  voxels.resetHashBucketMutex();
  writeDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, 1.25f, 2.5f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  streamer.streamAllOut();
  voxels.resetHashBucketMutex();
  writeDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, 3.75f, 1.5f);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  streamer.mergeResidentBlocksFromHost();
  EXPECT_EQ(streamer.getGrid().begin()->second->getNElements(), 0);
  float* device_statistics = nullptr;
  ASSERT_EQ(cudaMalloc(&device_statistics, 2 * sizeof(float)), cudaSuccess);
  readDirectionalStatisticsKernel<<<1, 1>>>(voxels.d_instance_, device_statistics);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  float statistics[2];
  ASSERT_EQ(
    cudaMemcpy(statistics, device_statistics, sizeof(statistics), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_statistics), cudaSuccess);
  EXPECT_FLOAT_EQ(statistics[0], 5.f);
  EXPECT_FLOAT_EQ(statistics[1], 4.f);
}

TEST(DirectionalTSDF, FrustumStreamingKeepsOnlyCurrentCameraWorkingSetResident) {
  GeometricVoxelContainer voxels(
    64, 64, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  GeometricStreamer streamer(&voxels, false, "", "");
  streamer.create(Eigen::Vector3f::Ones(), 64, 1, false);
  Eigen::Matrix3f intrinsic;
  intrinsic << 100.f, 0.f, 15.5f, 0.f, 100.f, 15.5f, 0.f, 0.f, 1.f;
  Camera camera(CUDAMat3(intrinsic), 32, 32, 0.1f, 2.f, Pinhole);
  Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  camera.setCamInWorld(pose.matrix());

  voxels.resetHashBucketMutex();
  allocateOpposingBlocksKernel<<<1, 1>>>(voxels.d_instance_);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int* pointers = nullptr;
  ASSERT_EQ(cudaMalloc(&pointers, 2 * sizeof(int)), cudaSuccess);

  streamer.stream(camera, pose);
  readOpposingBlockPointersKernel<<<1, 1>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int host_pointers[2];
  ASSERT_EQ(cudaMemcpy(host_pointers, pointers, sizeof(host_pointers), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_GE(host_pointers[0], 0);
  EXPECT_EQ(host_pointers[1], FREE_ENTRY);

  pose.linear() = Eigen::AngleAxisf(3.14159265358979323846f, Eigen::Vector3f::UnitY()).toRotationMatrix();
  camera.setCamInWorld(pose.matrix());
  streamer.stream(camera, pose);
  readOpposingBlockPointersKernel<<<1, 1>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(host_pointers, pointers, sizeof(host_pointers), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(host_pointers[0], FREE_ENTRY);
  EXPECT_GE(host_pointers[1], 0);
  ASSERT_EQ(cudaFree(pointers), cudaSuccess);
}

TEST(DirectionalTSDF, FrustumStreamInDoesNotExceedFreeBlockCapacity) {
  constexpr int block_count = 64;
  GeometricVoxelContainer voxels(
    block_count, block_count, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  GeometricStreamer streamer(&voxels, false, "", "");
  streamer.create(Eigen::Vector3f::Ones(), block_count, 1, false);
  Eigen::Matrix3f intrinsic;
  intrinsic << 100.f, 0.f, 15.5f, 0.f, 100.f, 15.5f, 0.f, 0.f, 1.f;
  Camera camera(CUDAMat3(intrinsic), 32, 32, 0.1f, 2.f, Pinhole);
  const Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  camera.setCamInWorld(pose.matrix());

  voxels.resetHashBucketMutex();
  allocateVisibleBlocksKernel<<<1, 1>>>(voxels.d_instance_);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  streamer.streamAllOut();

  int* pointers = nullptr;
  ASSERT_EQ(cudaMalloc(&pointers, 62 * sizeof(int)), cudaSuccess);
  exhaustHighHeapKernel<<<1, 62>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(voxels.getHeapHighFreeCount(), 2);

  streamer.streamInToGPU(camera, pose);
  EXPECT_EQ(voxels.getHeapHighFreeCount(), 0);
  EXPECT_FALSE(streamer.streamInDone());
  size_t remaining_blocks = 0;
  for (const auto& [position, chunk] : streamer.getGrid())
    remaining_blocks += chunk->getNElements();
  EXPECT_EQ(remaining_blocks, 6);
  ASSERT_EQ(cudaFree(pointers), cudaSuccess);
}

TEST(DirectionalTSDF, DepthBandStreamingRemovesOccludedFrustumLayers) {
  GeometricVoxelContainer voxels(
    64, 64, 2.f, 0.08f, 0.f, 0.02f, 1, 0, 0.f, false, true, false, "", "", "");
  GeometricStreamer streamer(&voxels, false, "", "");
  streamer.create(Eigen::Vector3f::Ones(), 64, 1, false);
  Eigen::Matrix3f intrinsic;
  intrinsic << 100.f, 0.f, 15.5f, 0.f, 100.f, 15.5f, 0.f, 0.f, 1.f;
  Camera camera(CUDAMat3(intrinsic), 32, 32, 0.1f, 2.f, Pinhole);
  const Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
  camera.setCamInWorld(pose.matrix());
  CUDAMatrixf depth(32, 32);
  for (uint row = 0; row < depth.rows(); ++row) {
    for (uint column = 0; column < depth.cols(); ++column)
      depth.at(row, column) = 0.86f;
  }
  depth.toDevice();

  voxels.resetHashBucketMutex();
  allocateDepthLayerBlocksKernel<<<1, 1>>>(voxels.d_instance_);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  streamer.stream(camera, pose, depth);

  int* pointers = nullptr;
  ASSERT_EQ(cudaMalloc(&pointers, 2 * sizeof(int)), cudaSuccess);
  readDepthLayerBlockPointersKernel<<<1, 1>>>(voxels.d_instance_, pointers);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  int host_pointers[2];
  ASSERT_EQ(cudaMemcpy(host_pointers, pointers, sizeof(host_pointers), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_GE(host_pointers[0], 0);
  EXPECT_EQ(host_pointers[1], FREE_ENTRY);
  ASSERT_EQ(cudaFree(pointers), cudaSuccess);
}
