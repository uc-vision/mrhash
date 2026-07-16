#include <gtest/gtest.h>

#include <sdf/voxel_hash_utils.cuh>

using namespace cupanutils::cugeoutils;

__global__ void fuseOpposingDistancesKernel(Voxel* result) {
  Voxel empty;
  Voxel first;
  first.sdf            = -1.f;
  first.weight         = 1;
  Voxel second;
  second.sdf            = 3.f;
  second.weight         = 1;
  Voxel accumulated;
  combineVoxel(empty, first, 100, accumulated);
  combineVoxel(accumulated, second, 100, result[0]);
}

TEST(TwoSidedSurface, FusionAveragesSignedDistance) {
  Voxel* device_result;
  ASSERT_EQ(cudaMalloc((void**) &device_result, sizeof(Voxel)), cudaSuccess);
  fuseOpposingDistancesKernel<<<1, 1>>>(device_result);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  Voxel result;
  ASSERT_EQ(cudaMemcpy(&result, device_result, sizeof(Voxel), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_result), cudaSuccess);
  EXPECT_FLOAT_EQ(result.sdf, 1.f);
  EXPECT_EQ(result.weight, 2);
  EXPECT_EQ(sizeof(Voxel), 12);
}

TEST(TwoSidedSurface, FusionAveragesUnsignedDistance) {
  Voxel stored;
  stored.sdf = 2.f;
  stored.weight = 1;
  Voxel input;
  input.sdf = 4.f;
  input.weight = 1;
  Voxel merged;
  combineTwoSidedSurfaceVoxel(stored, input, 100, merged);

  EXPECT_FLOAT_EQ(merged.sdf, 3.f);
  EXPECT_EQ(merged.weight, 2);
  EXPECT_FLOAT_EQ(twoSidedSurfaceDistance(merged), 3.f);
  EXPECT_EQ(twoSidedSurfaceWeight(merged), 2);
}

struct QefResult {
  int rank;
  float3 point;
  float3 normal;
};

__global__ void solveQefExamplesKernel(QefResult* results) {
  float3 direction;
  TudfQef qef;
  addTudfPlane(qef, make_float3(0.f, 0.f, 1.f), make_float3(0.f));
  addTudfPlane(qef, make_float3(0.f, 0.f, -1.f), make_float3(1.f, 1.f, 0.f));
  addTudfPlane(qef, make_float3(0.f, 0.f, 1.f), make_float3(-1.f, 2.f, 0.f));
  results[0].rank = solveTudfQef(
    qef, make_float3(1.f, 2.f, 0.7f), results[0].point, results[0].normal, direction);

  qef = TudfQef();
  const float3 corner = make_float3(1.f, 2.f, 3.f);
  addTudfPlane(qef, make_float3(1.f, 0.f, 0.f), corner);
  addTudfPlane(qef, make_float3(0.f, 1.f, 0.f), corner);
  addTudfPlane(qef, make_float3(0.f, 0.f, 1.f), corner);
  results[1].rank = solveTudfQef(
    qef, make_float3(0.f), results[1].point, results[1].normal, direction);
}

TEST(TwoSidedSurface, QefRecoversPlaneAndCornerRanks) {
  QefResult* device_results;
  ASSERT_EQ(cudaMalloc((void**) &device_results, 2 * sizeof(QefResult)), cudaSuccess);
  solveQefExamplesKernel<<<1, 1>>>(device_results);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  QefResult results[2];
  ASSERT_EQ(cudaMemcpy(results, device_results, sizeof(results), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaFree(device_results), cudaSuccess);

  EXPECT_EQ(results[0].rank, 1);
  EXPECT_NEAR(results[0].point.x, 1.f, 1e-5f);
  EXPECT_NEAR(results[0].point.y, 2.f, 1e-5f);
  EXPECT_NEAR(results[0].point.z, 0.f, 1e-5f);
  EXPECT_NEAR(fabsf(results[0].normal.z), 1.f, 1e-5f);
  EXPECT_EQ(results[1].rank, 3);
  EXPECT_NEAR(results[1].point.x, 1.f, 1e-5f);
  EXPECT_NEAR(results[1].point.y, 2.f, 1e-5f);
  EXPECT_NEAR(results[1].point.z, 3.f, 1e-5f);
}

TEST(TwoSidedSurface, NormalDerivativeUsesAvailableTsdfNeighbors) {
  Voxel negative;
  negative.sdf = -2.f;
  negative.weight = 2;
  Voxel center;
  center.sdf = 1.f;
  center.weight = 2;
  Voxel positive;
  positive.sdf = 4.f;
  positive.weight = 2;
  EXPECT_FLOAT_EQ(surfaceDerivative(negative, center, positive, 2), 3.f);

  negative.weight = 0;
  EXPECT_FLOAT_EQ(surfaceDerivative(negative, center, positive, 2), 3.f);
  positive.weight = 0;
  EXPECT_FLOAT_EQ(surfaceDerivative(negative, center, positive, 2), 0.f);
}
