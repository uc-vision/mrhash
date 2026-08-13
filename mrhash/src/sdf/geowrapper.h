#pragma once

#include "camera.cuh"
#include "streamer.cuh"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nanobind/ndarray.h>

#include <memory>

namespace nb = nanobind;

namespace pygeowrapper {

  using CudaDepthImage = nb::ndarray<float, nb::shape<-1, -1>, nb::pytorch, nb::device::cuda, nb::c_contig>;

  class DirectionalVoxelMap {
  public:
    DirectionalVoxelMap(float voxel_size, float minimum_depth, float maximum_depth);

    void setCamera(float focal_x,
                   float focal_y,
                   float center_x,
                   float center_y,
                   int rows,
                   int columns,
                   int camera_model = 0);
    void setPose(Eigen::Vector3f translation, Eigen::Vector4f quaternion_xyzw);
    void integrateDepthCUDA(CudaDepthImage depth);
    cupanutils::cugeoutils::DirectionalSurfaceData extractSurface();

    int getNumBlocks() const;
    int getFreeBlocks();
    float getVoxelSize() const;

    void serializeGrid(const std::string& filename);
    void deserializeGrid(const std::string& filename);

  private:
    float voxel_size_;
    float minimum_depth_;
    float maximum_depth_;
    int num_blocks_;
    Eigen::Isometry3f pose_ = Eigen::Isometry3f::Identity();
    cupanutils::cugeoutils::CUDAMatrixf depth_;
    std::unique_ptr<cupanutils::cugeoutils::Camera> camera_;
    std::unique_ptr<cupanutils::cugeoutils::GeometricVoxelContainer> voxels_;
    std::unique_ptr<cupanutils::cugeoutils::GeometricStreamer> streamer_;
  };

} // namespace pygeowrapper
