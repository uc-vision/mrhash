#include "geowrapper.h"

#include "serializer.h"

#include <algorithm>
#include <functional>
#include <malloc.h>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace pygeowrapper {

  using cupanutils::cugeoutils::Eig2CUDA;
  using cupanutils::cugeoutils::DirectionalSurfaceData;
  using cupanutils::cugeoutils::SurfaceVoxelData;
  using cupanutils::cugeoutils::TriangleMeshData;

  namespace {

    struct Vector3fHash {
      size_t operator()(const Eigen::Vector3f& value) const {
        const char* data = reinterpret_cast<const char*>(value.data());
        return std::hash<std::string_view>()(std::string_view(data, sizeof(float) * 3));
      }
    };

    struct Vector3fEqual {
      bool operator()(const Eigen::Vector3f& left, const Eigen::Vector3f& right) const {
        return left == right;
      }
    };

  } // namespace

  DirectionalVoxelMap::DirectionalVoxelMap(
    const float voxel_size, const float minimum_depth, const float maximum_depth) :
    voxel_size_(voxel_size),
    minimum_depth_(minimum_depth),
    maximum_depth_(maximum_depth) {
    size_t free_bytes = 0;
    size_t total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t voxel_bytes = sizeof(cupanutils::cugeoutils::Voxel) * total_sdf_block_size;
    const size_t hash_bytes = sizeof(cupanutils::cugeoutils::HashEntry) * hash_bucket_size;
    const size_t streaming_bytes =
      2 * max_streaming_blocks * (voxel_bytes + sizeof(cupanutils::cugeoutils::SDFBlockDesc));
    num_blocks_ = static_cast<int>(
      (free_bytes - streaming_bytes) * voxel_map_SDFBlocks_ratio / (voxel_bytes + hash_bytes));
    voxels_ = std::make_unique<cupanutils::cugeoutils::GeometricVoxelContainer>(
      num_blocks_,
      num_blocks_,
      maximum_depth,
      4.f * voxel_size,
      0.f,
      voxel_size,
      1,
      0,
      0.f,
      false,
      true,
      false,
      "",
      "",
      "");
    streamer_ = std::make_unique<cupanutils::cugeoutils::GeometricStreamer>(voxels_.get(), false, "", "");
    streamer_->create(
      Eigen::Vector3f::Ones(),
      std::min(static_cast<uint>(num_blocks_), max_streaming_blocks),
      0,
      false);
    setCamera(1.f, 1.f, 0.f, 0.f, 1, 1, 0);
  }

  void DirectionalVoxelMap::setCamera(
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const int rows,
    const int columns,
    const int camera_model) {
    Eigen::Matrix3f intrinsic;
    intrinsic << focal_x, 0.f, center_x, 0.f, focal_y, center_y, 0.f, 0.f, 1.f;
    camera_ = std::make_unique<cupanutils::cugeoutils::Camera>(
      cupanutils::cugeoutils::CUDAMat3(intrinsic),
      rows,
      columns,
      minimum_depth_,
      maximum_depth_,
      static_cast<cupanutils::cugeoutils::CameraModel>(camera_model));
  }

  void DirectionalVoxelMap::setPose(
    const Eigen::Vector3f translation, const Eigen::Vector4f quaternion_xyzw) {
    pose_.setIdentity();
    pose_.linear() = Eigen::Quaternionf(
                       quaternion_xyzw.w(),
                       quaternion_xyzw.x(),
                       quaternion_xyzw.y(),
                       quaternion_xyzw.z())
                       .toRotationMatrix();
    pose_.translation() = translation;
    camera_->setCamInWorld(pose_.matrix());
  }

  void DirectionalVoxelMap::integrateDepthCUDA(CudaDepthImage input) {
    const size_t rows = input.shape(0);
    const size_t columns = input.shape(1);
    depth_.resize(rows, columns);
    CUDA_CHECK(cudaMemcpy(
      depth_.data<cupanutils::cugeoutils::Device>(),
      input.data(),
      sizeof(float) * rows * columns,
      cudaMemcpyDeviceToDevice));
    voxels_->prepareDirectionalDepthMap(depth_, *camera_);
    if (voxels_->allocateDirectionalDepthRows(depth_, *camera_, 0, rows)) {
      streamer_->mergeResidentBlocksFromHost();
      voxels_->fuseDirectionalDepthRows(depth_, *camera_, 0, rows);
      voxels_->completeDirectionalDepthMap();
      return;
    }

    const std::function<void(int, int)> integrate_rows = [&](const int row_begin, const int row_end) {
      streamer_->stream(*camera_, pose_, depth_, row_begin, row_end);
      if (streamer_->streamInDone() &&
          voxels_->allocateDirectionalDepthRows(depth_, *camera_, row_begin, row_end)) {
        streamer_->mergeResidentBlocksFromHost();
        voxels_->fuseDirectionalDepthRows(depth_, *camera_, row_begin, row_end);
        return;
      }
      if (row_end - row_begin == 1)
        throw std::runtime_error("Directional TSDF image row exceeds GPU block capacity");
      const int middle = row_begin + (row_end - row_begin) / 2;
      integrate_rows(row_begin, middle);
      integrate_rows(middle, row_end);
    };
    const int middle = rows / 2;
    integrate_rows(0, middle);
    integrate_rows(middle, rows);
    voxels_->completeDirectionalDepthMap();
  }

  DirectionalSurfaceData DirectionalVoxelMap::extractSurface() {
    streamer_->streamAllOut();
    streamer_->rechunk(0.125f);
    std::vector<Eigen::Vector3i> chunks;
    float neighborhood_radius;
    size_t maximum_neighborhood_blocks;
    do {
      chunks = streamer_->chunks();
      neighborhood_radius = 3.1f * streamer_->getChunkRadiusInMeter();
      maximum_neighborhood_blocks = 0;
      for (const Eigen::Vector3i& owner : chunks) {
        size_t neighborhood_blocks = 0;
        for (const auto& [data_chunk, contents] : streamer_->getGrid()) {
          if (streamer_->isChunkInSphere(
                data_chunk, streamer_->chunkToWorld(owner), neighborhood_radius))
            neighborhood_blocks += contents->getNElements();
        }
        maximum_neighborhood_blocks = std::max(maximum_neighborhood_blocks, neighborhood_blocks);
      }
      if (maximum_neighborhood_blocks > static_cast<size_t>(num_blocks_ / 4))
        streamer_->rechunk(0.5f);
    } while (maximum_neighborhood_blocks > static_cast<size_t>(num_blocks_ / 4));
    if (chunks.empty())
      return DirectionalSurfaceData{SurfaceVoxelData(), TriangleMeshData()};
    std::sort(chunks.begin(), chunks.end(), [](const Eigen::Vector3i& left, const Eigen::Vector3i& right) {
      if (left.x() != right.x())
        return left.x() < right.x();
      if (left.y() != right.y())
        return left.y() < right.y();
      return left.z() < right.z();
    });
    std::cout << "Directional extraction: " << chunks.size() << " chunks, at most "
              << maximum_neighborhood_blocks << " resident blocks per neighborhood" << std::endl;
    std::vector<SurfaceVoxelData> voxel_chunks;
    std::vector<TriangleMeshData> mesh_chunks;
    Eigen::Index total_voxels = 0;
    Eigen::Index total_faces = 0;
    std::vector<size_t> last_use(chunks.size(), chunks.size());
    for (size_t data_index = 0; data_index < chunks.size(); ++data_index) {
      for (size_t owner_index = 0; owner_index < chunks.size(); ++owner_index) {
        if (streamer_->isChunkInSphere(
              chunks[data_index], streamer_->chunkToWorld(chunks[owner_index]), neighborhood_radius))
          last_use[data_index] = owner_index;
      }
    }
    for (size_t owner_index = 0; owner_index < chunks.size(); ++owner_index) {
      const Eigen::Vector3i owner = chunks[owner_index];
      streamer_->streamInToGPU(streamer_->chunkToWorld(owner), neighborhood_radius);
      SurfaceVoxelData selected_voxels = voxels_->directionalSurfaceVoxels(
        Eig2CUDA(owner), Eig2CUDA(streamer_->getChunkExtents()));
      TriangleMeshData selected_mesh = voxels_->directionalSurfaceMesh(
        Eig2CUDA(owner), Eig2CUDA(streamer_->getChunkExtents()));
      total_voxels += selected_voxels.indices.rows();
      total_faces += selected_mesh.faces.rows();
      voxel_chunks.push_back(std::move(selected_voxels));
      mesh_chunks.push_back(std::move(selected_mesh));
      std::vector<Eigen::Vector3i> expired;
      for (size_t data_index = 0; data_index < chunks.size(); ++data_index) {
        if (last_use[data_index] == owner_index) {
          expired.push_back(chunks[data_index]);
          streamer_->eraseChunk(chunks[data_index]);
        }
      }
      if (!expired.empty())
        streamer_->discardChunks(expired);
      std::cout << "\rExtracting directional TSDF chunks: " << owner_index + 1 << '/' << chunks.size() << std::flush;
    }
    std::cout << std::endl;
    malloc_trim(0);

    DirectionalSurfaceData result{SurfaceVoxelData(total_voxels), TriangleMeshData()};
    Eigen::Index offset = 0;
    for (const SurfaceVoxelData& chunk : voxel_chunks) {
      result.voxels.indices.middleRows(offset, chunk.indices.rows()) = chunk.indices;
      result.voxels.points.middleRows(offset, chunk.points.rows()) = chunk.points;
      result.voxels.confidence.middleRows(offset, chunk.confidence.rows()) = chunk.confidence;
      result.voxels.normals.middleRows(offset, chunk.normals.rows()) = chunk.normals;
      offset += chunk.indices.rows();
    }

    std::unordered_map<Eigen::Vector3f, int, Vector3fHash, Vector3fEqual> vertex_indices;
    vertex_indices.reserve(total_faces);
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> faces;
    vertices.reserve(total_faces);
    faces.reserve(total_faces);
    for (const TriangleMeshData& chunk : mesh_chunks) {
      for (Eigen::Index face_index = 0; face_index < chunk.faces.rows(); ++face_index) {
        Eigen::Vector3i face;
        for (int corner = 0; corner < 3; ++corner) {
          const Eigen::Vector3f position = chunk.vertices.row(chunk.faces(face_index, corner));
          const auto [iterator, inserted] = vertex_indices.emplace(position, vertices.size());
          if (inserted)
            vertices.push_back(position);
          face(corner) = iterator->second;
        }
        faces.push_back(face);
      }
    }
    result.mesh = TriangleMeshData(vertices.size(), faces.size());
    for (size_t index = 0; index < vertices.size(); ++index)
      result.mesh.vertices.row(index) = vertices[index];
    for (size_t index = 0; index < faces.size(); ++index)
      result.mesh.faces.row(index) = faces[index];
    return result;
  }

  int DirectionalVoxelMap::getNumBlocks() const {
    return num_blocks_;
  }

  int DirectionalVoxelMap::getFreeBlocks() {
    return voxels_->getHeapHighFreeCount();
  }

  float DirectionalVoxelMap::getVoxelSize() const {
    return voxel_size_;
  }

  void DirectionalVoxelMap::serializeGrid(const std::string& filename) {
    streamer_->streamAllOut();
    cupanutils::cugeoutils::Serializer<cupanutils::cugeoutils::Voxel>::serialize(streamer_->grid_, filename);
  }

  void DirectionalVoxelMap::deserializeGrid(const std::string& filename) {
    cupanutils::cugeoutils::Serializer<cupanutils::cugeoutils::Voxel>::deserialize(streamer_->grid_, filename);
  }

} // namespace pygeowrapper
