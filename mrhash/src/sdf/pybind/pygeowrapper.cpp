#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <geowrapper.h>
#include <serializer.h>

using cupanutils::cugeoutils::DirectionalSurfaceData;
using cupanutils::cugeoutils::SurfaceVoxelData;
using cupanutils::cugeoutils::TriangleMeshData;
using pygeowrapper::DirectionalVoxelMap;

NB_MODULE(pygeowrapper, module) {
  nb::class_<SurfaceVoxelData>(module, "SurfaceVoxelData")
    .def_ro("indices", &SurfaceVoxelData::indices)
    .def_ro("points", &SurfaceVoxelData::points)
    .def_ro("confidence", &SurfaceVoxelData::confidence)
    .def_ro("normals", &SurfaceVoxelData::normals);

  nb::class_<TriangleMeshData>(module, "TriangleMeshData")
    .def_ro("vertices", &TriangleMeshData::vertices)
    .def_ro("faces", &TriangleMeshData::faces);

  nb::class_<DirectionalSurfaceData>(module, "DirectionalSurfaceData")
    .def_ro("voxels", &DirectionalSurfaceData::voxels)
    .def_ro("mesh", &DirectionalSurfaceData::mesh);

  nb::class_<DirectionalVoxelMap>(module, "DirectionalVoxelMap")
    .def(nb::init<float, float, float>(),
         nb::arg("voxel_size"),
         nb::arg("minimum_depth"),
         nb::arg("maximum_depth"))
    .def("set_camera",
         &DirectionalVoxelMap::setCamera,
         nb::arg("focal_x"),
         nb::arg("focal_y"),
         nb::arg("center_x"),
         nb::arg("center_y"),
         nb::arg("rows"),
         nb::arg("columns"),
         nb::arg("camera_model") = 0)
    .def("set_pose", &DirectionalVoxelMap::setPose, nb::arg("translation"), nb::arg("quaternion_xyzw"))
    .def("integrate_depth_cuda", &DirectionalVoxelMap::integrateDepthCUDA, nb::arg("depth"))
    .def("extract_surface", &DirectionalVoxelMap::extractSurface)
    .def_prop_ro("num_blocks", &DirectionalVoxelMap::getNumBlocks)
    .def_prop_ro("free_blocks", &DirectionalVoxelMap::getFreeBlocks)
    .def_prop_ro("voxel_size", &DirectionalVoxelMap::getVoxelSize)
    .def("serialize_grid",
         &DirectionalVoxelMap::serializeGrid,
         nb::arg("filename") = cupanutils::cugeoutils::default_serializer_filename)
    .def("deserialize_grid",
         &DirectionalVoxelMap::deserializeGrid,
         nb::arg("filename") = cupanutils::cugeoutils::default_serializer_filename);
}
