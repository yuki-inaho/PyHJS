#include <string>
#include "pybind11/pybind11.h"
#include "ndarray_converter.h"
#include "hjs.h"
#include "frame.h"

namespace py = pybind11;

PYBIND11_MODULE(pyhjs, m)
{
    NDArrayConverter::init_numpy();
    py::class_<BinaryFrame>(m, "BinaryFrame")
        .def(
            py::init<const cv::Mat &>(),
            py::arg("binary_image"));
    py::class_<HamiltonJacobiSkeleton>(m, "PyHJS")
        .def(
            py::init<float, float, float>(),
            py::arg("gamma") = 2.5,
            py::arg("epsilon") = 1.0,
            py::arg("threshold_arc_angle_inscribed_circle") = 0)  /// default is desabled
        .def("compute", &HamiltonJacobiSkeleton::compute, py::arg("frame"), py::arg("enable_anisotropic_diffusion")=false)
        .def("set_parameters", &HamiltonJacobiSkeleton::setParameters)
        .def("get_skeleton_image", &HamiltonJacobiSkeleton::getSkeletonImage)
        .def("get_distance_transform_image", &HamiltonJacobiSkeleton::getDistanceTransformImage)
        .def("get_flux_image", &HamiltonJacobiSkeleton::getFluxImage);
}
