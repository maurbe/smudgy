#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>

#include "../cpp/functions.h"  // your backend declarations

namespace py = pybind11;

// Example: wrapper for ngp_2d
py::tuple ngp_2d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     int periodic) // for ngp this is a dummy argument
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    ngp_2d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple ngp_3d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     int periodic) // for ngp this is a dummy argument
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    ngp_3d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_2d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    cic_2d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_3d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    cic_3d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}


py::tuple cic_2d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic,
                     py::array_t<float> pcellsizesHalf)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto pcs_buf = pcellsizesHalf.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    cic_2d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   pcs_ptr, fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_3d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic,
                     py::array_t<float> pcellsizesHalf)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto pcs_buf = pcellsizesHalf.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    cic_3d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   pcs_ptr, fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_2d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    tsc_2d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_3d_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    tsc_3d_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_2d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic,
                     py::array_t<float> pcellsizesHalf)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto pcs_buf = pcellsizesHalf.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    tsc_2d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   pcs_ptr, fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_3d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> extent,
                     int gridnum,
                     bool periodic,
                     py::array_t<float> pcellsizesHalf)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto pcs_buf = pcellsizesHalf.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    // Call backend function (direct loop)
    tsc_3d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, extent_min, extent_max, gridnum, periodic,
                   pcs_ptr, fields.data(), weights.data());

    // Return numpy arrays
    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple isotropic_2d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> extent,
    int gridnum,
    bool periodic,
    py::array_t<float> hsm,
    const std::string& kernel_name,
    const std::string& integration_method
) {
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto hsm_buf = hsm.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* hsm_ptr = static_cast<float*>(hsm_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    isotropic_kernel_deposition_2d_cpp(
        pos_ptr,
        q_ptr,
        hsm_ptr,
        N,
        num_fields,
        extent_min,
        extent_max,
        gridnum,
        periodic,
        kernel_name,
        integration_method,
        fields.data(),
        weights.data()
    );

    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple isotropic_3d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> extent,
    int gridnum,
    bool periodic,
    py::array_t<float> hsm,
    const std::string& kernel_name,
    const std::string& integration_method
) {
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto hsm_buf = hsm.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* hsm_ptr = static_cast<float*>(hsm_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    isotropic_kernel_deposition_3d_cpp(
        pos_ptr,
        q_ptr,
        hsm_ptr,
        N,
        num_fields,
        extent_min,
        extent_max,
        gridnum,
        periodic,
        kernel_name,
        integration_method,
        fields.data(),
        weights.data()
    );

    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple anisotropic_2d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> extent,
    int gridnum,
    bool periodic,
    py::array_t<float> hmat_eigvecs,
    py::array_t<float> hmat_eigvals,
    const std::string& kernel_name,
    const std::string& integration_method
) {
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto vec_buf = hmat_eigvecs.request();
    auto val_buf = hmat_eigvals.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* vec_ptr = static_cast<float*>(vec_buf.ptr);
    float* val_ptr = static_cast<float*>(val_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    anisotropic_kernel_deposition_2d_cpp(
        pos_ptr,
        q_ptr,
        vec_ptr,
        val_ptr,
        N,
        num_fields,
        extent_min,
        extent_max,
        gridnum,
        periodic,
        kernel_name,
        integration_method,
        fields.data(),
        weights.data()
    );

    py::array_t<float> fields_arr({gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple anisotropic_3d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> extent,
    int gridnum,
    bool periodic,
    py::array_t<float> hmat_eigvecs,
    py::array_t<float> hmat_eigvals,
    const std::string& kernel_name,
    const std::string& integration_method
) {
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto ext_buf = extent.request();
    auto vec_buf = hmat_eigvecs.request();
    auto val_buf = hmat_eigvals.request();

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    std::vector<float> fields(gridnum * gridnum * gridnum * num_fields, 0.0f);
    std::vector<float> weights(gridnum * gridnum * gridnum, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* vec_ptr = static_cast<float*>(vec_buf.ptr);
    float* val_ptr = static_cast<float*>(val_buf.ptr);

    float extent_min = static_cast<float*>(ext_buf.ptr)[0];
    float extent_max = static_cast<float*>(ext_buf.ptr)[1];

    anisotropic_kernel_deposition_3d_cpp(
        pos_ptr,
        q_ptr,
        vec_ptr,
        val_ptr,
        N,
        num_fields,
        extent_min,
        extent_max,
        gridnum,
        periodic,
        kernel_name,
        integration_method,
        fields.data(),
        weights.data()
    );

    py::array_t<float> fields_arr({gridnum, gridnum, gridnum, num_fields}, fields.data());
    py::array_t<float> weights_arr({gridnum, gridnum, gridnum}, weights.data());

    return py::make_tuple(fields_arr, weights_arr);
}


// -------------------------------------------------
PYBIND11_MODULE(functions, m) {
    m.doc() = "C++ deposition functions";

    m.def("ngp_2d", &ngp_2d_py, 
          "NGP deposition 2D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));

    m.def("ngp_3d", &ngp_3d_py, 
          "NGP deposition 3D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));

    m.def("cic_2d", &cic_2d_py, 
          "CIC deposition 2D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));
    
    m.def("cic_3d", &cic_3d_py,     
          "CIC deposition 3D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));

    m.def("cic_2d_adaptive", &cic_2d_adaptive_py,     
          "CIC adaptive deposition 2D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"),
          py::arg("pcellsizesHalf"));
    
    m.def("cic_3d_adaptive", &cic_3d_adaptive_py,     
          "CIC adaptive deposition 3D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"),
          py::arg("pcellsizesHalf"));

    m.def("tsc_2d", &tsc_2d_py,     
          "TSC deposition 2D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));

    m.def("tsc_3d", &tsc_3d_py,     
          "TSC deposition 3D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"));

    m.def("tsc_2d_adaptive", &tsc_2d_adaptive_py,     
          "TSC adaptive deposition 2D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"),
          py::arg("pcellsizesHalf"));

    m.def("tsc_3d_adaptive", &tsc_3d_adaptive_py,     
          "TSC adaptive deposition 3D",
          py::arg("pos"), 
          py::arg("quantities"), 
          py::arg("extent"),
          py::arg("gridnum"),
          py::arg("periodic"),
          py::arg("pcellsizesHalf"));

    m.def("isotropic_2d", &isotropic_2d_py,
        "Isotropic SPH kernel deposition 2D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("extent"),
        py::arg("gridnum"),
        py::arg("periodic"),
        py::arg("hsm"),
        py::arg("kernel_name"),
        py::arg("integration_method"));

    m.def("isotropic_3d", &isotropic_3d_py,
        "Isotropic SPH kernel deposition 3D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("extent"),
        py::arg("gridnum"),
        py::arg("periodic"),
        py::arg("hsm"),
        py::arg("kernel_name"),
        py::arg("integration_method"));

    m.def("anisotropic_2d", &anisotropic_2d_py,
        "Anisotropic SPH kernel deposition 2D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("extent"),
        py::arg("gridnum"),
        py::arg("periodic"),
        py::arg("hmat_eigvecs"),
        py::arg("hmat_eigvals"),
        py::arg("kernel_name"),
        py::arg("integration_method"));

    m.def("anisotropic_3d", &anisotropic_3d_py,
        "Anisotropic SPH kernel deposition 3D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("extent"),
        py::arg("gridnum"),
        py::arg("periodic"),
        py::arg("hmat_eigvecs"),
        py::arg("hmat_eigvals"),
        py::arg("kernel_name"),
        py::arg("integration_method"));
}
