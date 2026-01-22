#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "../cpp/functions.h"  // your backend declarations

namespace py = pybind11;

// Example: wrapper for ngp_2d
py::tuple ngp_2d_py(py::array_t<float> pos,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                    bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    ngp_2d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr,
               periodic_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple ngp_3d_py(py::array_t<float> pos,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                    bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    ngp_3d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr,
               periodic_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_2d_py(py::array_t<float> pos,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                    bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    cic_2d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
               use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_3d_py(py::array_t<float> pos,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                    bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    cic_3d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
               use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}


py::tuple cic_2d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> boxsizes,
                     py::array_t<int> gridnums,
                     py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                     py::array_t<float> pcellsizesHalf,
                     bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto pcs_buf = pcellsizesHalf.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();

    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    cic_2d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
                        pcs_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple cic_3d_adaptive_py(py::array_t<float> pos,
                     py::array_t<float> quantities,
                     py::array_t<float> boxsizes,
                     py::array_t<int> gridnums,
                     py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
                     py::array_t<float> pcellsizesHalf,
                     bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto pcs_buf = pcellsizesHalf.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    cic_3d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
                        pcs_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_2d_py(py::array_t<float> pos,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
             bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    tsc_2d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
               use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_3d_py(py::array_t<float> pos,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
             bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    tsc_3d_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
               use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_2d_adaptive_py(py::array_t<float> pos,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
             py::array_t<float> pcellsizesHalf,
             bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto pcs_buf = pcellsizesHalf.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    tsc_2d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
                        pcs_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple tsc_3d_adaptive_py(py::array_t<float> pos,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
             py::array_t<float> pcellsizesHalf,
             bool use_openmp = true, int omp_threads = 0)
{
    // Request buffer info
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto pcs_buf = pcellsizesHalf.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];

    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* pcs_ptr = static_cast<float*>(pcs_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);


    // Call backend function (direct loop)
    tsc_3d_adaptive_cpp(pos_ptr, q_ptr, N, num_fields, boxsizes_ptr, grid_ptr, periodic_ptr,
                        pcs_ptr, use_openmp, omp_threads, fields_ptr, weights_ptr);

    // Return numpy arrays
    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple isotropic_2d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
    py::array_t<float> hsm,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp = true, int omp_threads = 0)
{
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto hsm_buf = hsm.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* hsm_ptr = static_cast<float*>(hsm_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);

    isotropic_kernel_deposition_2d_cpp(
        pos_ptr,
        q_ptr,
        hsm_ptr,
        N,
        num_fields,
        boxsizes_ptr,
        grid_ptr,
        periodic_ptr,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_ptr,
        weights_ptr
    );

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple isotropic_3d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
    py::array_t<float> hsm,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp = true, int omp_threads = 0)
{
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto hsm_buf = hsm.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    
    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* hsm_ptr = static_cast<float*>(hsm_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    isotropic_kernel_deposition_3d_cpp(
        pos_ptr,
        q_ptr,
        hsm_ptr,
        N,
        num_fields,
        boxsizes_ptr,
        grid_ptr,
        periodic_ptr,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_ptr,
        weights_ptr
    );

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple anisotropic_2d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
    py::array_t<float> hmat_eigvecs,
    py::array_t<float> hmat_eigvals,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp = true, int omp_threads = 0)
{
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto vec_buf = hmat_eigvecs.request();
    auto val_buf = hmat_eigvals.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    
    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* vec_ptr = static_cast<float*>(vec_buf.ptr);
    float* val_ptr = static_cast<float*>(val_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y, 0.0f);

    anisotropic_kernel_deposition_2d_cpp(
        pos_ptr,
        q_ptr,
        vec_ptr,
        val_ptr,
        N,
        num_fields,
        boxsizes_ptr,
        grid_ptr,
        periodic_ptr,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_ptr,
        weights_ptr
    );

    return py::make_tuple(fields_arr, weights_arr);
}

py::tuple anisotropic_3d_py(
    py::array_t<float> pos,
    py::array_t<float> quantities,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    py::array_t<bool, py::array::c_style | py::array::forcecast> periodic,
    py::array_t<float> hmat_eigvecs,
    py::array_t<float> hmat_eigvals,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp = true, int omp_threads = 0)
{
    auto pos_buf = pos.request();
    auto q_buf = quantities.request();
    auto box_buf = boxsizes.request();
    auto vec_buf = hmat_eigvecs.request();
    auto val_buf = hmat_eigvals.request();
    auto grid_buf = gridnums.request();
    auto periodic_buf = periodic.request();
    
    float* pos_ptr = static_cast<float*>(pos_buf.ptr);
    float* q_ptr   = static_cast<float*>(q_buf.ptr);
    float* vec_ptr = static_cast<float*>(vec_buf.ptr);
    float* val_ptr = static_cast<float*>(val_buf.ptr);
    float* boxsizes_ptr = static_cast<float*>(box_buf.ptr);
    const int* grid_ptr = static_cast<int*>(grid_buf.ptr);
    const bool* periodic_ptr = static_cast<bool*>(periodic_buf.ptr);

    int N = pos_buf.shape[0];
    int num_fields = q_buf.shape[1];
    int gridnum_x = grid_ptr[0];
    int gridnum_y = grid_ptr[1];
    int gridnum_z = grid_ptr[2];

    py::array_t<float> fields_arr({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights_arr({gridnum_x, gridnum_y, gridnum_z});
    float* fields_ptr = fields_arr.mutable_data();
    float* weights_ptr = weights_arr.mutable_data();
    std::fill_n(fields_ptr, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_ptr, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    anisotropic_kernel_deposition_3d_cpp(
        pos_ptr,
        q_ptr,
        vec_ptr,
        val_ptr,
        N,
        num_fields,
        boxsizes_ptr,
        grid_ptr,
        periodic_ptr,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_ptr,
        weights_ptr
    );

    return py::make_tuple(fields_arr, weights_arr);
}


// -------------------------------------------------
PYBIND11_MODULE(functions, m) {
    m.doc() = "C++ deposition functions";

    m.def("ngp_2d", &ngp_2d_py, 
        "NGP deposition 2D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("ngp_3d", &ngp_3d_py, 
        "NGP deposition 3D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("cic_2d", &cic_2d_py, 
        "CIC deposition 2D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("cic_3d", &cic_3d_py,     
        "CIC deposition 3D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("cic_2d_adaptive", &cic_2d_adaptive_py,     
        "CIC adaptive deposition 2D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("pcellsizesHalf"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("cic_3d_adaptive", &cic_3d_adaptive_py,     
        "CIC adaptive deposition 3D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("pcellsizesHalf"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("tsc_2d", &tsc_2d_py,     
        "TSC deposition 2D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("tsc_3d", &tsc_3d_py,     
        "TSC deposition 3D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("tsc_2d_adaptive", &tsc_2d_adaptive_py,     
        "TSC adaptive deposition 2D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("pcellsizesHalf"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("tsc_3d_adaptive", &tsc_3d_adaptive_py,     
        "TSC adaptive deposition 3D",
        py::arg("pos"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("pcellsizesHalf"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("isotropic_2d", &isotropic_2d_py,
        "Isotropic SPH kernel deposition 2D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("hsm"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("isotropic_3d", &isotropic_3d_py,
        "Isotropic SPH kernel deposition 3D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("hsm"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("anisotropic_2d", &anisotropic_2d_py,
        "Anisotropic SPH kernel deposition 2D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("hmat_eigvecs"),
        py::arg("hmat_eigvals"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);

    m.def("anisotropic_3d", &anisotropic_3d_py,
        "Anisotropic SPH kernel deposition 3D",
        py::arg("pos"),
        py::arg("quantities"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("hmat_eigvecs"),
        py::arg("hmat_eigvals"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp") = true,
        py::arg("omp_threads") = 0);
}
