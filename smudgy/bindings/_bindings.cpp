#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>

#include "../core/cpp/_functions.h"  // backend declarations
#include "../core/cpp/_kernels.h"  // for kernel integral testing

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;


py::tuple _ngp_2d_cpp(py::array_t<float> positions,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    bool periodic, 
                    bool use_openmp, 
                    int omp_threads
                )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    ngp_2d_cpp(positions_pointer, quantities_pointer, num_particles, num_fields, boxsizes_pointer, gridnums_pointer,
               periodic, use_openmp, omp_threads, fields_pointer, weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _ngp_3d_cpp(py::array_t<float> positions,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    bool periodic,
                    bool use_openmp, 
                    int omp_threads
                )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    ngp_3d_cpp(positions_pointer, quantities_pointer, num_particles, num_fields, boxsizes_pointer, gridnums_pointer,
               periodic, use_openmp, omp_threads, fields_pointer, weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _cic_2d_cpp(py::array_t<float> positions,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    bool periodic,
                    bool use_openmp, 
                    int omp_threads
                )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];

    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    cic_2d_cpp(positions_pointer, quantities_pointer, num_particles, num_fields, boxsizes_pointer, gridnums_pointer,
        periodic, use_openmp, omp_threads, fields_pointer, weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _cic_3d_cpp(py::array_t<float> positions,
                    py::array_t<float> quantities,
                    py::array_t<float> boxsizes,
                    py::array_t<int> gridnums,
                    bool periodic,
                    bool use_openmp, 
                    int omp_threads
                )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];

    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    cic_3d_cpp(positions_pointer, quantities_pointer, num_particles, num_fields, boxsizes_pointer, gridnums_pointer, 
        periodic, use_openmp, omp_threads, fields_pointer, weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _cic_2d_adaptive_cpp(py::array_t<float> positions,
                     py::array_t<float> quantities,
                     py::array_t<float> smoothing_lengths,
                     py::array_t<float> boxsizes,
                     py::array_t<int> gridnums,
                     bool periodic,
                     bool use_openmp, 
                     int omp_threads
                    )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    cic_2d_adaptive_cpp(positions_pointer, 
                        quantities_pointer, 
                        smoothing_lengths_pointer, 
                        num_particles, 
                        num_fields, 
                        boxsizes_pointer, 
                        gridnums_pointer, 
                        periodic,
                        use_openmp, 
                        omp_threads, 
                        fields_pointer, 
                        weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _cic_3d_adaptive_cpp(py::array_t<float> positions,
                     py::array_t<float> quantities,
                     py::array_t<float> smoothing_lengths,
                     py::array_t<float> boxsizes,
                     py::array_t<int> gridnums,
                     bool periodic,
                     bool use_openmp, 
                     int omp_threads
                    )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    cic_3d_adaptive_cpp(positions_pointer, 
                        quantities_pointer, 
                        smoothing_lengths_pointer, 
                        num_particles, 
                        num_fields, 
                        boxsizes_pointer, 
                        gridnums_pointer, 
                        periodic,
                        use_openmp, 
                        omp_threads, 
                        fields_pointer, 
                        weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _tsc_2d_cpp(py::array_t<float> positions,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             bool periodic,
            bool use_openmp, 
            int omp_threads)
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    tsc_2d_cpp(positions_pointer, 
               quantities_pointer, 
               num_particles, 
               num_fields, 
               boxsizes_pointer, 
               gridnums_pointer, 
               periodic,
               use_openmp, 
               omp_threads, 
               fields_pointer, 
               weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _tsc_3d_cpp(py::array_t<float> positions,
             py::array_t<float> quantities,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             bool periodic,
            bool use_openmp, 
            int omp_threads)
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    tsc_3d_cpp(positions_pointer, quantities_pointer, num_particles, num_fields, boxsizes_pointer, gridnums_pointer, periodic,
               use_openmp, omp_threads, fields_pointer, weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _tsc_2d_adaptive_cpp(py::array_t<float> positions,
             py::array_t<float> quantities,
             py::array_t<float> smoothing_lengths,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             bool periodic,
            bool use_openmp, 
            int omp_threads)
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    tsc_2d_adaptive_cpp(positions_pointer, 
                        quantities_pointer, 
                        smoothing_lengths_pointer, 
                        num_particles, 
                        num_fields, 
                        boxsizes_pointer, 
                        gridnums_pointer, 
                        periodic,
                        use_openmp, 
                        omp_threads, 
                        fields_pointer, 
                        weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _tsc_3d_adaptive_cpp(py::array_t<float> positions,
             py::array_t<float> quantities,
             py::array_t<float> smoothing_lengths,
             py::array_t<float> boxsizes,
             py::array_t<int> gridnums,
             bool periodic,
             bool use_openmp, 
             int omp_threads
            )
{
    // Request buffer info
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    // Call backend function (direct loop)
    tsc_3d_adaptive_cpp(positions_pointer, 
                        quantities_pointer, 
                        smoothing_lengths_pointer, 
                        num_particles, 
                        num_fields, 
                        boxsizes_pointer, 
                        gridnums_pointer, 
                        periodic,
                        use_openmp, 
                        omp_threads, 
                        fields_pointer, 
                        weights_pointer);

    // Return numpy arrays
    return py::make_tuple(fields, weights);
}

py::tuple _separable_2d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_lengths,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp, 
    int omp_threads
)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    separable_kernel_deposition_2d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_lengths_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::tuple _separable_3d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_lengths,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    bool use_openmp, 
    int omp_threads
)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    separable_kernel_deposition_3d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_lengths_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::tuple _isotropic_2d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_lengths,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations_per_axis,
    bool use_openmp, 
    int omp_threads)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    isotropic_kernel_deposition_2d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_lengths_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        min_kernel_evaluations_per_axis,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::tuple _isotropic_3d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_lengths,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations_per_axis,
    bool use_openmp, int omp_threads)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_lengths_buffer = smoothing_lengths.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_lengths_pointer = static_cast<float*>(smoothing_lengths_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    isotropic_kernel_deposition_3d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_lengths_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        min_kernel_evaluations_per_axis,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::tuple _anisotropic_2d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_tensor_eigvecs,
    py::array_t<float> smoothing_tensor_eigvals,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations_per_axis,
    bool use_openmp, int omp_threads)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_tensor_eigvecs_buffer = smoothing_tensor_eigvecs.request();
    auto smoothing_tensor_eigvals_buffer = smoothing_tensor_eigvals.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];

    py::array_t<float> fields({gridnum_x, gridnum_y, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_tensor_eigvecs_pointer = static_cast<float*>(smoothing_tensor_eigvecs_buffer.ptr);
    float* smoothing_tensor_eigvals_pointer = static_cast<float*>(smoothing_tensor_eigvals_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    anisotropic_kernel_deposition_2d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_tensor_eigvecs_pointer,
        smoothing_tensor_eigvals_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        min_kernel_evaluations_per_axis,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::tuple _anisotropic_3d_cpp(
    py::array_t<float> positions,
    py::array_t<float> quantities,
    py::array_t<float> smoothing_tensor_eigvecs,
    py::array_t<float> smoothing_tensor_eigvals,
    py::array_t<float> boxsizes,
    py::array_t<int> gridnums,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    int min_kernel_evaluations_per_axis,
    bool use_openmp, int omp_threads)
{
    auto positions_buffer = positions.request();
    auto quantities_buffer = quantities.request();
    auto boxsizes_buffer = boxsizes.request();
    auto smoothing_tensor_eigvecs_buffer = smoothing_tensor_eigvecs.request();
    auto smoothing_tensor_eigvals_buffer = smoothing_tensor_eigvals.request();
    auto gridnums_buffer = gridnums.request();
    const int* gridnums_pointer = static_cast<int*>(gridnums_buffer.ptr);

    int num_particles = positions_buffer.shape[0];
    int num_fields = quantities_buffer.shape[1];
    int gridnum_x = gridnums_pointer[0];
    int gridnum_y = gridnums_pointer[1];
    int gridnum_z = gridnums_pointer[2];

    py::array_t<float> fields({gridnum_x, gridnum_y, gridnum_z, num_fields});
    py::array_t<float> weights({gridnum_x, gridnum_y, gridnum_z});
    float* fields_pointer = fields.mutable_data();
    float* weights_pointer = weights.mutable_data();
    std::fill_n(fields_pointer, gridnum_x * gridnum_y * gridnum_z * num_fields, 0.0f);
    std::fill_n(weights_pointer, gridnum_x * gridnum_y * gridnum_z, 0.0f);

    float* positions_pointer = static_cast<float*>(positions_buffer.ptr);
    float* quantities_pointer = static_cast<float*>(quantities_buffer.ptr);
    float* smoothing_tensor_eigvecs_pointer = static_cast<float*>(smoothing_tensor_eigvecs_buffer.ptr);
    float* smoothing_tensor_eigvals_pointer = static_cast<float*>(smoothing_tensor_eigvals_buffer.ptr);
    float* boxsizes_pointer = static_cast<float*>(boxsizes_buffer.ptr);

    anisotropic_kernel_deposition_3d_cpp(
        positions_pointer,
        quantities_pointer,
        smoothing_tensor_eigvecs_pointer,
        smoothing_tensor_eigvals_pointer,
        num_particles,
        num_fields,
        boxsizes_pointer,
        gridnums_pointer,
        periodic,
        kernel_name,
        integration_method,
        min_kernel_evaluations_per_axis,
        use_openmp,
        omp_threads,
        fields_pointer,
        weights_pointer
    );

    return py::make_tuple(fields, weights);
}

py::float_ _compute_total_integral_separable(
    const std::string& kernel_name,
    const int dim){
    return compute_total_integral_separable(kernel_name, dim);
}

py::float_ _compute_total_integral_spherical(
    const std::string& kernel_name,
    const int dim,
    const int min_kernel_evaluations_per_axis){
    return compute_total_integral_spherical(kernel_name, dim, min_kernel_evaluations_per_axis);
}

py::tuple _get_separable_kernel_values_1D(
    const std::string& kernel_name){
    auto [positions, values] = get_separable_kernel_values_1D(kernel_name);
    return py::make_tuple(positions, values);
}

py::tuple _get_spherical_kernel_values_1D(
    const std::string& kernel_name){
    auto [positions, values] = get_spherical_kernel_values_1D(kernel_name);
    return py::make_tuple(positions, values);
}

// -------------------------------------------------
PYBIND11_MODULE(_cpp_functions_ext, m) {
    m.doc() = "C++ deposition functions";


    #ifdef _OPENMP
        m.attr("has_openmp") = true;
    #else
        m.attr("has_openmp") = false;
    #endif

    
    m.def("openmp_thread_count", []() {
    #ifdef _OPENMP
        return omp_get_max_threads();
    #else
        return 1;
    #endif
    });

    m.def("get_separable_kernel_values_1D", &_get_separable_kernel_values_1D, 
        R"doc(Get the 1D domain and kernel value samples. Mostly used for plotting kernel shapes.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel function.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            A tuple containing:
            - positions: 1D array of positions where the kernel is evaluated.
            - values: 1D array of kernel values at the corresponding positions.
        )doc",
        py::arg("kernel_name"));

    m.def("get_spherical_kernel_values_1D", &_get_spherical_kernel_values_1D, 
        R"doc(Get the 1D domain and kernel value samples for a spherical kernel. Mostly used for plotting kernel shapes.
        
        Parameters
        ----------
        kernel_name : str
            The name of the spherical kernel function.
        
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            A tuple containing:
            - positions: 1D array of positions where the kernel is evaluated.
            - values: 1D array of kernel values at the corresponding positions.
        )doc",
        py::arg("kernel_name"));
    
    m.def("compute_total_integral_separable", &_compute_total_integral_separable, 
        R"doc(Compute the total integral of a separable kernel function.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel function.
        dim : int
            The dimensionality of the kernel.

        Returns
        -------
        float
            The total integral of the kernel function.
        )doc",
        py::arg("kernel_name"),
        py::arg("dim")
    );

    m.def("compute_total_integral_spherical", &_compute_total_integral_spherical, 
        R"doc(Compute the total integral of a spherical kernel function.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel function.
        dim : int
            The dimensionality of the kernel.
        min_kernel_evaluations_per_axis : int, optional
            Minimum number of evaluations per axis for numerical integration (if needed).

        Returns
        -------
        float
            The total integral of the kernel function.
        )doc",
        py::arg("kernel_name"),
        py::arg("dim"),        
        py::arg("min_kernel_evaluations_per_axis"));

    m.def("_ngp_2d_cpp", &_ngp_2d_cpp, 
        R"doc(
        Deposit particle quantities onto a 2D grid using NGP (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions, where ``N`` is the number of particles.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_ngp_3d_cpp", &_ngp_3d_cpp, 
        R"doc(
        Deposit particle quantities onto a 3D grid using NGP (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions, where ``N`` is the number of particles.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_cic_2d_cpp", &_cic_2d_cpp, 
        R"doc(
        Deposit particle quantities onto a 2D grid using CIC (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_cic_3d_cpp", &_cic_3d_cpp,     
        R"doc(
        Deposit particle quantities onto a 3D grid using CIC (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_cic_2d_adaptive_cpp", &_cic_2d_adaptive_cpp,     
        R"doc(
        Deposit particle quantities onto a 2D grid using adaptive CIC (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 2)
            Smoothing lengths per particle (adaptive support).
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_cic_3d_adaptive_cpp", &_cic_3d_adaptive_cpp,     
        R"doc(
        Deposit particle quantities onto a 3D grid using adaptive CIC (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 3)
            Smoothing lengths per particle (adaptive support).
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_tsc_2d_cpp", &_tsc_2d_cpp,     
        R"doc(
        Deposit particle quantities onto a 2D grid using TSC (C++ backend).

        Parameters
        ----------
        pos : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_tsc_3d_cpp", &_tsc_3d_cpp,     
        R"doc(
        Deposit particle quantities onto a 3D grid using TSC (C++ backend).

        Parameters
        ----------
        pos : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_tsc_2d_adaptive_cpp", &_tsc_2d_adaptive_cpp,     
        R"doc(
        Deposit particle quantities onto a 2D grid using adaptive TSC (C++ backend).

        Parameters
        ----------
        pos : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 2)
            Smoothing lengths per particle (adaptive support).
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_tsc_3d_adaptive_cpp", &_tsc_3d_adaptive_cpp,     
        R"doc(
        Deposit particle quantities onto a 3D grid using adaptive TSC (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 3)
            Smoothing lengths per particle (adaptive support).
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"), 
        py::arg("quantities"), 
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));
    
    m.def("_separable_2d_cpp", &_separable_2d_cpp,
        R"doc(
        Deposit particle quantities onto a 2D grid using a separable kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 2)
            Smoothing lengths per particle and axis.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name.
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_separable_3d_cpp", &_separable_3d_cpp,
        R"doc(
        Deposit particle quantities onto a 3D grid using a separable kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N, 3)
            Smoothing lengths per particle and axis.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name.
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_isotropic_2d_cpp", &_isotropic_2d_cpp,
        R"doc(
        Deposit particle quantities onto a 2D grid using an isotropic SPH kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N,)
            Smoothing lengths per particle.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name (e.g., ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``, ``"wendland_c2"``).
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        min_kernel_evaluations_per_axis : int
            Minimum kernel samples per axis and per particle.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("min_kernel_evaluations_per_axis"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_isotropic_3d_cpp", &_isotropic_3d_cpp,
        R"doc(
        Deposit particle quantities onto a 3D grid using an isotropic SPH kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_lengths : numpy.ndarray, shape (N,)
            Smoothing lengths per particle.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name (e.g., ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``, ``"wendland_c2"``).
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        min_kernel_evaluations_per_axis : int
            Minimum kernel samples per axis and per particle.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_lengths"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("min_kernel_evaluations_per_axis"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_anisotropic_2d_cpp", &_anisotropic_2d_cpp,
        R"doc(
        Deposit particle quantities onto a 2D grid using an anisotropic SPH kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 2)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_tensor_eigvecs : numpy.ndarray, shape (N, 2, 2)
            Eigenvectors of the smoothing tensor per particle.
        smoothing_tensor_eigvals : numpy.ndarray, shape (N, 2)
            Eigenvalues of the smoothing tensor per particle.
        boxsizes : array_like, shape (2,)
            Domain size per axis.
        gridnums : array_like, shape (2,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name (e.g., ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``, ``"wendland_c2"``).
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        min_kernel_evaluations_per_axis : int
            Minimum kernel samples per axis and per particle.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_tensor_eigvecs"),
        py::arg("smoothing_tensor_eigvals"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("min_kernel_evaluations_per_axis"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));

    m.def("_anisotropic_3d_cpp", &_anisotropic_3d_cpp,
        R"doc(
        Deposit particle quantities onto a 3D grid using an anisotropic SPH kernel (C++ backend).

        Parameters
        ----------
        positions : numpy.ndarray, shape (N, 3)
            Particle positions.
        quantities : numpy.ndarray, shape (N, F)
            Per-particle fields to deposit.
        smoothing_tensor_eigvecs : numpy.ndarray, shape (N, 3, 3)
            Eigenvectors of the smoothing tensor per particle.
        smoothing_tensor_eigvals : numpy.ndarray, shape (N, 3)
            Eigenvalues of the smoothing tensor per particle.
        boxsizes : array_like, shape (3,)
            Domain size per axis.
        gridnums : array_like, shape (3,)
            Number of grid cells per axis.
        periodic : bool
            Periodic boundaries.
        kernel_name : str
            Kernel name (e.g., ``"gaussian"``, ``"cubic_spline"``, ``"quintic_spline"``, ``"wendland_c2"``).
        integration_method : str
            Integration method (``"midpoint"``, ``"trapezoidal"``, or ``"simpson"``).
        min_kernel_evaluations_per_axis : int
            Minimum kernel samples per axis and per particle.
        use_openmp : bool
            Enable OpenMP parallelism.
        omp_threads : int
            Number of OpenMP threads (0 uses the default).

        Returns
        -------
        fields : numpy.ndarray, shape (Gx, Gy, Gz, F)
            Deposited field values.
        weights : numpy.ndarray, shape (Gx, Gy, Gz)
            Weight sum per cell.
        )doc",
        py::arg("positions"),
        py::arg("quantities"),
        py::arg("smoothing_tensor_eigvecs"),
        py::arg("smoothing_tensor_eigvals"),
        py::arg("boxsizes"),
        py::arg("gridnums"),
        py::arg("periodic"),
        py::arg("kernel_name"),
        py::arg("integration_method"),
        py::arg("min_kernel_evaluations_per_axis"),
        py::arg("use_openmp"),
        py::arg("omp_threads"));
}
