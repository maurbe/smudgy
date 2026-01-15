#pragma once
#include <vector>
#include <string>

void ngp_2d_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    float* fields,               // (gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum)
);

void ngp_3d_cpp(
    const float* pos,            // (N, 3)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    float* fields,               // (gridnum, gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum, gridnum)
);

void cic_2d_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,               // (gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum)
);  

void cic_3d_cpp(
    const float* pos,            // (N, 3)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,               // (gridnum, gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum, gridnum)
);

void cic_2d_adaptive_cpp(
    const float* pos,            // (N, 2)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,               // (gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum)
);

void cic_3d_adaptive_cpp(
    const float* pos,            // (N, 3)
    const float* quantities,     // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,               // (gridnum, gridnum, gridnum, num_fields)
    float* weights               // (gridnum, gridnum, gridnum)
);

void tsc_2d_cpp(
    const float* pos,        // (N,2)
    const float* quantities, // (N, num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,           // (gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum)
);

void tsc_3d_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    float* fields,           // (gridnum, gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum, gridnum)
);

void tsc_2d_adaptive_cpp(
    const float* pos,        // (N,2)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,           // (gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum)
);

void tsc_3d_adaptive_cpp(
    const float* pos,        // (N,3)
    const float* quantities, // (N,num_fields)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const float* pcellsizesHalf, // (N)
    float* fields,           // (gridnum, gridnum, gridnum, num_fields)
    float* weights           // (gridnum, gridnum, gridnum)
);

void isotropic_kernel_deposition_2d_cpp(
    const float* pos,          // (N, 2)
    const float* quantities,   // (N, num_fields)
    const float* hsm,          // (N)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,             // (gridnum, gridnum, num_fields)
    float* weights             // (gridnum, gridnum)
);

void isotropic_kernel_deposition_3d_cpp(
    const float* pos,          // (N, 3)
    const float* quantities,   // (N, num_fields)
    const float* hsm,          // (N)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,             // (gridnum, gridnum, gridnum, num_fields)
    float* weights             // (gridnum, gridnum, gridnum)
);

void anisotropic_kernel_deposition_2d_cpp(
    const float* pos,                // (N, 2)
    const float* quantities,         // (N, num_fields)
    const float* hmat_eigvecs,       // (N, 2, 2)
    const float* hmat_eigvals,       // (N, 2)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,                   // (gridnum, gridnum, num_fields)
    float* weights                   // (gridnum, gridnum)
);

void anisotropic_kernel_deposition_3d_cpp(
    const float* pos,                // (N, 3)
    const float* quantities,         // (N, num_fields)
    const float* hmat_eigvecs,       // (N, 3, 3)
    const float* hmat_eigvals,       // (N, 3)
    int N,
    int num_fields,
    float extent_min,
    float extent_max,
    int gridnum,
    bool periodic,
    const std::string& kernel_name,
    const std::string& integration_method,
    float* fields,                   // (gridnum, gridnum, gridnum, num_fields)
    float* weights                   // (gridnum, gridnum, gridnum)
);