#include "cuda_utils.cuh"

__device__ void from_sparse_f32_fwd(
    const size_t numel,
    const float * values,
    const size_t *values_info,
    const float *indeces,
    const size_t *indeces_info,
    float* output,
    const size_t *output_info,
    const size_t num_dims,
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *values_dims = values_info;
    const size_t *values_strides = values_info + 1;
    const size_t *indeces_dims = indeces_info;
    const size_t *indeces_strides = indeces_info + 2;
    const size_t *output_dims = output_info;
    const size_t *output_strides = output_info + num_dims;

    unsigned int index = 0;
    for (unsigned int d = num_dims - 1; d >= 0; d--) {
        index += get_strided_index(i * num_dims + d, 2, indeces_dims, indeces_strides) * output_dims[d];
    }
    float value = values[get_strided_index(i, 1, values_dims, values_strides)];
    output[get_strided_index(index, num_dims, output_dims, output_strides)] = value;
}

__device__ void sum_to_bwd(
    const size_t numel,
    float * values,
    const size_t *values_info,
    const float *indeces,
    const size_t *indeces_info,
    const float* output,
    const size_t *output_info,
    const size_t num_dims,
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    const size_t *values_dims = values_info;
    const size_t *values_strides = values_info + 1;
    const size_t *indeces_dims = indeces_info;
    const size_t *indeces_strides = indeces_info + 2;
    const size_t *output_dims = output_info;
    const size_t *output_strides = output_info + num_dims;

    unsigned int index = 0;
    for (unsigned int d = num_dims - 1; d >= 0; d--) {
        index += get_strided_index(i * num_dims + d, 2, indeces_dims, indeces_strides) * output_dims[d];
    }
    float value = output[get_strided_index(index, num_dims, output_dims, output_strides)];
    values[get_strided_index(i, 1, values_dims, values_strides)] = value;
}