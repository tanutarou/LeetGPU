#include <cuda_runtime.h>
#include <cfloat>

__global__ void find_max_stage1(const float *input, float *partial_max, int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = -FLT_MAX;

    for (int i = global_tid; i < N; i += stride)
    {
        local_max = fmaxf(local_max, input[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        partial_max[blockIdx.x] = sdata[0];
    }
}

__global__ void find_max_stage2(const float *partial_max, float *max_value, int num_blocks)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float local_max = -FLT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        local_max = fmaxf(local_max, partial_max[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        max_value[0] = sdata[0];
    }
}

__global__ void sum_exp_stage1(const float *input, const float *max_value, float *partial_sum, int N, float *output)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0;

    float m = max_value[0];
    for (int i = global_tid; i < N; i += stride)
    {
        output[i] = expf(input[i] - m);
        sum += output[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        partial_sum[blockIdx.x] = sdata[0];
    }
}

__global__ void sum_exp_stage2(const float *partial_sum, float *sum_value, int num_blocks)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;

    float sum = 0.0;
    for (int i = tid; i < num_blocks; i += blockDim.x)
    {
        sum += partial_sum[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        sum_value[0] = sdata[0];
    }
}

__global__ void softmax_kernel(const float *sum_value, float *output, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
    {
        output[idx] /= sum_value[0];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int N)
{
    int threads = 256;
    int blocks = std::min(1024, (N + threads - 1) / threads);

    float *partial_max, *max_value, *partial_sum, *sum_value;
    cudaMalloc(&partial_max, blocks * sizeof(float));
    cudaMalloc(&max_value, sizeof(float));
    cudaMalloc(&partial_sum, blocks * sizeof(float));
    cudaMalloc(&sum_value, sizeof(float));

    // find max value
    find_max_stage1<<<blocks, threads, threads * sizeof(float)>>>(input, partial_max, N);
    find_max_stage2<<<1, threads, threads * sizeof(float)>>>(partial_max, max_value, blocks);

    // find sum value
    sum_exp_stage1<<<blocks, threads, threads * sizeof(float)>>>(input, max_value, partial_sum, N, output);
    sum_exp_stage2<<<1, threads, threads * sizeof(float)>>>(partial_sum, sum_value, blocks);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    softmax_kernel<<<blocksPerGrid, threadsPerBlock>>>(sum_value, output, N);
    cudaDeviceSynchronize();

    cudaFree(partial_max);
    cudaFree(max_value);
    cudaFree(partial_sum);
    cudaFree(sum_value);
}
