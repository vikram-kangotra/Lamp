pub const ARITHMETIC_SRC: &str = r#"
extern "C" __global__ void neg(const float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = -a[i];
    }
}

extern "C" __global__ void add(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

extern "C" __global__ void add_scalar(const float *a, const float b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b;
    }
}

extern "C" __global__ void sub(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

extern "C" __global__ void sub_scalar(const float *a, const float b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b;
    }
}

extern "C" __global__ void mul(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

extern "C" __global__ void mul_scalar(const float *a, const float b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b;
    }
}

extern "C" __global__ void div(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / b[i];
    }
}

extern "C" __global__ void div_scalar(const float *a, const float b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] / b;
    }
}

extern "C" __global__ void sum(float *a, float *c, int size) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? a[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        c[blockIdx.x] = sdata[0];
    }
}

extern "C" __global__ void transpose(const float *a, float *c, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        c[j * m + i] = a[i * n + j];
    }
}
"#;

pub const UTILS_SRC: &str = r#"
extern "C" __global__ void get_flat_item(const float *data, float *result, int offset) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        result[0] = data[offset];
    }
}
"#;
