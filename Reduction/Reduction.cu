#include <iostream>



__global__ void ConvergentReduction(float* in , float* out){
    int tx = threadIdx.x;
    int seg = blockIdx.x * blockDim.x * 2;

    for(int stride = blockDim.x; stride >= 1; stride /= 2){
        if( i < stride )
            in[i] += in[i +stride];
        __syncthreads();
        
    }
    if (threadIdx.x == 0)
        *out = in[threadIdx.x];


}


int main(){


    float* A_h, *R_h;

    A_h = (float * ) malloc(4096*sizeof(float));
    R_h = (float * ) malloc(1*sizeof(float));

    for(int i = 0 ; i < 4096; i++){
        A_h[i] = i;
    }
    R_h[0] = 0;


    float * A_d, *R_d;
    void*  t_A, *t_R;

    cudaMalloc(&t_A,4096*sizeof(float));
    cudaMalloc(&t_R,1*sizeof(float));

    A_d = static_cast<float*>(t_A);
    R_d = static_cast<float*>(t_R);

    cudaMemcpy(A_d, A_h,4096*sizeof(float),cudaMemcpyHostToDevice );
    cudaMemcpy(R_d, R_h,1*sizeof(float),cudaMemcpyHostToDevice );

    dim3 GridDim(8,1,1);
    dim3 BlockDim(256,1,1);

    ConvergentReduction<<<GridDim,BlockDim>>>(A_d,R_d);

    cudaMemcpy(R_h,R_d,1*sizeof(float),cudaMemcpyDeviceToHost );

    std::cout << R_h[0] << std::endl;

    cudaFree(A_d); cudaFree(R_d);
    free(A_h); free(R_h);






}