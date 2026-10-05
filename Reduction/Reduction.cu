#include <iostream>
#include <cuda/atomic>

#define SIZE 512

__global__ void ConvergentReduction(unsigned int* in , unsigned int* out , int length){
    int seg = 2*blockDim.x * blockIdx.x;
    int i = seg + threadIdx.x;

    __shared__ unsigned int in_s[SIZE];
    
    
    for(int j = 0; j < 2*blockDim.x ; j+=blockDim.x ){
        if ( i < length){
             in_s[threadIdx.x+j] = in[i+j];
        }else{
             in_s[threadIdx.x+j] = 0;
        }
            
    }
      

    __syncthreads();


    for(int stride = blockDim.x; stride >= 1; stride /= 2){
        
        if( threadIdx.x < stride ){
            in_s[threadIdx.x] += in_s[threadIdx.x+stride];
        }
            
        __syncthreads();
        
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outRef(*out);
        outRef.fetch_add(in_s[0], cuda::memory_order_relaxed);
    }


}


int main(){


    unsigned int* A_h, *R_h;

    A_h = (unsigned int * ) malloc(4096*sizeof(unsigned int));
    R_h = (unsigned int * ) malloc(1*sizeof(unsigned int));

    for(int i = 0 ; i < 4096; i++){
        A_h[i] = i;
    }
    R_h[0] = 0;


    unsigned int * A_d, *R_d;
    void*  t_A, *t_R;

    cudaMalloc(&t_A,4096*sizeof(unsigned int));
    cudaMalloc(&t_R,1*sizeof(unsigned int));

    A_d = static_cast<unsigned int*>(t_A);
    R_d = static_cast<unsigned int*>(t_R);

    cudaMemcpy(A_d, A_h,4096*sizeof(unsigned int),cudaMemcpyHostToDevice );
    cudaMemcpy(R_d, R_h,1*sizeof(unsigned int),cudaMemcpyHostToDevice );

    dim3 GridDim(8,1,1);
    dim3 BlockDim(256,1,1);

    ConvergentReduction<<<GridDim,BlockDim>>>(A_d,R_d,4096);

    cudaMemcpy(R_h,R_d,1*sizeof(unsigned int),cudaMemcpyDeviceToHost );

    std::cout << R_h[0] << std::endl;

    cudaFree(A_d); cudaFree(R_d);
    free(A_h); free(R_h);






}