#include <iostream>
#include <cstdlib>
#include <cuda/atomic>

#define BINS 256
#define COARSE_FACTOR 4



__global__ void Histogram(unsigned int* A, unsigned int* bin,unsigned int length){

    __shared__ unsigned int bP[BINS];

    for(unsigned int b = threadIdx.x; b < BINS; b+= blockDim.x){
        bP[b] =0;
    }

    __syncthreads();

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < length){
        unsigned char b = A[i];
        cuda::atomic_ref<unsigned int,cuda::thread_scope_block> binRefP(bP[b]);
        binRefP.fetch_add(1,cuda::memory_order_relaxed);
    }
    __syncthreads();

    for(unsigned int b = threadIdx.x; b < BINS; b+=blockDim.x){
        if (bP[b] > 0 ){
            cuda::atomic_ref<unsigned int,cuda::thread_scope_device> binRef(bin[b]);
            binRef.fetch_add(bP[b],cuda::memory_order_relaxed);
        }
        
    }


}

__global__ void contiguousCoarsenedHistogram(unsigned int* A, unsigned int* bin,unsigned int length){

    __shared__ unsigned int bP[BINS];

    for(unsigned int b = threadIdx.x; b < BINS; b+= blockDim.x){
        bP[b] =0;
    }

    __syncthreads();


    unsigned int segment = COARSE_FACTOR* blockDim.x * blockIdx.x;

   

    
    for (int c = 0 ; c < COARSE_FACTOR; ++c){

            unsigned int i = segment + COARSE_FACTOR*threadIdx.x + c;
            if (i  < length){
                unsigned char b = A[i];
                cuda::atomic_ref<unsigned int,cuda::thread_scope_block> binRefP(bP[b]);
                binRefP.fetch_add(1,cuda::memory_order_relaxed);
            }
    }
       
    __syncthreads();

    for(unsigned int b = threadIdx.x; b < BINS; b+=blockDim.x){
        if (bP[b] > 0 ){
            cuda::atomic_ref<unsigned int,cuda::thread_scope_device> binRef(bin[b]);
            binRef.fetch_add(bP[b],cuda::memory_order_relaxed);
        }
        
    }


}

__global__ void interleavedCoarsenedHistogram(unsigned int* A, unsigned int* bin,unsigned int length){

    __shared__ unsigned int bP[BINS];

    for(unsigned int b = threadIdx.x; b < BINS; b+= blockDim.x){
        bP[b] =0;
    }

    __syncthreads();


    unsigned int segment = COARSE_FACTOR* blockDim.x * blockIdx.x;

    
    for (int c = 0 ; c < COARSE_FACTOR; ++c){

            unsigned int i = segment + c*blockDim.x + threadIdx.x;
            if (i  < length){
                unsigned char b = A[i];
                cuda::atomic_ref<unsigned int,cuda::thread_scope_block> binRefP(bP[b]);
                binRefP.fetch_add(1,cuda::memory_order_relaxed);
            }
    }
       
    __syncthreads();

    for(unsigned int b = threadIdx.x; b < BINS; b+=blockDim.x){
        if (bP[b] > 0 ){
            cuda::atomic_ref<unsigned int,cuda::thread_scope_device> binRef(bin[b]);
            binRef.fetch_add(bP[b],cuda::memory_order_relaxed);
        }
        
    }


}

int main(){


    unsigned int* A_h, *R_h;

    A_h = (unsigned int * ) malloc(4096*sizeof(unsigned int));
    R_h = (unsigned int * ) malloc(BINS*sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 4096; i++){
        A_h[i] = rand() % 256;
       
    }
    


    unsigned int * A_d, *R_d;
    void*  t_A, *t_R;

    cudaMalloc(&t_A,4096*sizeof(unsigned int));
    cudaMalloc(&t_R,BINS*sizeof(unsigned int));

    A_d = static_cast<unsigned int*>(t_A);
    R_d = static_cast<unsigned int*>(t_R);

    cudaMemcpy(A_d, A_h,4096*sizeof(unsigned int),cudaMemcpyHostToDevice );
    cudaMemset(R_d, 0, BINS*sizeof(unsigned int));


    dim3 GridDim(32,1,1);
    dim3 BlockDim(128,1,1);

    interleavedCoarsenedHistogram<<<GridDim,BlockDim>>>(A_d,R_d,4096);

    cudaMemcpy(R_h,R_d,BINS*sizeof(unsigned int),cudaMemcpyDeviceToHost );

    unsigned int* m = (unsigned int*) calloc(BINS ,sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 4096; i++){
        m[A_h[i]] +=1;
    }

    for(unsigned int i = 0 ; i < BINS; i++){
        std::cout << m[i] << " " << R_h[i] << " " << (m[i] == R_h[i]) << " " << std::endl;
    }


    cudaFree(A_d); cudaFree(R_d);
    free(A_h); free(R_h);
    free(m);






}