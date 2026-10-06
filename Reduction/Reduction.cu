#include <iostream>
#include <cuda/atomic>

#define SIZE 512
#define WARP_SIZE 32

__global__ void ConvergentReduction(unsigned int* in , unsigned int* out , unsigned int length){
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

__global__ void ConvergentReduction2(unsigned int* in , unsigned int* out , unsigned int length){
    int seg = 2*blockDim.x * blockIdx.x;
    int i = seg + threadIdx.x;

    __shared__ unsigned int in_s[SIZE/2];
    if ( i < length){
        in_s[threadIdx.x] = in[i] + in[i+blockDim.x];
    }else{
        in_s[threadIdx.x] = 0;
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


__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }


__device__ __inline__ unsigned int WarpReduction(unsigned int val){
    unsigned int partialVal = val;

    for(int stride = WARP_SIZE/2; stride > 0; stride /= 2){
        partialVal += __shfl_down_sync(0xffffffff,partialVal,stride);
    }
    return partialVal;

}

__global__ void ConvergentReductionWithWarpPrimitives(unsigned int* in , unsigned int* out , unsigned int length){
    int seg = 2*blockDim.x * blockIdx.x;
    int i = seg + threadIdx.x;

    __shared__ unsigned int in_s[SIZE/2];
    if ( i < length){
        in_s[threadIdx.x] = in[i] + in[i+blockDim.x];
    }else{
        in_s[threadIdx.x] = 0;
    }
    
    __syncthreads();


    for(int stride = blockDim.x; stride >= WARP_SIZE; stride /= 2){
        
        if( threadIdx.x < stride ){
            in_s[threadIdx.x] += in_s[threadIdx.x+stride];
        }
            
        __syncthreads();
        
    }
    if (warpIdx() == 0) {
        unsigned int s = in_s[threadIdx.x];
        unsigned int v = WarpReduction(s);
        if(threadIdx.x == 0){
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outRef(*out);
            outRef.fetch_add(v, cuda::memory_order_relaxed);
        }
        
    }


}


__global__ void WarpTwoStageReduction(unsigned int* in , unsigned int* out , unsigned int length){
    int seg = 2*blockDim.x * blockIdx.x;
    int i = seg + threadIdx.x;

    __shared__ unsigned int in_s[(SIZE/2) / WARP_SIZE];
    unsigned int val = in[i] + in[i+blockDim.x];
    val = WarpReduction(val);
    if(laneIdx() == 0 ){
        in_s[warpIdx()] = val;
    }
    __syncthreads();

    if (warpIdx() == 0) {
        unsigned int s = in_s[threadIdx.x];
        unsigned int v = WarpReduction(s);
        if(threadIdx.x == 0){
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outRef(*out);
            outRef.fetch_add(v, cuda::memory_order_relaxed);
        }
        
    }


}




int main(){


    unsigned int* A_h, *R_h;

    A_h = (unsigned int * ) malloc(65536*sizeof(unsigned int));
    R_h = (unsigned int * ) malloc(1*sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 65536; i++){
        A_h[i] = i;
    }


    unsigned int * A_d, *R_d;
    void*  t_A, *t_R;

    cudaMalloc(&t_A,65536*sizeof(unsigned int));
    cudaMalloc(&t_R,1*sizeof(unsigned int));

    A_d = static_cast<unsigned int*>(t_A);
    R_d = static_cast<unsigned int*>(t_R);

    cudaMemcpy(A_d, A_h,65536*sizeof(unsigned int),cudaMemcpyHostToDevice );
    cudaMemset(R_d, 0, sizeof(unsigned int));  

    dim3 GridDim(256,1,1);
    dim3 BlockDim(256,1,1);


    cudaEvent_t start,stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    WarpTwoStageReduction<<<GridDim, BlockDim>>>(A_d, R_d,65536);   // warm-up
    cudaMemset(R_d, 0, sizeof(unsigned int));  

    cudaEventRecord(start,0);

    WarpTwoStageReduction<<<GridDim,BlockDim>>>(A_d,R_d,65536);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime,start,stop);
    std::cout<<"Elapsed Time = "<< (elapsedTime) <<" milliseconds" << std::endl;

    cudaMemcpy(R_h,R_d,1*sizeof(unsigned int),cudaMemcpyDeviceToHost );
    unsigned int ans = static_cast<unsigned int>(65536 * 65535);
    std::cout << R_h[0] << " " << ( ans/2) << " " << std::endl;

    cudaFree(A_d); cudaFree(R_d);
    free(A_h); 






}