#include <iostream>

#define SIZE 256
#define WARP_SIZE 32
#define NUM_WARPS 8

#define COARSE_FACTOR 4
#define BLOCK_DIM 256


__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }


__global__ void parallelScan(unsigned int* in, unsigned int N ){ // Kooge Stone Algorithm Based

    __shared__ unsigned int in_s[SIZE];
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N){
        in_s[threadIdx.x] = in[i];
    }else{
        in_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride*=2){
        unsigned int temp;
        if(threadIdx.x >= stride){
            temp = in_s[threadIdx.x] + in_s[threadIdx.x - stride];
        }
        __syncthreads();
        if(threadIdx.x >= stride){
            in_s[threadIdx.x] = temp;
        }
         __syncthreads();

    }
    if(i < N){
        in[i] = in_s[threadIdx.x];
    }

}

__global__ void parallelScanDoubleBuffered(unsigned int* in, unsigned int N ){ // Kooge Stone Algorithm Based

    __shared__ unsigned int in_s[SIZE];
    __shared__ unsigned int out_s[SIZE];

    unsigned int* in_s_ = in_s;
    unsigned int* out_s_ = out_s;


    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < N){
        in_s[threadIdx.x] = in[i];
    }else{
        in_s[threadIdx.x] = 0;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride*=2){
        
        if(threadIdx.x >= stride){
            out_s_[threadIdx.x] = in_s_[threadIdx.x] + in_s_[threadIdx.x - stride];
        }else{
            out_s_[threadIdx.x] = in_s_[threadIdx.x];
        }

        unsigned int* temp;
        temp = in_s_;
        in_s_ = out_s_;
        out_s_ = temp;
        __syncthreads();


    }
    if(i < N){
        in[i] = in_s[threadIdx.x];
    }

}


// warp Scan using Warp Primitives

__device__ __inline__ int warpScan(unsigned int val){

    for(int stride = 1; stride < WARP_SIZE; stride*=2){
        unsigned int lVal =__shfl_up_sync(0xFFFFFFFF,val,stride);
        if(threadIdx.x % WARP_SIZE >= stride){
            val+=lVal;
        }
    }

    return val;

}

__device__ __inline__ int BlockScan(unsigned int val){

    val = warpScan(val);

    __shared__ unsigned int WarpScans[NUM_WARPS];

    if(laneIdx() == WARP_SIZE -1 )
        WarpScans[warpIdx()] = val;

    __syncthreads();
    if (warpIdx() ==0 ){
        unsigned int wSum = (threadIdx.x < NUM_WARPS)? WarpScans[threadIdx.x] : 0;
        wSum = warpScan(wSum);
        if(threadIdx.x < NUM_WARPS){
            WarpScans[threadIdx.x] = wSum;
        }
    }
    __syncthreads();

    if(warpIdx() > 0){
        val += WarpScans[warpIdx()-1];
    }

    return val;

}


__global__ void warppedParallelScan(unsigned int* in, unsigned int N ){ // Kooge Stone Algorithm Based using Warp Primitives in a 2 stage method

    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    unsigned int val = (i < N)? in[i]:0;
    val = BlockScan(val);

    if(i < N){
        in[i] = val;
    }

}

__global__ void coarsenedParrallelScan(unsigned int* in, unsigned int N){


    unsigned int blockSeg = COARSE_FACTOR*blockDim.x*blockIdx.x;
    unsigned int threadSeg = COARSE_FACTOR*threadIdx.x;

    __shared__ unsigned int in_s[BLOCK_DIM*COARSE_FACTOR];

    for(int c = 0; c <COARSE_FACTOR;c++){
        in_s[ c*blockDim.x  + threadIdx.x ] += in[blockSeg + c*blockDim.x + threadIdx.x];
    }

    __syncthreads();

    for(int c = 1; c < COARSE_FACTOR; c++){
        in_s[threadSeg+c] += in_s[threadSeg+ c -1 ];
    }

    unsigned int tSum = in_s[threadSeg+COARSE_FACTOR-1];
    tSum = BlockScan(tSum);

    __shared__ unsigned int t_S[BLOCK_DIM];
    t_S[threadIdx.x] = tSum;

    if(threadIdx.x > 0){
        unsigned int pSum = t_S[threadIdx.x - 1];
        for(int c = 0; c <COARSE_FACTOR;c++){
            in_s[threadSeg+c]+= pSum;
        }
    }
    __syncthreads();
    for(int c = 0; c <COARSE_FACTOR;c++){
            in[blockSeg + c*blockDim.x + threadIdx.x] = in_s[ c*blockDim.x  + threadIdx.x ];
    }


}







int main(){

    unsigned int* A_h = (unsigned int* ) malloc(256*sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 256; i++){
        A_h[i] = rand() % 256;
       
    }

    unsigned int* m = (unsigned int* ) calloc(256,sizeof(unsigned int));
    for(int i = 0; i < 256; i++ ){
        unsigned int temp =0;
        for(int j = 0; j <=i ;j++ ){
            temp += A_h[j];
        }
        m[i] = temp;
    }


    unsigned int * A_d;
    void*  t_A;

    cudaMalloc(&t_A,256*sizeof(unsigned int));
    A_d = static_cast<unsigned int*>(t_A);
    cudaMemcpy(A_d, A_h,256*sizeof(unsigned int),cudaMemcpyHostToDevice );

    dim3 GridDim(1,1,1);
    dim3 BlockDim(256,1,1);
    
    

    coarsenedParrallelScan<<<GridDim,BlockDim>>>(A_d,256);

    cudaMemcpy(A_h,A_d,256*sizeof(unsigned int),cudaMemcpyDeviceToHost );
    for(int i = 0; i < 256; i++ ){
         std::cout << A_h[i] << " " << m[i] << " " << ( A_h[i] == m[i] ) << " " << std::endl;
    }
   


    cudaFree(A_d);
    free(A_h); 
    free(m);









}