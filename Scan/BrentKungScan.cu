#include <iostream>



#define BLOCK_SIZE 128

#define WARP_SIZE 32

__device__ int warpIdx() { return threadIdx.x/WARP_SIZE; }
__device__  int laneIdx() { return threadIdx.x%WARP_SIZE; }


__device__ __inline__ unsigned int warpScan(unsigned int val){

    for(int stride = 1; stride < WARP_SIZE ; stride*=2){
        unsigned int lVal = __shfl_up_sync(0xFFFFFFFF,val,stride);
        if(laneIdx() % (2*stride) == (2*stride - 1)){
            val+=lVal;
        }

    }
    for(int stride = WARP_SIZE/4; stride >0  ; stride/=2){
        unsigned int rVal = __shfl_up_sync(0xFFFFFFFF,val,stride);
        if((laneIdx()-stride) > 0 ){
            if((laneIdx()-stride) % (2*stride) == (2*stride - 1)){
                val+=rVal;
            }
        }
    }  
    return val;

        
}
__global__ void BK_Scan(unsigned int* in ,  unsigned int N){

    unsigned int val = in[threadIdx.x];
    val = warpScan(val);
    if(threadIdx.x % WARP_SIZE ==0){
        
    }

    __syncthreads();
    in[threadIdx.x] = val;

}


__global__ void naiveBK_Scan(unsigned int* in ,  unsigned int N){


    unsigned int tx = threadIdx.x;
    unsigned int seg = 2*blockDim.x*blockIdx.x;

    __shared__ unsigned int in_s[BLOCK_SIZE*2];

    in_s[tx] = (tx+seg < N) ? in[tx+seg]:0;
    in_s[tx+blockDim.x] = (tx+seg+blockDim.x < N) ? in[tx+seg+blockDim.x]:0;
    __syncthreads();



    
    for(int stride = 1; stride <= blockDim.x ; stride*=2){
        unsigned int idx = (threadIdx.x+1)*2*stride - 1;
        if(idx < N ){
            in_s[idx] += + in_s[idx - stride];

        }
        __syncthreads();
    }



    for(int stride = N/4; stride > 0 ; stride/=2){
        unsigned int idx = (threadIdx.x+1)*2*stride -1;
        if((threadIdx.x+1)*stride  < N){
            in_s[idx + stride] += in_s[idx];
        }
        __syncthreads();

    }
    
    if (seg + tx < N)              in[seg + tx]              = in_s[tx];
    if (seg + tx + blockDim.x < N) in[seg + tx + blockDim.x] = in_s[tx + blockDim.x];


}













int main(){

    unsigned int* A_h = (unsigned int* ) malloc(256*sizeof(unsigned int));
    unsigned int* B_h = (unsigned int* ) malloc(256*sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 256; i++){
        unsigned int v = rand() % 256;
        A_h[i] = v;
        B_h[i] = v;
       
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
    dim3 BlockDim(128,1,1);
    
    

    BK_Scan<<<GridDim,BlockDim>>>(A_d,256);

    cudaMemcpy(A_h,A_d,256*sizeof(unsigned int),cudaMemcpyDeviceToHost );
    for(int i = 0; i < 256; i++ ){
         std::cout << A_h[i] << " " << m[i] << " " << ( A_h[i] == m[i] )  << " " << B_h[i]  << " | " << i << " " << " " << std::endl;
    }
   


    cudaFree(A_d);
    free(A_h); 
    free(m);









}