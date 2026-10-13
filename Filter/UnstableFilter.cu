#include <iostream>
#include <cuda/atomic>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;   

#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void naiveUnstableFilter(unsigned int* in , unsigned int* out, unsigned int N,unsigned int* M){
    unsigned int i = threadIdx.x + (blockDim.x*blockIdx.x);


    if(i < N){
        unsigned int val = in[i];
        if(val%2 ==0 ){
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> output_ref(*M);
            unsigned int j = output_ref.fetch_add(1,cuda::memory_order_relaxed);
            out[j] = val;
        }
    }


}

__global__ void coalescedAtomicsUnstableFilter(unsigned int* in , unsigned int* out, unsigned int N,unsigned int* M){
     unsigned int i = threadIdx.x + (blockDim.x*blockIdx.x);
     if(i < N){
        unsigned int val = in[i];
        if(val%2 ==0 ){
            unsigned int activeThreads = __activemask();
            unsigned int threadLeader = __ffs(activeThreads)-1;
            unsigned int j;
            if(threadIdx.x%WARP_SIZE == threadLeader){
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> output_ref(*M);
                j = output_ref.fetch_add(__popc(activeThreads),cuda::memory_order_relaxed);
            }
            j = __shfl_sync(activeThreads,j,threadLeader);
            unsigned int prevThreads = (1 << (threadIdx.x%WARP_SIZE)) - 1;
            unsigned int offset = __popc(prevThreads & activeThreads);
            out[j+offset] = val;
        }
    }

}

__global__ void coalescedAtomicCooperativeGroupsUnstableFilter(unsigned int* in , unsigned int* out, unsigned int N,unsigned int* M){
     unsigned int i = threadIdx.x + (blockDim.x*blockIdx.x);
     if(i < N){
        unsigned int val = in[i];
        if(val%2 ==0 ){
            cg::coalesced_group activeThreads = cg::coalesced_threads();
            unsigned int j;
            if(activeThreads.thread_rank() == 0){
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device> output_ref(*M);
                j = output_ref.fetch_add(activeThreads.size(),cuda::memory_order_relaxed);
            }
            j = activeThreads.shfl(j,0);
            out[j+activeThreads.thread_rank()] = val;
        }
    }

}


int main(){


    unsigned int* A_h = (unsigned int* ) malloc(256*sizeof(unsigned int)); 
    unsigned int* out_h = (unsigned int* ) malloc(256*sizeof(unsigned int));
    unsigned int* M = (unsigned int* ) malloc(1*sizeof(unsigned int));

    for(unsigned int i = 0 ; i < 256; i++){
        unsigned int v = rand() % 256;
        A_h[i] = v;
       
    }
    int k = 0;
    for(int i = 0 ; i < 256; i++){
        if( A_h[i]%2  ==0 ){
            k+=1;
        }
    }


    unsigned int * A_d;unsigned int * OUT_d; unsigned int * M_d;
    void*  t_A; void*  t_OUT; void*  t_M;

    cudaMalloc(&t_A,256*sizeof(unsigned int));
    cudaMalloc(&t_OUT,128*sizeof(unsigned int));
    cudaMalloc(&t_M,1*sizeof(unsigned int));
    A_d = static_cast<unsigned int*>(t_A);
    OUT_d = static_cast<unsigned int*>(t_OUT);
    M_d = static_cast<unsigned int*>(t_M);
    cudaMemset(OUT_d,0,128);
    cudaMemset(M_d,0,1);
    cudaMemcpy(A_d, A_h,256*sizeof(unsigned int),cudaMemcpyHostToDevice );



    dim3 GridDim(1,1,1);
    dim3 BlockDim(256,1,1);
    
    

    coalescedAtomicCooperativeGroupsUnstableFilter<<<GridDim,BlockDim>>>(A_d,OUT_d,256,M_d);
    

    cudaMemcpy(out_h,OUT_d,128*sizeof(unsigned int),cudaMemcpyDeviceToHost );
    cudaMemcpy(M,M_d,1*sizeof(unsigned int),cudaMemcpyDeviceToHost );
    
    std::cout << "hi " << *M  << " " << k << " "<<  std::endl;
    
    for(int i = 0 ; i < *M; i++){
        std::cout << out_h[i] <<  std::endl;
    }

   
    cudaFree(A_d); cudaFree(OUT_d); cudaFree(M_d);
    free(A_h); free(out_h);free(M);





}