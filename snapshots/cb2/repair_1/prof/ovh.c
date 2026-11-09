#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  size_t N=110u<<20; float*W=aligned_alloc(16384,N*4); for(size_t i=0;i<N;i++)W[i]=(i%7)*1e-3f;
  float x[4096],*y=malloc(1<<24),s; for(int i=0;i<4096;i++)x[i]=1e-3f;
  for(int rep=0;rep<3;rep++){
  double b=1e9; for(int r=0;r<5;r++){double t=now();vDSP_sve(W,1,&s,N);t=now()-t;if(t<b)b=t;} printf("pure read %.1f GB/s | ",N*4/b/1e9);
  /* streamed 256-wide blocks, n=768, same total bytes as model (~438MB) */
  size_t per=768*256, cnt=N/per; b=1e9;
  for(int r=0;r<5;r++){double t=now();for(size_t c=0;c<cnt;c++)cblas_sgemv(CblasRowMajor,CblasTrans,768,256,1,W+c*per,256,x,1,0,y+(c*256)%(1<<20),1);t=now()-t;if(t<b)b=t;}
  printf("stream sgemvT 768x256: %.1f GB/s (%zu calls, %.2f ms) | ",cnt*per*4/b/1e9,cnt,b*1e3);
  /* overhead: tiny cached block */
  b=1e9; for(int r=0;r<5;r++){double t=now();for(int c=0;c<10000;c++)cblas_sgemv(CblasRowMajor,CblasTrans,64,256,1,W,256,x,1,0,y,1);t=now()-t;if(t<b)b=t;}
  printf("cached 64x256 call: %.2f us\n",b/10000*1e6);}
}
