#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define R 6
int main(){
  size_t N=110u<<20; /* 440MB of floats */
  float*W=aligned_alloc(16384,N*4); for(size_t i=0;i<N;i++)W[i]=(i%7)*1e-3f;
  float x[8192],*y=malloc(1<<22),s; for(int i=0;i<8192;i++)x[i]=1e-3f;
  double t,b;
  b=1e9;for(int r=0;r<R;r++){t=now();vDSP_sve(W,1,&s,N);t=now()-t;if(t<b)b=t;} printf("vDSP_sve pure read   %.1f GB/s\n",N*4/b/1e9);
  /* sgemv shapes covering the whole buffer */
  int shapes[][2]={{768,768},{2304,768},{4096,768},{768,2048},{32000,768},{1024,8192},{8192,1024},{128,768}};
  for(int k=0;k<8;k++){int d=shapes[k][0],n=shapes[k][1]; size_t per=(size_t)d*n, cnt=N/per;
    b=1e9;for(int r=0;r<R;r++){t=now();for(size_t c=0;c<cnt;c++)cblas_sgemv(CblasRowMajor,CblasNoTrans,d,n,1,W+c*per,n,x,1,0,y,1);t=now()-t;if(t<b)b=t;}
    double bn=1e9;for(int r=0;r<R;r++){t=now();for(size_t c=0;c<cnt;c++)cblas_sgemv(CblasRowMajor,CblasTrans,n,d,1,W+c*per,d,x,1,0,y,1);t=now()-t;if(t<bn)bn=t;}
    double bm=1e9;for(int r=0;r<R;r++){t=now();for(size_t c=0;c<cnt;c++)cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,d,n,1,x,n,W+c*per,n,0,y,d);t=now()-t;if(t<bm)bm=t;}
    double bmm=1e9;for(int r=0;r<R;r++){t=now();for(size_t c=0;c<cnt;c++)vDSP_mmul(W+c*per,1,x,1,y,1,d,1,n);t=now()-t;if(t<bmm)bmm=t;}
    printf("%5dx%-5d sgemvN %.1f  sgemvT(layoutT) %.1f  sgemm(M=1) %.1f  vDSP_mmul %.1f GB/s\n",d,n,cnt*per*4/b/1e9,cnt*per*4/bn/1e9,cnt*per*4/bm/1e9,cnt*per*4/bmm/1e9);}
}
