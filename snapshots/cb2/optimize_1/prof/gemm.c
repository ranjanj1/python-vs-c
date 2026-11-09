#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
#define BEST(expr) ({double b=1e9;for(int r=0;r<5;r++){double t=now();for(int L=0;L<NL;L++){expr;}t=now()-t;if(t<b)b=t;} b;})
int main(){
  int n=768,d=4096,NL=12; size_t per=(size_t)n*d;
  float*W=aligned_alloc(16384,per*NL*4); for(size_t i=0;i<per*NL;i++)W[i]=(i%7)*1e-3f;
  float*X=malloc(1024*n*4),*Y=malloc(1024*(size_t)d*4); for(int i=0;i<1024*n;i++)X[i]=1e-3f;
  int Ms[]={16,76,128,512};
  for(int k=0;k<4;k++){int M=Ms[k]; double fl=2.0*M*n*d*NL;
    double a=BEST(cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,d,n,1,X,n,W+L*per,n,0,Y,d));
    double b=BEST(cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,d,n,1,X,n,W+L*per,d,0,Y,d));
    double c=BEST(for(int j=0;j<d;j+=384){int w=d-j<384?d-j:384; cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,w,n,1,X,n,W+L*per+(size_t)n*j,w,0,Y+j,d);});
    double e=BEST(for(int j=0;j<d;j+=384){int w=d-j<384?d-j:384; cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,w,n,1,X,n,W+L*per+(size_t)n*j,w,0,Y+(size_t)j*M,w);});
    printf("M=%4d  rowW NT %.0f  fullT NN %.0f  blocks NN %.0f  blocks contigY %.0f GFLOP/s\n",M,fl/a/1e9,fl/b/1e9,fl/c/1e9,fl/e/1e9);}
}
