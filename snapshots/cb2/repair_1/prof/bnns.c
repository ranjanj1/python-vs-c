#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static BNNSNDArrayDescriptor mat(float*p,int rows,int cols,int ld){BNNSNDArrayDescriptor d; memset(&d,0,sizeof d);
  d.layout=BNNSDataLayoutRowMajorMatrix; d.size[0]=cols; d.size[1]=rows; d.stride[0]=1; d.stride[1]=ld; d.data=p; d.data_type=BNNSDataTypeFloat32; return d;}
int main(){
  size_t NW=48u<<20; float*W=aligned_alloc(16384,NW*4); for(size_t i=0;i<NW;i++)W[i]=(i%7)*1e-3f;
  float*X=malloc(512*2048*4),*Y=malloc(512*(size_t)8192*4),*Y2=malloc(512*(size_t)8192*4); for(int i=0;i<512*2048;i++)X[i]=1e-3f*(i%3);
  int n=768,w=256,d=8192; size_t per=(size_t)n*w, cnt=NW/per;
  int Ms[]={1,16,76,128};
  for(int k=0;k<4;k++){int M=Ms[k]; double b1=1e9,b2=1e9;
    BNNSNDArrayDescriptor A=mat(X,M,n,n);
    BNNSNDArrayDescriptor B0=mat(W,n,w,w), C0=mat(Y2,M,w,d);
    ssize_t ws=BNNSMatMulWorkspaceSize(false,false,1.0f,&A,&B0,&C0,NULL); void*wsp=ws>0?malloc(ws):NULL;
    for(int r=0;r<4;r++){double t=now();for(size_t c=0;c<cnt;c++){ if(M==1) cblas_sgemv(CblasRowMajor,CblasTrans,n,w,1,W+c*per,w,X,1,0,Y+(c*w)%d,1); else cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,w,n,1,X,n,W+c*per,w,0,Y+(c*w)%d,d);}t=now()-t;if(t<b1)b1=t;
      t=now();for(size_t c=0;c<cnt;c++){BNNSNDArrayDescriptor B=mat(W+c*per,n,w,w),C=mat(Y2+(c*w)%d,M,w,d); if(BNNSMatMul(false,false,1.0f,&A,&B,&C,wsp,NULL)){printf("bnns fail\n");return 1;}}t=now()-t;if(t<b2)b2=t;}
    double err=0; for(int i=0;i<M;i++)for(int j=0;j<w;j++){double e=Y[i*d+j]-Y2[i*d+j]; if(e<0)e=-e; if(e>err)err=e;}
    double fl=2.0*M*n*w*cnt; printf("M=%3d  BLAS %.0f GFLOP/s (%.1f GB/s)   BNNSMatMul %.0f GFLOP/s (%.1f GB/s)  maxdiff %.2g  ws=%zd\n",M,fl/b1/1e9,cnt*per*4/b1/1e9,fl/b2/1e9,cnt*per*4/b2/1e9,err,ws);}
}
