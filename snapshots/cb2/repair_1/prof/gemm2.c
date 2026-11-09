#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  size_t NW=48u<<20; float*W=aligned_alloc(16384,NW*4); for(size_t i=0;i<NW;i++)W[i]=(i%7)*1e-3f; /* 192MB, > caches */
  float*X=malloc(512*2048*4),*Y=malloc(512*(size_t)8192*4),*Yc=malloc(512*(size_t)8192*4); for(int i=0;i<512*2048;i++)X[i]=1e-3f;
  int Ms[]={16,76,128}; int ns[]={768,2048}; int ws[]={256,512,1024};
  for(int a=0;a<2;a++)for(int k=0;k<3;k++){int n=ns[a],M=Ms[k];
   printf("K=%4d M=%3d:",n,M);
   for(int wi=0;wi<3;wi++){int w=ws[wi]; int d=8192; size_t per=(size_t)n*w, cnt=NW/per; double b1=1e9,b2=1e9;
    for(int r=0;r<4;r++){double t=now();for(size_t c=0;c<cnt;c++)cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,w,n,1,X,n,W+c*per,w,0,Y+(c*w)%d,d);t=now()-t;if(t<b1)b1=t;
                         t=now();for(size_t c=0;c<cnt;c++)cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,w,n,1,X,n,W+c*per,w,0,Yc,w);t=now()-t;if(t<b2)b2=t;}
    double fl=2.0*M*n*w*cnt; printf("  w=%d ldc=d %.0f / ldc=w %.0f",w,fl/b1/1e9,fl/b2/1e9);}
   printf(" GFLOP/s\n");}
}
