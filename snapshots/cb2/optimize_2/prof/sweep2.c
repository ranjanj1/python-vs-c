#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(){
  size_t N=96u<<20; float*W=aligned_alloc(16384,N*4); for(size_t i=0;i<N;i++)W[i]=(i%7)*1e-3f;
  float x[8192],*y=malloc(1<<24); for(int i=0;i<8192;i++)x[i]=1e-3f;
  int ns[]={768,2048};
  for(int a=0;a<2;a++){int n=ns[a]; printf("n=%d:",n);
   for(int db=224;db<=704;db+=32){ size_t per=(size_t)n*db, cnt=N/per; double b=1e9;
    for(int r=0;r<5;r++){double t=now();for(size_t c=0;c<cnt;c++)cblas_sgemv(CblasRowMajor,CblasTrans,n,db,1,W+c*per,db,x,1,0,y+(c*db)%(1<<21),1);t=now()-t;if(t<b)b=t;}
    printf(" %d:%.0f",db,cnt*per*4/b/1e9);}
   printf("\n");}
}
