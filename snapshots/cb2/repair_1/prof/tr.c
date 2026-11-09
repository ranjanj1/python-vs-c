#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void tr(const float*src,float*dst,size_t d,int n,int w){for(size_t j=0;j<d;j+=w)vDSP_mtrans(src+j*n,1,dst+j*n,1,n,w);}
int main(int c,char**v){
  int fd=open(v[1],O_RDONLY); size_t sz=lseek(fd,0,SEEK_END);
  size_t d=32000; int n=768; size_t N=d*n; double t;
  for(int rep=0;rep<2;rep++){
  char*map=mmap(NULL,sz,PROT_READ,MAP_PRIVATE,fd,0); const float*src=(const float*)(map+28);
  float*dst=mmap(NULL,N*4,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
  t=now(); tr(src,dst,d,n,256); double a=now()-t;              /* fresh map + fresh dst: as in load */
  t=now(); tr(src,dst,d,n,256); double b=now()-t;              /* both warm: pure transpose */
  float*dst2=mmap(NULL,N*4,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
  t=now(); tr(src,dst2,d,n,256); double e=now()-t;             /* warm src, fresh dst: dst faults */
  munmap(map,sz); map=mmap(NULL,sz,PROT_READ,MAP_PRIVATE,fd,0); src=(const float*)(map+28);
  t=now(); tr(src,dst,d,n,256); double f=now()-t;              /* fresh src map, warm dst: src faults */
  float*stage=malloc(N*4); memset(stage,0,N*4);
  t=now(); pread(fd,stage,N*4,28); tr(stage,dst,d,n,256); double g=now()-t; /* pread to warm staging + transpose */
  printf("98MB cls: load-like %.1f | pure transpose %.1f | +dst faults %.1f | +src faults %.1f | pread-stage+transpose(warm dst) %.1f ms\n",a*1e3,b*1e3,e*1e3,f*1e3,g*1e3);
  munmap(map,sz); munmap(dst,N*4); munmap(dst2,N*4); free(stage);}
}
