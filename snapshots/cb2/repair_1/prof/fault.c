#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_vm.h>
#include <mach/mach_error.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int c,char**v){
  int fd=open(v[1],O_RDONLY); size_t sz=lseek(fd,0,SEEK_END); size_t N=(sz-28)/4/1024*1024;
  double t; float s;
  char*map=mmap(NULL,sz,PROT_READ,MAP_PRIVATE,fd,0); t=now(); vDSP_sve((float*)(map+28),1,&s,N); printf("first read of mmap (faults)   %.1f ms\n",(now()-t)*1e3);
  t=now(); vDSP_sve((float*)(map+28),1,&s,N); printf("second read of mmap           %.1f ms\n",(now()-t)*1e3); munmap(map,sz);
  float*d; posix_memalign((void**)&d,16384,N*4); t=now(); memset(d,0,N*4); printf("first touch malloc (faults)   %.1f ms\n",(now()-t)*1e3);
  t=now(); memset(d,1,N*4); printf("second memset                 %.1f ms\n",(now()-t)*1e3);
  t=now(); pread(fd,d,N*4,28); printf("pread into touched buffer     %.1f ms\n",(now()-t)*1e3);
  float*e; posix_memalign((void**)&e,16384,N*4); t=now(); pread(fd,e,N*4,28); printf("pread into fresh buffer       %.1f ms\n",(now()-t)*1e3);
  /* superpage attempt */
  mach_vm_address_t a=0; kern_return_t kr=mach_vm_allocate(mach_task_self(),&a,N*4,VM_FLAGS_ANYWHERE|VM_FLAGS_SUPERPAGE_SIZE_ANY); printf("superpage alloc: %s\n",kr==KERN_SUCCESS?"ok":mach_error_string(kr));
  if(kr==KERN_SUCCESS){t=now(); memset((void*)a,0,N*4); printf("first touch superpage %.1f ms\n",(now()-t)*1e3);}
  /* transpose cost in hot memory */
  t=now(); for(size_t b=0;b+768*256<=N/2;b+=768*256) vDSP_mtrans(e+b,1,d+b,1,768,256); printf("mtrans 768x256 blocks, warm, half model  %.1f ms\n",(now()-t)*1e3);
}
