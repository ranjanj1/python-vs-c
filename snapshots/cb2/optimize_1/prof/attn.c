#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
enum{NH=12,HS=64,KVD=768,S=1024,NL=12};
static void softmax(float*x,int n){float mx,sum;vDSP_maxv(x,1,&mx,n);float neg=-mx;vDSP_vsadd(x,1,&neg,x,1,n);vvexpf(x,x,&n);vDSP_sve(x,1,&sum,n);vDSP_vsdiv(x,1,&sum,x,1,n);}
int main(){
  size_t per=(size_t)S*KVD; float*K=malloc(per*NL*4),*Vc=malloc(per*NL*4),*Kh=malloc(per*NL*4),*Vh=malloc(per*NL*4);
  for(size_t i=0;i<per*NL;i++){K[i]=Kh[i]=(i%17)*1e-2f;Vc[i]=Vh[i]=(i%13)*1e-2f;}
  float q[KVD],out[KVD],att[NH*S],sc[NH*S],qb[KVD*NH],ho[NH*KVD]; for(int i=0;i<KVD;i++)q[i]=1e-2f*(i%5); memset(qb,0,sizeof qb);
  float sq=8; double sink=0;
  int Ts[]={8,24,40,64,100,300,1000};
  for(int ti=0;ti<7;ti++){int T=Ts[ti]; double best[4]={1e9,1e9,1e9,1e9};
   for(int rep=0;rep<20;rep++){
    double t=now();
    for(int l=0;l<NL;l++){float*kc=K+l*per,*vc=Vc+l*per;
      for(int h=0;h<NH;h++){cblas_sgemv(CblasRowMajor,CblasNoTrans,T,HS,1,kc+h*HS,KVD,q+h*HS,1,0,att,1);vDSP_vsdiv(att,1,&sq,att,1,T);softmax(att,T);cblas_sgemv(CblasRowMajor,CblasTrans,T,HS,1,vc+h*HS,KVD,att,1,0,out+h*HS,1);}}
    t=now()-t; if(t<best[0])best[0]=t; sink+=out[3];
    t=now();
    for(int l=0;l<NL;l++){float*kc=K+l*per,*vc=Vc+l*per;
      for(int h=0;h<NH;h++)cblas_scopy(HS,q+h*HS,1,qb+h*HS*NH+h,NH);
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,NH,KVD,1,kc,KVD,qb,NH,0,sc,NH);
      vDSP_mtrans(sc,1,att,1,NH,T);
      for(int h=0;h<NH;h++){vDSP_vsdiv(att+h*T,1,&sq,att+h*T,1,T);softmax(att+h*T,T);}
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NH,KVD,T,1,att,T,vc,KVD,0,ho,KVD);
      for(int h=0;h<NH;h++)memcpy(out+h*HS,ho+h*KVD+h*HS,HS*4);}
    t=now()-t; if(t<best[1])best[1]=t; sink+=out[3];
    t=now(); /* head-major cache: (head, S, HS) contiguous per head */
    for(int l=0;l<NL;l++){float*kc=Kh+l*per,*vc=Vh+l*per;
      for(int h=0;h<NH;h++){cblas_sgemv(CblasRowMajor,CblasNoTrans,T,HS,1,kc+(size_t)h*S*HS,HS,q+h*HS,1,0,att,1);vDSP_vsdiv(att,1,&sq,att,1,T);softmax(att,T);cblas_sgemv(CblasRowMajor,CblasTrans,T,HS,1,vc+(size_t)h*S*HS,HS,att,1,0,out+h*HS,1);}}
    t=now()-t; if(t<best[2])best[2]=t; sink+=out[3];
    t=now();
    for(int l=0;l<NL;l++){float*kc=K+l*per,*vc=Vc+l*per;
      for(int h=0;h<NH;h++)cblas_scopy(HS,q+h*HS,1,qb+h*HS*NH+h,NH);
      cblas_sgemm(CblasRowMajor,CblasTrans,CblasTrans,NH,T,KVD,1,qb,NH,kc,KVD,0,att,T);
      for(int h=0;h<NH;h++){vDSP_vsdiv(att+h*T,1,&sq,att+h*T,1,T);softmax(att+h*T,T);}
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NH,KVD,T,1,att,T,vc,KVD,0,ho,KVD);
      for(int h=0;h<NH;h++)memcpy(out+h*HS,ho+h*KVD+h*HS,HS*4);}
    t=now()-t; if(t<best[3])best[3]=t; sink+=out[3];
   }
   printf("T=%4d per token(12 layers): per-head strided %.3f ms   block-diag sgemm %.3f ms   head-major sgemv %.3f ms  blockdiag-TT %.3f\n",T,best[0]*1e3,best[1]*1e3,best[2]*1e3,best[3]*1e3);}
  return sink==1234.5;
}
