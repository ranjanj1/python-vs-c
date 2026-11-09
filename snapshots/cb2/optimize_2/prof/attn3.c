#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
enum{NH=12,HS=64,KVD=768,S=1024,NL=12};
static void softmax(float*x,int n){float mx,sum;vDSP_maxv(x,1,&mx,n);float neg=-mx;vDSP_vsadd(x,1,&neg,x,1,n);vvexpf(x,x,&n);vDSP_sve(x,1,&sum,n);vDSP_vsdiv(x,1,&sum,x,1,n);}
static float *junk; static size_t JN=24u<<20; static float js;
static void flush(void){vDSP_sve(junk,1,&js,JN);} /* 96MB read evicts caches, like the weight stream */
int main(){
  junk=malloc(JN*4); for(size_t i=0;i<JN;i++)junk[i]=1;
  size_t per=(size_t)S*KVD; float*K=malloc(per*NL*4),*Vc=malloc(per*NL*4),*Kh=malloc(per*NL*4),*Vh=malloc(per*NL*4);
  for(size_t i=0;i<per*NL;i++){K[i]=Kh[i]=(i%17)*1e-2f;Vc[i]=Vh[i]=(i%13)*1e-2f;}
  static float q[KVD],out[KVD],att[NH*S],sc[NH*S],qb[KVD*NH],ho[NH*KVD]; for(int i=0;i<KVD;i++)q[i]=1e-2f*(i%5);
  float sq=8; double sink=0;
  int Ts[]={8,24,48,80,120,160,300,1000};
  for(int ti=0;ti<8;ti++){int T=Ts[ti]; double tt[4]={0};
   for(int rep=0;rep<6;rep++) for(int l=0;l<NL;l++){float*kc=K+l*per,*vc=Vc+l*per,*kh=Kh+l*per,*vh=Vh+l*per; double t;
    flush(); t=now();
    for(int h=0;h<NH;h++){cblas_sgemv(CblasRowMajor,CblasNoTrans,T,HS,1,kc+h*HS,KVD,q+h*HS,1,0,att,1);vDSP_vsdiv(att,1,&sq,att,1,T);softmax(att,T);cblas_sgemv(CblasRowMajor,CblasTrans,T,HS,1,vc+h*HS,KVD,att,1,0,out+h*HS,1);}
    tt[0]+=now()-t; sink+=out[1];
    flush(); t=now();
    for(int h=0;h<NH;h++)cblas_scopy(HS,q+h*HS,1,qb+h*HS*NH+h,NH);
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,NH,KVD,1,kc,KVD,qb,NH,0,sc,NH);
    vDSP_mtrans(sc,1,att,1,NH,T);
    for(int h=0;h<NH;h++){vDSP_vsdiv(att+h*T,1,&sq,att+h*T,1,T);softmax(att+h*T,T);}
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NH,KVD,T,1,att,T,vc,KVD,0,ho,KVD);
    for(int h=0;h<NH;h++)memcpy(out+h*HS,ho+h*KVD+h*HS,HS*4);
    tt[1]+=now()-t; sink+=out[1];
    flush(); t=now(); /* head-major (head,S,HS) */
    for(int h=0;h<NH;h++){cblas_sgemv(CblasRowMajor,CblasNoTrans,T,HS,1,kh+(size_t)h*S*HS,HS,q+h*HS,1,0,att,1);vDSP_vsdiv(att,1,&sq,att,1,T);softmax(att,T);cblas_sgemv(CblasRowMajor,CblasTrans,T,HS,1,vh+(size_t)h*S*HS,HS,att,1,0,out+h*HS,1);}
    tt[2]+=now()-t; sink+=out[1];
    flush(); t=now(); /* head-major, K stored transposed per head (head,HS,S): scores via sgemvT over T columns */
    for(int h=0;h<NH;h++){cblas_sgemv(CblasRowMajor,CblasTrans,HS,T,1,kh+(size_t)h*S*HS,S,q+h*HS,1,0,att,1);vDSP_vsdiv(att,1,&sq,att,1,T);softmax(att,T);cblas_sgemv(CblasRowMajor,CblasTrans,T,HS,1,vh+(size_t)h*S*HS,HS,att,1,0,out+h*HS,1);}
    tt[3]+=now()-t; sink+=out[1];
   }
   printf("T=%4d ms/token (cold): strided %.3f  blockdiag %.3f  head-major %.3f  head-major Kt %.3f\n",T,tt[0]/6*1e3,tt[1]/6*1e3,tt[2]/6*1e3,tt[3]/6*1e3);}
  return sink==1.5;
}
