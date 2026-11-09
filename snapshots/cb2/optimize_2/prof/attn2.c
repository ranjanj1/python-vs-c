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
  size_t per=(size_t)S*KVD; float*K=malloc(per*NL*4),*Vc=malloc(per*NL*4);
  for(size_t i=0;i<per*NL;i++){K[i]=(i%17)*1e-2f;Vc[i]=(i%13)*1e-2f;}
  static float att[NH*S],sc[NH*S],qb[KVD*NH],ho[NH*KVD],q[KVD]; for(int i=0;i<KVD;i++)q[i]=1e-2f*(i%5);
  for(int h=0;h<NH;h++)cblas_scopy(HS,q+h*HS,1,qb+h*HS*NH+h,NH);
  float sq=8; int T=1000; double tt[6]={0};
  for(int rep=0;rep<20;rep++) for(int l=0;l<NL;l++){float*kc=K+l*per,*vc=Vc+l*per; double t=now(),u;
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,NH,KVD,1,kc,KVD,qb,NH,0,sc,NH); u=now();tt[0]+=u-t;t=u;
      vDSP_mtrans(sc,1,att,1,NH,T); u=now();tt[1]+=u-t;t=u;
      for(int h=0;h<NH;h++){vDSP_vsdiv(att+h*T,1,&sq,att+h*T,1,T);softmax(att+h*T,T);} u=now();tt[2]+=u-t;t=u;
      cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,NH,KVD,T,1,att,T,vc,KVD,0,ho,KVD); u=now();tt[3]+=u-t;t=u;
      /* alternative: scores as sgemm with transposed output: sc_t (NH x T) = qb^T (NH x KVD) K^T */
      cblas_sgemm(CblasRowMajor,CblasTrans,CblasTrans,NH,T,KVD,1,qb,NH,kc,KVD,0,att,T); u=now();tt[4]+=u-t;t=u;
      /* alternative: gemv over whole K giving sum over all heads (bandwidth probe) */
      cblas_sgemv(CblasRowMajor,CblasNoTrans,T,KVD,1,kc,KVD,q,1,0,sc,1); u=now();tt[5]+=u-t;
  }
  const char*nm[]={"K@qblk sgemm","mtrans","softmax x12","att@V sgemm","alt: qblk^T K^T sgemm","probe: K gemv"};
  for(int i=0;i<6;i++)printf("%-24s %.3f ms/token\n",nm[i],tt[i]/20*1e3);
}
