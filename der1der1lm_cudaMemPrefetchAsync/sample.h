#ifdef SAMPLE_PRE_LOOP
// START SAMPLE_PRE_LOOP

#include "zTileSize.h"
#ifndef __NVCC__
extern float* ch1dxx;
extern float* ch1dyy;
extern float* ch1dzz;
extern float* ch1dxy;
extern float* ch1dyz;
extern float* ch1dxz;
extern float* v2px;
extern float* v2pz;
extern float* v2sz;
extern float* v2pn;
extern float* pDx;
extern float* pDy;
extern float* qDx;
extern float* qDy;
#endif

const int strideX=ind(1,0,0)-ind(0,0,0);
const int strideY=ind(0,1,0)-ind(0,0,0);
const int strideZ=ind(0,0,1)-ind(0,0,0);

const float dxxinv=1.0f/(dx*dx);
const float dyyinv=1.0f/(dy*dy);
const float dzzinv=1.0f/(dz*dz);
const float dxinv=1.0f/dx;
const float dyinv=1.0f/dy;
const float dzinv=1.0f/dz;

// END SAMPLE_PRE_LOOP
#endif

#ifdef SAMPLE_LOOP_1
pDx[i+itz]= Der1(pc, i, strideX, dxinv);
#endif

#ifdef SAMPLE_LOOP_2
pDy[i+itz]= Der1(pc, i, strideY, dyinv);
#endif

#ifdef SAMPLE_LOOP_3
qDx[i+itz]= Der1(qc, i, strideX, dxinv);
#endif

#ifdef SAMPLE_LOOP_4
qDy[i+itz]= Der1(qc, i, strideY, dyinv);
#endif

#ifdef SAMPLE_LOOP_5
// START ONE SAMPLE

// xy derivative of p

const float pxy=(L1*(pDx[i+itP0+  strideY]-pDx[i+itP0-  strideY])+
                 L2*(pDx[i+itP0+2*strideY]-pDx[i+itP0-2*strideY])+
                 L3*(pDx[i+itP0+3*strideY]-pDx[i+itP0-3*strideY])+
                 L4*(pDx[i+itP0+4*strideY]-pDx[i+itP0-4*strideY]))*(dyinv);

// xz derivative of p

const float pxz =(L1*(pDx[i+itP1]-pDx[i+itM1])+
                  L2*(pDx[i+itP2]-pDx[i+itM2])+
                  L3*(pDx[i+itP3]-pDx[i+itM3])+
                  L4*(pDx[i+itP4]-pDx[i+itM4]))*(dzinv);

// yz derivative of p

const float pyz =(L1*(pDy[i+itP1]-pDy[i+itM1])+
                  L2*(pDy[i+itP2]-pDy[i+itM2])+
                  L3*(pDy[i+itP3]-pDy[i+itM3])+
                  L4*(pDy[i+itP4]-pDy[i+itM4]))*(dzinv);

// second order derivatives of p

const float pxx= Der2(pc, i, strideX, dxxinv);
const float pyy= Der2(pc, i, strideY, dyyinv);
const float pzz= Der2(pc, i, strideZ, dzzinv);

// H1(p) and H2(p)

const float cpxx=ch1dxx[i]*pxx;
const float cpyy=ch1dyy[i]*pyy;
const float cpzz=ch1dzz[i]*pzz;
const float cpxy=ch1dxy[i]*pxy;
const float cpxz=ch1dxz[i]*pxz;
const float cpyz=ch1dyz[i]*pyz;
const float h1p=cpxx+cpyy+cpzz+cpxy+cpxz+cpyz;
const float h2p=pxx+pyy+pzz-h1p;

// xy derivative of q

const float qxy=(L1*(qDx[i+itP0+  strideY]-qDx[i+itP0-  strideY])+
                 L2*(qDx[i+itP0+2*strideY]-qDx[i+itP0-2*strideY])+
                 L3*(qDx[i+itP0+3*strideY]-qDx[i+itP0-3*strideY])+
                 L4*(qDx[i+itP0+4*strideY]-qDx[i+itP0-4*strideY]))*(dyinv);

// xz derivative of q

const float qxz =(L1*(qDx[i+itP1]-qDx[i+itM1])+
                  L2*(qDx[i+itP2]-qDx[i+itM2])+
                  L3*(qDx[i+itP3]-qDx[i+itM3])+
                  L4*(qDx[i+itP4]-qDx[i+itM4]))*(dzinv);

// yz derivative of q
const float qyz =(L1*(qDy[i+itP1]-qDy[i+itM1])+
                  L2*(qDy[i+itP2]-qDy[i+itM2])+
                  L3*(qDy[i+itP3]-qDy[i+itM3])+
                  L4*(qDy[i+itP4]-qDy[i+itM4]))*(dzinv);

// q second order derivatives

const float qxx= Der2(qc, i, strideX, dxxinv);
const float qyy= Der2(qc, i, strideY, dyyinv);
const float qzz= Der2(qc, i, strideZ, dzzinv);

// H1(q) and H2(q)

const float cqxx=ch1dxx[i]*qxx;
const float cqyy=ch1dyy[i]*qyy;
const float cqzz=ch1dzz[i]*qzz;
const float cqxy=ch1dxy[i]*qxy;
const float cqxz=ch1dxz[i]*qxz;
const float cqyz=ch1dyz[i]*qyz;
const float h1q=cqxx+cqyy+cqzz+cqxy+cqxz+cqyz;
const float h2q=qxx+qyy+qzz-h1q;

// p-q derivatives, H1(p-q) and H2(p-q)

const float h1pmq=h1p-h1q;
const float h2pmq=h2p-h2q;

// rhs of p and q equations

const float rhsp=v2px[i]*h2p + v2pz[i]*h1q + v2sz[i]*h1pmq;
const float rhsq=v2pn[i]*h2p + v2pz[i]*h1q - v2sz[i]*h2pmq;

// new p and q

pp[i]=2.0f*pc[i] - pp[i] + rhsp*dt*dt;
qp[i]=2.0f*qc[i] - qp[i] + rhsq*dt*dt;

// END ONE SAMPLE
#endif

#ifdef SAMPLE_LOOP_6
const int itz=indzTile(0,0,(iz+4)%DIM_Z_TILE)-ind(0,0,iz+4);
const int itP4=indzTile(0,0,(iz+4)%DIM_Z_TILE)-ind(0,0,iz);
const int itP3=indzTile(0,0,(iz+3)%DIM_Z_TILE)-ind(0,0,iz);
const int itP2=indzTile(0,0,(iz+2)%DIM_Z_TILE)-ind(0,0,iz);
const int itP1=indzTile(0,0,(iz+1)%DIM_Z_TILE)-ind(0,0,iz);
const int itP0=indzTile(0,0,(iz  )%DIM_Z_TILE)-ind(0,0,iz);
const int itM1=indzTile(0,0,(iz-1)%DIM_Z_TILE)-ind(0,0,iz);
const int itM2=indzTile(0,0,(iz-2)%DIM_Z_TILE)-ind(0,0,iz);
const int itM3=indzTile(0,0,(iz-3)%DIM_Z_TILE)-ind(0,0,iz);
const int itM4=indzTile(0,0,(iz-4)%DIM_Z_TILE)-ind(0,0,iz);
#endif


#ifdef SAMPLE_POST_LOOP
#endif

