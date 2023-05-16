#ifndef CWB_CORE_LIKELIHOOD_H
#define CWB_CORE_LIKELIHOOD_H

#include "wseries.hh"
#include "netcluster.hh"
#include "monster.hh"
#include "skymap.hh"
#include "detector.hh"
#include "wat.hh"
#ifndef __CINT__
#include "watsse.hh"
#include "watsse4.hh"
#endif

using namespace std;


long likelihoodWP(std::vector <netcluster> wc_List, size_t nIFO,
                  size_t skyMask_size, short *skyMask, std::vector<double> skyMaskCC,
                  skymap nSkyStat, skymap nSensitivity, skymap nAlignment, skymap nDisbalance,
                  skymap nLikelihood, skymap nNullEnergy, skymap nCorrEnergy, skymap nCorrelation,
                  skymap nEllipticity, skymap nPolarisation, skymap nNetIndex, skymap nAntenaPrior,
                  skymap nProbability, skymap tau,
                  monster wdmMRA, double netCC, bool EFEC,
                  double precision,double gamma, bool optim, double netRHO, double delta, double acor,
                  char mode, int lag, int iID, char *Search);

// Function to allocate memory, initialize to zero, and return the pointer
inline float* alloc_and_init(size_t size) {
    float* ptmp = (float*) _mm_malloc(size * sizeof(float), 32);
    std::fill(ptmp, ptmp + size, 0.0f);
    return ptmp;
}

// Function to free memory if container is not empty
inline void free_if_not_empty(std::vector<float*>& vec) {
    if (!vec.empty()) {
        _avx_free_ps(vec);
        vec.clear();
    }
}

inline void pnt_(float** q, float** p, short** m, int l, int n) {
// point 0-7 float pointers to first network pixel
    NETX(q[0] = (p[0] + m[0][l]*n);,
    q[1] = (p[1] + m[1][l]*n);,
    q[2] = (p[2] + m[2][l]*n);,
    q[3] = (p[3] + m[3][l]*n);,
    q[4] = (p[4] + m[4][l]*n);,
    q[5] = (p[5] + m[5][l]*n);,
    q[6] = (p[6] + m[6][l]*n);,
    q[7] = (p[7] + m[7][l]*n);)
    return;
}


inline wavearray<float> _avx_norm_ps(monster wdmMRA, float** p, float** q,
                                              std::vector<float*> &pAVX, int I) {
    wavearray<float> norm(NIFO+1);     // output array for packet norms
    float* g = norm.data+1; norm=0.;

#ifndef __CINT__

// return packet norm for each detector
    int i,j,k,n,m;
    int  M = abs(I);
    int II = abs(I*2);
    float o = 1.e-12;
    float* mk = pAVX[1];                 // pixel energy mask
    float* rn = pAVX[22];                // halo noise
    wavearray<float> tmp(NIFO);          // array to store data
    float* t = tmp.data; tmp=0.;
    float e,u,v;

    float am[4*8] _ALIGNED; __m128* _am = (__m128*)am;
    float  x[4*8] _ALIGNED; __m128*  _x = (__m128*)x;
    float  h[4*8] _ALIGNED; __m128*  _h = (__m128*)h;  // halo

    float* an = pAVX[17];                // M*4*NIFO array
    NETX(__m128* _a0 = (__m128*)(an+4*M*0);, __m128* _a1 = (__m128*)(an+4*M*1);,
    __m128* _a2 = (__m128*)(an+4*M*2);, __m128* _a3 = (__m128*)(an+4*M*3);,
    __m128* _a4 = (__m128*)(an+4*M*4);, __m128* _a5 = (__m128*)(an+4*M*5);,
    __m128* _a6 = (__m128*)(an+4*M*6);, __m128* _a7 = (__m128*)(an+4*M*7);)

    for(m=0; m<M; m++) {
        if(I>0) rn[m] = 0.;
        NETX(_a0[m] = _mm_set_ps(q[0][m],q[0][m],p[0][m],p[0][m]); q[0][M+m]=0; ,
        _a1[m] = _mm_set_ps(q[1][m],q[1][m],p[1][m],p[1][m]); q[1][M+m]=0; ,
        _a2[m] = _mm_set_ps(q[2][m],q[2][m],p[2][m],p[2][m]); q[2][M+m]=0; ,
        _a3[m] = _mm_set_ps(q[3][m],q[3][m],p[3][m],p[3][m]); q[3][M+m]=0; ,
        _a4[m] = _mm_set_ps(q[4][m],q[4][m],p[4][m],p[4][m]); q[4][M+m]=0; ,
        _a5[m] = _mm_set_ps(q[5][m],q[5][m],p[5][m],p[5][m]); q[5][M+m]=0; ,
        _a6[m] = _mm_set_ps(q[6][m],q[6][m],p[6][m],p[6][m]); q[6][M+m]=0; ,
        _a7[m] = _mm_set_ps(q[7][m],q[7][m],p[7][m],p[7][m]); q[7][M+m]=0; )
    }

    for(m=0; m<M; m++) {
        if(mk[m]<=0.) continue;

        int      J = wdmMRA.size(m)*2;
        float   cc = 0;
        float*   c = wdmMRA.getXTalk(m);
        __m128* _c = (__m128*)(c+4);

        NETX(u=p[0][m]; v=q[0][m]; _am[0]=_mm_set_ps(v,u,v,u); _x[0]=_mm_setzero_ps(); ,
        u=p[1][m]; v=q[1][m]; _am[1]=_mm_set_ps(v,u,v,u); _x[1]=_mm_setzero_ps(); ,
        u=p[2][m]; v=q[2][m]; _am[2]=_mm_set_ps(v,u,v,u); _x[2]=_mm_setzero_ps(); ,
        u=p[3][m]; v=q[3][m]; _am[3]=_mm_set_ps(v,u,v,u); _x[3]=_mm_setzero_ps(); ,
        u=p[4][m]; v=q[4][m]; _am[4]=_mm_set_ps(v,u,v,u); _x[4]=_mm_setzero_ps(); ,
        u=p[5][m]; v=q[5][m]; _am[5]=_mm_set_ps(v,u,v,u); _x[5]=_mm_setzero_ps(); ,
        u=p[6][m]; v=q[6][m]; _am[6]=_mm_set_ps(v,u,v,u); _x[6]=_mm_setzero_ps(); ,
        u=p[7][m]; v=q[7][m]; _am[7]=_mm_set_ps(v,u,v,u); _x[7]=_mm_setzero_ps(); )

        for(j=0; j<J; j+=2) {
            n = int(c[j*4]);
            NETX(_x[0]=_mm_add_ps(_x[0],_mm_mul_ps(_c[j],_a0[n]));,
            _x[1]=_mm_add_ps(_x[1],_mm_mul_ps(_c[j],_a1[n]));,
            _x[2]=_mm_add_ps(_x[2],_mm_mul_ps(_c[j],_a2[n]));,
            _x[3]=_mm_add_ps(_x[3],_mm_mul_ps(_c[j],_a3[n]));,
            _x[4]=_mm_add_ps(_x[4],_mm_mul_ps(_c[j],_a4[n]));,
            _x[5]=_mm_add_ps(_x[5],_mm_mul_ps(_c[j],_a5[n]));,
            _x[6]=_mm_add_ps(_x[6],_mm_mul_ps(_c[j],_a6[n]));,
            _x[7]=_mm_add_ps(_x[7],_mm_mul_ps(_c[j],_a7[n]));)
        }

        NETX(_h[0]=_mm_mul_ps(_x[0],_am[0]);,
        _h[1]=_mm_mul_ps(_x[1],_am[1]);,
        _h[2]=_mm_mul_ps(_x[2],_am[2]);,
        _h[3]=_mm_mul_ps(_x[3],_am[3]);,
        _h[4]=_mm_mul_ps(_x[4],_am[4]);,
        _h[5]=_mm_mul_ps(_x[5],_am[5]);,
        _h[6]=_mm_mul_ps(_x[6],_am[6]);,
        _h[7]=_mm_mul_ps(_x[7],_am[7]);)

        NETX(t[0]=h[ 0]+h[ 1]+h[ 2]+h[ 3]; t[0]=t[0]>0?t[0]:0; g[0]+=t[0]; ,
        t[1]=h[ 4]+h[ 5]+h[ 6]+h[ 7]; t[1]=t[1]>0?t[1]:0; g[1]+=t[1]; ,
        t[2]=h[ 8]+h[ 9]+h[10]+h[11]; t[2]=t[2]>0?t[2]:0; g[2]+=t[2]; ,
        t[3]=h[12]+h[13]+h[14]+h[15]; t[3]=t[3]>0?t[3]:0; g[3]+=t[3]; ,
        t[4]=h[16]+h[17]+h[18]+h[19]; t[4]=t[4]>0?t[4]:0; g[4]+=t[4]; ,
        t[5]=h[20]+h[21]+h[22]+h[23]; t[5]=t[5]>0?t[5]:0; g[5]+=t[5]; ,
        t[6]=h[24]+h[25]+h[26]+h[27]; t[6]=t[6]>0?t[6]:0; g[6]+=t[6]; ,
        t[7]=h[28]+h[29]+h[30]+h[31]; t[7]=t[7]>0?t[7]:0; g[7]+=t[7]; )

        if(I<0) continue;

        NETX(u=p[0][m]; v=q[0][m]; e=(u*u+v*v)/(t[0]+o); q[0][M+m]=(e>=1)?0:e; ,
        u=p[1][m]; v=q[1][m]; e=(u*u+v*v)/(t[1]+o); q[1][M+m]=(e>=1)?0:e; ,
        u=p[2][m]; v=q[2][m]; e=(u*u+v*v)/(t[2]+o); q[2][M+m]=(e>=1)?0:e; ,
        u=p[3][m]; v=q[3][m]; e=(u*u+v*v)/(t[3]+o); q[3][M+m]=(e>=1)?0:e; ,
        u=p[4][m]; v=q[4][m]; e=(u*u+v*v)/(t[4]+o); q[4][M+m]=(e>=1)?0:e; ,
        u=p[5][m]; v=q[5][m]; e=(u*u+v*v)/(t[5]+o); q[5][M+m]=(e>=1)?0:e; ,
        u=p[6][m]; v=q[6][m]; e=(u*u+v*v)/(t[6]+o); q[6][M+m]=(e>=1)?0:e; ,
        u=p[7][m]; v=q[7][m]; e=(u*u+v*v)/(t[7]+o); q[7][M+m]=(e>=1)?0:e; )

        NETX(u=x[ 0]+x[ 2]; v=x[ 1]+x[ 3]; rn[m]+=u*u+v*v; ,
        u=x[ 4]+x[ 6]; v=x[ 5]+x[ 7]; rn[m]+=u*u+v*v; ,
        u=x[ 8]+x[10]; v=x[ 9]+x[11]; rn[m]+=u*u+v*v; ,
        u=x[12]+x[14]; v=x[13]+x[15]; rn[m]+=u*u+v*v; ,
        u=x[16]+x[18]; v=x[17]+x[19]; rn[m]+=u*u+v*v; ,
        u=x[20]+x[22]; v=x[21]+x[23]; rn[m]+=u*u+v*v; ,
        u=x[24]+x[26]; v=x[25]+x[27]; rn[m]+=u*u+v*v; ,
        u=x[28]+x[30]; v=x[29]+x[31]; rn[m]+=u*u+v*v; )

    }

    for(n=1; n<=XIFO; n++) {                        // save norms
        if(I>0) {
            e = q[n-1][II+4]*q[n-1][II+4];            // TF-Domain SNR
            if(norm.data[n]<2.) norm.data[n]=2;
            q[n-1][II+5] = norm.data[n];              // save norms
            norm.data[n] = e/norm.data[n];            // detector {1:NIFO} SNR
        }
        norm.data[0] += norm.data[n];                // total SNR
    }
#endif
    return norm;
}

inline wavearray<float> _avx_norm_ps(float** p, float** q, float* ec, int I) {
// use GW norm from data packet
// p - GW array
// q - Data array
    float e;
    int II = abs(I*2);
    wavearray<float> norm(NIFO+1);          // array for packet norms
    float* nn = norm.data;                  // array for packet norms
    norm = 0;
    for(int n=1; n<=XIFO; n++) {            // save norms
        nn[n] = q[n-1][II+5];                // get data norms
        p[n-1][II+5] = nn[n];                // save norms
        e = p[n-1][II+4]*p[n-1][II+4];       // TF-Domain SNR
        nn[n] = e/nn[n];                     // detector {1:NIFO} SNR
        nn[0] += nn[n];                      // total SNR
        for(int i=0; i<I; i++) {             // save signal norms
            p[n-1][I+i] = ec[i]>0 ? q[n-1][I+i] : 0.;        // set signal norms
        }
    }
    return norm;
}

#endif