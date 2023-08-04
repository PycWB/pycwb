/*
# Copyright (C) 2019 Sergey Klimenko, Gabriele Vedovato
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/


// Wavelet Analysis Tool
// S.Klimenko, University of Florida
// library of general functions

#ifndef WATAVX_HH
#define WATAVX_HH

#include <xmmintrin.h>
#include <pmmintrin.h>    // need for hadd
#include <immintrin.h>    // AVX
//#include <x86intrin.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <TMath.h>
#include "wat.hh"
#include "wavearray.hh"

// WAT AVX functions


static inline void _avx_free_ps(std::vector<float*> &v) {
   for(int n=0; n<v.size(); n++) _mm_free(v[n]); 
   v.clear(); std::vector<float*>().swap(v);
}
static inline  float _wat_hsum(__m128 y) {
   float x[4]; _mm_storeu_ps(x,y);
   return x[0]+x[1]+x[2]+x[3];
}

static inline void _avx_cpf_ps(float** p, float ** q,
			       float** u, float ** v, int I) {
// copy arrays from p-->u and from q-->v
   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; , 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; , 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; , 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; , 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; , 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; , 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; , 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; ) 

   NETX(__m128* _u0 = (__m128*)u[0]; __m128* _v0 = (__m128*)v[0]; , 
	__m128* _u1 = (__m128*)u[1]; __m128* _v1 = (__m128*)v[1]; , 
	__m128* _u2 = (__m128*)u[2]; __m128* _v2 = (__m128*)v[2]; , 
	__m128* _u3 = (__m128*)u[3]; __m128* _v3 = (__m128*)v[3]; , 
	__m128* _u4 = (__m128*)u[4]; __m128* _v4 = (__m128*)v[4]; , 
	__m128* _u5 = (__m128*)u[5]; __m128* _v5 = (__m128*)v[5]; , 
	__m128* _u6 = (__m128*)u[6]; __m128* _v6 = (__m128*)v[6]; , 
	__m128* _u7 = (__m128*)u[7]; __m128* _v7 = (__m128*)v[7]; ) 

   for(int i=0; i<I; i+=4) { 
      NETX(*_u0++ = *_p0++; *_v0++ = *_q0++; ,
	   *_u1++ = *_p1++; *_v1++ = *_q1++; ,                    
	   *_u2++ = *_p2++; *_v2++ = *_q2++; ,  
	   *_u3++ = *_p3++; *_v3++ = *_q3++; ,  
	   *_u4++ = *_p4++; *_v4++ = *_q4++; ,  
	   *_u5++ = *_p5++; *_v5++ = *_q5++; ,  
	   *_u6++ = *_p6++; *_v6++ = *_q6++; ,  
	   *_u7++ = *_p7++; *_v7++ = *_q7++; )
	 }
   return;
}                    


static inline float _avx_dpf_ps(double** Fp, double** Fx, int l,
			       std::vector<float*> &pAPN,
			       std::vector<float*> &pAVX, int I) {
// calculate dpf sin and cos and antenna f+ and fx amplitudes
// Fp, Fx array of pointers for antenna pattrens
// l - sky location index
// pRMS - vector with noise rms data
// pAVX - vector with pixel statistics  
// in likelihoodWP these arrays should be stored exactly in the same order.
// this function increments pointers stored in tne input pointer arrays.  
   __m128* _fp = (__m128*)pAVX[2];       
   __m128* _fx = (__m128*)pAVX[3];
   __m128* _si = (__m128*)pAVX[4];
   __m128* _co = (__m128*)pAVX[5];
   __m128* _ni = (__m128*)pAVX[18];
   __m128 _ff,_FF,_fF,_AP,_cc,_ss,_nn,_cn,_sn;
   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _o = _mm_set1_ps(0.0001);
   static const __m128 sm = _mm_set1_ps(-0.f);          // -0.f = 1 << 31
                __m128 _NI= _mm_setzero_ps();
                __m128 _NN= _mm_setzero_ps();

   NETX(__m128* _f0=(__m128*)pAPN[0]; __m128* _F0=(__m128*)(pAPN[0]+I); __m128* _r0=(__m128*)(pAPN[0]+2*I);,
	__m128* _f1=(__m128*)pAPN[1]; __m128* _F1=(__m128*)(pAPN[1]+I); __m128* _r1=(__m128*)(pAPN[1]+2*I);,
	__m128* _f2=(__m128*)pAPN[2]; __m128* _F2=(__m128*)(pAPN[2]+I); __m128* _r2=(__m128*)(pAPN[2]+2*I);,
	__m128* _f3=(__m128*)pAPN[3]; __m128* _F3=(__m128*)(pAPN[3]+I); __m128* _r3=(__m128*)(pAPN[3]+2*I);,
	__m128* _f4=(__m128*)pAPN[4]; __m128* _F4=(__m128*)(pAPN[4]+I); __m128* _r4=(__m128*)(pAPN[4]+2*I);,
	__m128* _f5=(__m128*)pAPN[5]; __m128* _F5=(__m128*)(pAPN[5]+I); __m128* _r5=(__m128*)(pAPN[5]+2*I);,
	__m128* _f6=(__m128*)pAPN[6]; __m128* _F6=(__m128*)(pAPN[6]+I); __m128* _r6=(__m128*)(pAPN[6]+2*I);,
	__m128* _f7=(__m128*)pAPN[7]; __m128* _F7=(__m128*)(pAPN[7]+I); __m128* _r7=(__m128*)(pAPN[7]+2*I);)

   NETX(__m128 _Fp0 = _mm_set1_ps(*(Fp[0]+l)); __m128 _Fx0 = _mm_set1_ps(*(Fx[0]+l)); , 
	__m128 _Fp1 = _mm_set1_ps(*(Fp[1]+l)); __m128 _Fx1 = _mm_set1_ps(*(Fx[1]+l)); ,  
	__m128 _Fp2 = _mm_set1_ps(*(Fp[2]+l)); __m128 _Fx2 = _mm_set1_ps(*(Fx[2]+l)); ,  
	__m128 _Fp3 = _mm_set1_ps(*(Fp[3]+l)); __m128 _Fx3 = _mm_set1_ps(*(Fx[3]+l)); ,  
	__m128 _Fp4 = _mm_set1_ps(*(Fp[4]+l)); __m128 _Fx4 = _mm_set1_ps(*(Fx[4]+l)); ,  
	__m128 _Fp5 = _mm_set1_ps(*(Fp[5]+l)); __m128 _Fx5 = _mm_set1_ps(*(Fx[5]+l)); ,  
	__m128 _Fp6 = _mm_set1_ps(*(Fp[6]+l)); __m128 _Fx6 = _mm_set1_ps(*(Fx[6]+l)); ,  
        __m128 _Fp7 = _mm_set1_ps(*(Fp[7]+l)); __m128 _Fx7 = _mm_set1_ps(*(Fx[7]+l)); )  

   float sign=0;
   NETX(if(*(pAPN[0]+2*I)>0.) sign+=*(Fp[0]+l) * *(Fx[0]+l);, 
	if(*(pAPN[1]+2*I)>0.) sign+=*(Fp[1]+l) * *(Fx[1]+l);,  
	if(*(pAPN[2]+2*I)>0.) sign+=*(Fp[2]+l) * *(Fx[2]+l);,  
	if(*(pAPN[3]+2*I)>0.) sign+=*(Fp[3]+l) * *(Fx[3]+l);,  
	if(*(pAPN[4]+2*I)>0.) sign+=*(Fp[4]+l) * *(Fx[4]+l);,  
	if(*(pAPN[5]+2*I)>0.) sign+=*(Fp[5]+l) * *(Fx[5]+l);,  
	if(*(pAPN[6]+2*I)>0.) sign+=*(Fp[6]+l) * *(Fx[6]+l);,  
        if(*(pAPN[7]+2*I)>0.) sign+=*(Fp[7]+l) * *(Fx[7]+l);)  
   sign = sign>0 ? 1. : -1.;
   __m128 _sign = _mm_set1_ps(sign);
     
   for(int i=0; i<I; i+=4) { 
      NETX(
	   *_f0 = _mm_mul_ps(*_r0,_Fp0); *_F0 = _mm_mul_ps(*_r0++,_Fx0);
	    _ff = _mm_mul_ps(*_f0,*_f0); 
	    _FF = _mm_mul_ps(*_F0,*_F0); 
	    _fF = _mm_mul_ps(*_f0,*_F0); ,
	   
	   *_f1 = _mm_mul_ps(*_r1,_Fp1); *_F1 = _mm_mul_ps(*_r1++,_Fx1);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f1,*_f1));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F1,*_F1));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f1,*_F1)); ,

	   *_f2 = _mm_mul_ps(*_r2,_Fp2); *_F2 = _mm_mul_ps(*_r2++,_Fx2);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f2,*_f2));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F2,*_F2));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f2,*_F2)); ,

	   *_f3 = _mm_mul_ps(*_r3,_Fp3); *_F3 = _mm_mul_ps(*_r3++,_Fx3);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f3,*_f3));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F3,*_F3));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f3,*_F3)); ,

	   *_f4 = _mm_mul_ps(*_r4,_Fp4); *_F4 = _mm_mul_ps(*_r4++,_Fx4);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f4,*_f4));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F4,*_F4));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f4,*_F4)); ,

	   *_f5 = _mm_mul_ps(*_r5,_Fp5); *_F5 = _mm_mul_ps(*_r5++,_Fx5);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f5,*_f5));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F5,*_F5));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f5,*_F5)); ,

	   *_f6 = _mm_mul_ps(*_r6,_Fp6); *_F6 = _mm_mul_ps(*_r6++,_Fx6);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f6,*_f6));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F6,*_F6));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f6,*_F6)); ,

	   *_f7 = _mm_mul_ps(*_r7,_Fp7); *_F7 = _mm_mul_ps(*_r7++,_Fx7);
	    _ff = _mm_add_ps(_ff,_mm_mul_ps(*_f7,*_f7));                        
	    _FF = _mm_add_ps(_FF,_mm_mul_ps(*_F7,*_F7));                        
	    _fF = _mm_add_ps(_fF,_mm_mul_ps(*_f7,*_F7)); )

      *_si = _mm_mul_ps(_fF,_2);                              // rotation 2*sin*cos*norm
      *_co = _mm_sub_ps(_ff,_FF);                             // rotation (cos^2-sin^2)*norm
       _AP = _mm_add_ps(_ff,_FF);                             // total antenna norm
       _cc = _mm_mul_ps(*_co,*_co);
       _ss = _mm_mul_ps(*_si,*_si);
       _nn = _mm_sqrt_ps(_mm_add_ps(_cc,_ss));                // co/si norm
      *_fp = _mm_div_ps(_mm_add_ps(_AP,_nn),_2);              // |f+|^2 
       _cc = _mm_div_ps(*_co,_mm_add_ps(_nn,_o));             // cos(2p)
       _nn = _mm_and_ps(_mm_cmpgt_ps(*_si,_0),_1);            // 1 if sin(2p)>0. or 0 if sin(2p)<0.  
       _ss = _mm_sub_ps(_mm_mul_ps(_2,_nn),_1);               // 1 if sin(2p)>0. or-1 if sin(2p)<0.  
      *_si = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(_1,_cc),_2));  // |sin(p)|
      *_co = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_1,_cc),_2));  // |cos(p)|
      *_co = _mm_mul_ps(*_co,_ss);                            // cos(p)

// DPF antenna patterns
   NETX(                                                    
	_ff = _mm_add_ps(_mm_mul_ps(*_f0,*_co),_mm_mul_ps(*_F0,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F0,*_co),_mm_mul_ps(*_f0,*_si)); *_f0 = _ff; *_F0 = _FF;       // fx
	_nn = _mm_mul_ps(_cc,_cc); _fF = _mm_mul_ps(_ff,_FF);,                                       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f1,*_co),_mm_mul_ps(*_F1,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F1,*_co),_mm_mul_ps(*_f1,*_si)); *_f1 = _ff; *_F1 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f2,*_co),_mm_mul_ps(*_F2,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F2,*_co),_mm_mul_ps(*_f2,*_si)); *_f2 = _ff; *_F2 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f3,*_co),_mm_mul_ps(*_F3,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F3,*_co),_mm_mul_ps(*_f3,*_si)); *_f3 = _ff; *_F3 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f4,*_co),_mm_mul_ps(*_F4,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F4,*_co),_mm_mul_ps(*_f4,*_si)); *_f4 = _ff; *_F4 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f5,*_co),_mm_mul_ps(*_F5,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F5,*_co),_mm_mul_ps(*_f5,*_si)); *_f5 = _ff; *_F5 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f6,*_co),_mm_mul_ps(*_F6,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F6,*_co),_mm_mul_ps(*_f6,*_si)); *_f6 = _ff; *_F6 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));,       // ni 

	_ff = _mm_add_ps(_mm_mul_ps(*_f7,*_co),_mm_mul_ps(*_F7,*_si)); _cc = _mm_mul_ps(_ff,_ff);    // f+ 
	_FF = _mm_sub_ps(_mm_mul_ps(*_F7,*_co),_mm_mul_ps(*_f7,*_si)); *_f7 = _ff; *_F7 = _FF;       // fx
	_nn = _mm_add_ps(_nn,_mm_mul_ps(_cc,_cc)); _fF = _mm_add_ps(_fF,_mm_mul_ps(_ff,_FF));)       // ni 

   _fF = _mm_div_ps(_fF,_mm_add_ps(*_fp,_o));

   NETX(                                                 
	*_F0 = _mm_sub_ps(*_F0,_mm_mul_ps(*_f0++,_fF)); *_fx = _mm_mul_ps(*_F0,*_F0);                  _F0++;,
	*_F1 = _mm_sub_ps(*_F1,_mm_mul_ps(*_f1++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F1,*_F1)); _F1++;,
	*_F2 = _mm_sub_ps(*_F2,_mm_mul_ps(*_f2++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F2,*_F2)); _F2++;,
	*_F3 = _mm_sub_ps(*_F3,_mm_mul_ps(*_f3++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F3,*_F3)); _F3++;,
	*_F4 = _mm_sub_ps(*_F4,_mm_mul_ps(*_f4++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F4,*_F4)); _F4++;,
	*_F5 = _mm_sub_ps(*_F5,_mm_mul_ps(*_f5++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F5,*_F5)); _F5++;,
	*_F6 = _mm_sub_ps(*_F6,_mm_mul_ps(*_f6++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F6,*_F6)); _F6++;,
	*_F7 = _mm_sub_ps(*_F7,_mm_mul_ps(*_f7++,_fF)); *_fx = _mm_add_ps(*_fx,_mm_mul_ps(*_F7,*_F7)); _F7++;)

      *_ni = _mm_div_ps(_nn,_mm_add_ps(_mm_mul_ps(*_fp,*_fp),_o));     // network index
       _ff = _mm_add_ps(_mm_mul_ps(_1,*_ni),_o);                       // network index
       _NI = _mm_add_ps(_NI,(_mm_div_ps(*_fx,_ff)));                   // sum of |fx|^2/2/ni
       _NN = _mm_add_ps(_NN,_mm_and_ps(_mm_cmpgt_ps(*_fp,_0),_1));     // pixel count

       _fp++; _fx++; _si++; _co++; _ni++;	 
   }
   return sqrt(_wat_hsum(_NI)/(_wat_hsum(_NN)+0.01));
} 


static inline float _avx_GW_ps(float** p, float ** q,    
			       std::vector<float*> &pAPN, float* rr,
			       std::vector<float*> &pAVX, int II) {
// initialize GW strain amplitudes and impose regulator
// p,q  - input - data vector
// pAVX - pixel statistics
// J    - number of AVX pixels
// in likelihoodWP these arrays should be stored exactly in the same order.
// this function updates the definition of the mask array:

   int I = abs(II);
   
   __m128* _et = (__m128*)pAVX[0];
   __m128* _mk = (__m128*)pAVX[1];
   __m128* _fp = (__m128*)pAVX[2];       
   __m128* _fx = (__m128*)pAVX[3];
   __m128* _au = (__m128*)pAVX[10];
   __m128* _AU = (__m128*)pAVX[11];
   __m128* _av = (__m128*)pAVX[12];
   __m128* _AV = (__m128*)pAVX[13];
   __m128* _ni = (__m128*)pAVX[18];

   __m128  _uu = _mm_setzero_ps();
   __m128  _UU = _mm_setzero_ps();
   __m128  _uU = _mm_setzero_ps();
   __m128  _vv = _mm_setzero_ps();
   __m128  _VV = _mm_setzero_ps();
   __m128  _vV = _mm_setzero_ps();
   __m128  _NN = _mm_setzero_ps();

   __m128 _h,_H,_f,_F,_R,_a,_nn,_m,_ff,_xp,_XP,_xx,_XX;
   float cu,su,cv,sv,cc,ss,nn,uu,UU,vv,VV,et,ET;
   static const __m128 _0  = _mm_set1_ps(0);
   static const __m128 _1  = _mm_set1_ps(1);
   static const __m128 o1  = _mm_set1_ps(0.1);
   static const __m128 _o  = _mm_set1_ps(1.e-5);
                __m128 _rr = _mm_set1_ps(rr[0]);
                __m128 _RR = _mm_set1_ps(rr[1]);

   // pointers to antenna patterns
   NETX(__m128* _f0=(__m128*)pAPN[0]; __m128* _F0=(__m128*)(pAPN[0]+I);,
	__m128* _f1=(__m128*)pAPN[1]; __m128* _F1=(__m128*)(pAPN[1]+I);,
	__m128* _f2=(__m128*)pAPN[2]; __m128* _F2=(__m128*)(pAPN[2]+I);,
	__m128* _f3=(__m128*)pAPN[3]; __m128* _F3=(__m128*)(pAPN[3]+I);,
	__m128* _f4=(__m128*)pAPN[4]; __m128* _F4=(__m128*)(pAPN[4]+I);,
	__m128* _f5=(__m128*)pAPN[5]; __m128* _F5=(__m128*)(pAPN[5]+I);,
	__m128* _f6=(__m128*)pAPN[6]; __m128* _F6=(__m128*)(pAPN[6]+I);,
	__m128* _f7=(__m128*)pAPN[7]; __m128* _F7=(__m128*)(pAPN[7]+I);)

   // pointers to data
   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0];, 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1];, 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2];, 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3];, 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4];, 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5];, 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6];, 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7];) 

   for(int i=0; i<I; i+=4) {                                 // calculate scalar products 

      NETX(                                                 
	   _xp = _mm_mul_ps(*_f0,*_p0);                      // (x,f+)
	   _XP = _mm_mul_ps(*_f0,*_q0);                      // (X,f+)
	   _xx = _mm_mul_ps(*_F0,*_p0);                      // (x,fx)
	   _XX = _mm_mul_ps(*_F0,*_q0);                  ,   // (X,fx)
	   			                            
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f1,*_p1));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f1,*_q1));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F1,*_p1));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F1,*_q1));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f2,*_p2));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f2,*_q2));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F2,*_p2));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F2,*_q2));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f3,*_p3));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f3,*_q3));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F3,*_p3));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F3,*_q3));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f4,*_p4));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f4,*_q4));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F4,*_p4));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F4,*_q4));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f5,*_p5));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f5,*_q5));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F5,*_p5));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F5,*_q5));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f6,*_p6));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f6,*_q6));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F6,*_p6));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F6,*_q6));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f7,*_p7));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f7,*_q7));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F7,*_p7));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F7,*_q7));  )   // (X,fx)
	 
// regulator

      _f = _mm_add_ps(_mm_mul_ps(_xp,_xp),_mm_mul_ps(_XP,_XP));        // f=(x,f+)^2+(X,f+)^2
      _f = _mm_div_ps(_f,_mm_add_ps(*_et,_o));                         // f=f/[ni*(|x|^2+|X|^2)]
      _f = _mm_sqrt_ps(_mm_mul_ps(*_ni,_f));                           // f=ni*sqrt[(x,f+)^2+(X,f+)^2/(|x|^2+|X|^2)]
      _f = _mm_sub_ps(_mm_mul_ps(_f,_rr),*_fp);                        // f*R-|f+|^2
      _f = _mm_mul_ps(_f,_mm_and_ps(_mm_cmpgt_ps(_f,_0),_1));          // f if f*R>|f+|^2 or 0 if f*R<|f+|^2
      _f = _mm_div_ps(*_mk,_mm_add_ps(*_fp,_mm_add_ps(_o,_f)));        // 1 / {|f+|^2+epsilon}

      _h = _mm_mul_ps(_xp,_f);                                         // use inequality
      _H = _mm_mul_ps(_XP,_f);                                         // fx^2 *(s^2+e^2*c^2)/(1+e^2) < fx^2
      _h = _mm_add_ps(_mm_mul_ps(_h,_h),_mm_mul_ps(_H,_H));
      _H = _mm_add_ps(_mm_mul_ps(_xx,_xx),_mm_mul_ps(_XX,_XX));        // (x,fx)^2+(X,fx)^2
      _F = _mm_sqrt_ps(_mm_div_ps(_H,_mm_add_ps(_h,_o)));
      _R = _mm_add_ps(o1,_mm_div_ps(_RR,_mm_add_ps(*_et,_o)));         // dynamic x-regulator
      _F = _mm_sub_ps(_mm_mul_ps(_F,_R),*_fx);                         // F*R-|fx|^2
      _F = _mm_mul_ps(_F,_mm_and_ps(_mm_cmpgt_ps(_F,_0),_1));          // F*R-|fx|^2 if F*R>|fx|^2 or 0 if F*R<|fx|^2
      _F = _mm_div_ps(*_mk,_mm_add_ps(*_fx,_mm_add_ps(_o,_F)));        // 1/ {|fx|^2+epsilon}

      *_au = _mm_mul_ps(_xp,_f);
      *_AU = _mm_mul_ps(_XP,_f);
      *_av = _mm_mul_ps(_xx,_F);
      *_AV = _mm_mul_ps(_XX,_F);

        _a = _mm_add_ps(_mm_mul_ps(_f,*_fp),_mm_mul_ps(_F,*_fx));       // Gaussin noise correction
       _NN = _mm_add_ps(_NN,*_mk);                                      // number of pixels
      *_mk = _mm_add_ps(_a,_mm_sub_ps(*_mk,_1));                        // -1 - rejected, >=0 accepted

      _mk++; _et++; _fp++; _fx++; _ni++;

// set GW amplitudes in wavelet domain
      NETX(*_p0++ = _mm_add_ps(_mm_mul_ps(*_f0,  *_au),_mm_mul_ps(*_F0  ,*_av));
	   *_q0++ = _mm_add_ps(_mm_mul_ps(*_f0++,*_AU),_mm_mul_ps(*_F0++,*_AV)); ,
           *_p1++ = _mm_add_ps(_mm_mul_ps(*_f1  ,*_au),_mm_mul_ps(*_F1  ,*_av));
	   *_q1++ = _mm_add_ps(_mm_mul_ps(*_f1++,*_AU),_mm_mul_ps(*_F1++,*_AV)); ,
           *_p2++ = _mm_add_ps(_mm_mul_ps(*_f2  ,*_au),_mm_mul_ps(*_F2  ,*_av));
	   *_q2++ = _mm_add_ps(_mm_mul_ps(*_f2++,*_AU),_mm_mul_ps(*_F2++,*_AV)); ,
           *_p3++ = _mm_add_ps(_mm_mul_ps(*_f3  ,*_au),_mm_mul_ps(*_F3  ,*_av));
	   *_q3++ = _mm_add_ps(_mm_mul_ps(*_f3++,*_AU),_mm_mul_ps(*_F3++,*_AV)); ,
           *_p4++ = _mm_add_ps(_mm_mul_ps(*_f4  ,*_au),_mm_mul_ps(*_F4  ,*_av));
	   *_q4++ = _mm_add_ps(_mm_mul_ps(*_f4++,*_AU),_mm_mul_ps(*_F4++,*_AV)); ,
           *_p5++ = _mm_add_ps(_mm_mul_ps(*_f5  ,*_au),_mm_mul_ps(*_F5  ,*_av));
	   *_q5++ = _mm_add_ps(_mm_mul_ps(*_f5++,*_AU),_mm_mul_ps(*_F5++,*_AV)); ,
           *_p6++ = _mm_add_ps(_mm_mul_ps(*_f6  ,*_au),_mm_mul_ps(*_F6  ,*_av));
	   *_q6++ = _mm_add_ps(_mm_mul_ps(*_f6++,*_AU),_mm_mul_ps(*_F6++,*_AV)); ,
           *_p7++ = _mm_add_ps(_mm_mul_ps(*_f7  ,*_au),_mm_mul_ps(*_F7  ,*_av));
	   *_q7++ = _mm_add_ps(_mm_mul_ps(*_f7++,*_AU),_mm_mul_ps(*_F7++,*_AV)); )

      _uU = _mm_add_ps(_uU,_mm_mul_ps(*_au,*_AU));
      _vV = _mm_add_ps(_vV,_mm_mul_ps(*_av,*_AV));
      _uu = _mm_add_ps(_uu,_mm_mul_ps(*_au,*_au)); _au++;                       
      _UU = _mm_add_ps(_UU,_mm_mul_ps(*_AU,*_AU)); _AU++;                       
      _vv = _mm_add_ps(_vv,_mm_mul_ps(*_av,*_av)); _av++;                       
      _VV = _mm_add_ps(_VV,_mm_mul_ps(*_AV,*_AV)); _AV++;                       
   }	   
   su = _wat_hsum(_uU);  sv = _wat_hsum(_vV);                 // rotation sin*cos*norm
   uu = _wat_hsum(_uu);  vv = _wat_hsum(_vv);                 // 00 energy
   UU = _wat_hsum(_UU);  VV = _wat_hsum(_VV);                 // 90 energy

// first packet
   nn = sqrt((uu-UU)*(uu-UU)+4*su*su)+0.0001;                 // co/si norm
   cu = (uu-UU)/nn; et=uu+UU;                                 // rotation cos(2p) and sin(2p)
   uu = sqrt((et+nn)/2); UU = et>nn?sqrt((et-nn)/2):0;        // amplitude of first/second component
   nn = su>0 ? 1 : -1;                                        // norm^2 of 2*cos^2 and 2*sin*cos
   su = sqrt((1-cu)/2); cu = nn*sqrt((1+cu)/2);               // normalized rotation sin/cos

// second packet
   nn = sqrt((vv-VV)*(vv-VV)+4*sv*sv)+0.0001;                 // co/si norm
   cv = (vv-VV)/nn; ET=vv+VV;                                 // rotation cos(2p) and sin(2p)
   vv = sqrt((ET+nn)/2); VV = et>nn?sqrt((ET-nn)/2):0;        // first/second component energy
   nn = sv>0 ? 1 : -1;                                        // norm^2 of 2*cos^2 and 2*sin*cos
   sv = sqrt((1-cv)/2); cv = nn*sqrt((1+cv)/2);               // normalized rotation sin/cos// first packet

   //return _mm_set_ps(ET,et,VV+vv,UU+uu);                    // reversed order
   return _wat_hsum(_NN);                                     // returns number of pixels above threshold
} 


static inline void _avx_pol_ps(float** p, float ** q, 
                               wavearray<double>* pol00, wavearray<double>* pol90,
		               std::vector<float*> &pAPN,
			       std::vector<float*> &pAVX, int II) {
// calculates the polar coordinates of the input vector v in the DPF frame 
// p,q  - input/output - data vector 
// pol00 - output - 00 component in polar coordinates (pol00[0] : radius, pol00[1] : angle in radians)
// pol90 - output - 90 component in polar coordinates (pol90[0] : radius, pol90[1] : angle in radians)
// pRMS - vector with noise rms data
// pAVX - pixel statistics
// II   - number of AVX pixels
// in likelihoodWP these arrays should be stored exactly in the same order.

   int I = abs(II);
   
   __m128* _MK = (__m128*)pAVX[1];
   __m128* _fp = (__m128*)pAVX[2];       
   __m128* _fx = (__m128*)pAVX[3];

   __m128 _xp,_XP,_xx,_XX,_rr,_RR,_mk;

   static const __m128 _0  = _mm_set1_ps(0);
   static const __m128 _1  = _mm_set1_ps(1);
   static const __m128 _o  = _mm_set1_ps(1.e-5);

   __m128 _ss,_cc,_SS,_CC;
   float rpol[4],cpol[4],spol[4];
   float RPOL[4],CPOL[4],SPOL[4];

   double* r = pol00[0].data;
   double* a = pol00[1].data;
   double* R = pol90[0].data;
   double* A = pol90[1].data;

   // pointers to antenna patterns
   NETX(__m128* _f0=(__m128*)pAPN[0]; __m128* _F0=(__m128*)(pAPN[0]+I);,
	__m128* _f1=(__m128*)pAPN[1]; __m128* _F1=(__m128*)(pAPN[1]+I);,
	__m128* _f2=(__m128*)pAPN[2]; __m128* _F2=(__m128*)(pAPN[2]+I);,
	__m128* _f3=(__m128*)pAPN[3]; __m128* _F3=(__m128*)(pAPN[3]+I);,
	__m128* _f4=(__m128*)pAPN[4]; __m128* _F4=(__m128*)(pAPN[4]+I);,
	__m128* _f5=(__m128*)pAPN[5]; __m128* _F5=(__m128*)(pAPN[5]+I);,
	__m128* _f6=(__m128*)pAPN[6]; __m128* _F6=(__m128*)(pAPN[6]+I);,
	__m128* _f7=(__m128*)pAPN[7]; __m128* _F7=(__m128*)(pAPN[7]+I);)

   // pointers to data
   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0];, 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1];, 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2];, 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3];, 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4];, 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5];, 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6];, 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7];) 

   int m=0;
   for(int i=0; i<I; i+=4) {                                 

// Compute scalar products 

      _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1); // event mask - apply energy threshold En

      NETX(                                                 
	   _xp = _mm_mul_ps(*_f0,_mm_mul_ps(_mk,*_p0));                      // (x,f+)
	   _XP = _mm_mul_ps(*_f0,_mm_mul_ps(_mk,*_q0));                      // (X,f+)
	   _xx = _mm_mul_ps(*_F0,_mm_mul_ps(_mk,*_p0));                      // (x,fx)
	   _XX = _mm_mul_ps(*_F0,_mm_mul_ps(_mk,*_q0));                  ,   // (X,fx)
	   			                            
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f1,_mm_mul_ps(_mk,*_p1)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f1,_mm_mul_ps(_mk,*_q1)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F1,_mm_mul_ps(_mk,*_p1)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F1,_mm_mul_ps(_mk,*_q1)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f2,_mm_mul_ps(_mk,*_p2)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f2,_mm_mul_ps(_mk,*_q2)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F2,_mm_mul_ps(_mk,*_p2)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F2,_mm_mul_ps(_mk,*_q2)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f3,_mm_mul_ps(_mk,*_p3)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f3,_mm_mul_ps(_mk,*_q3)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F3,_mm_mul_ps(_mk,*_p3)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F3,_mm_mul_ps(_mk,*_q3)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f4,_mm_mul_ps(_mk,*_p4)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f4,_mm_mul_ps(_mk,*_q4)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F4,_mm_mul_ps(_mk,*_p4)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F4,_mm_mul_ps(_mk,*_q4)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f5,_mm_mul_ps(_mk,*_p5)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f5,_mm_mul_ps(_mk,*_q5)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F5,_mm_mul_ps(_mk,*_p5)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F5,_mm_mul_ps(_mk,*_q5)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f6,_mm_mul_ps(_mk,*_p6)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f6,_mm_mul_ps(_mk,*_q6)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F6,_mm_mul_ps(_mk,*_p6)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F6,_mm_mul_ps(_mk,*_q6)));  ,   // (X,fx)
	   		                          
	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f7,_mm_mul_ps(_mk,*_p7)));      // (x,f+)
	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f7,_mm_mul_ps(_mk,*_q7)));      // (X,f+)
	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F7,_mm_mul_ps(_mk,*_p7)));      // (x,fx)
	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F7,_mm_mul_ps(_mk,*_q7)));  )   // (X,fx)
	 
// 00/90 components in polar coordinates (pol00/90[0] : radius, pol00/90[1] : angle in radians)

      _cc = _mm_div_ps(_xp,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));     // (x,f+) / {|f+|+epsilon}
      _ss = _mm_div_ps(_xx,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));     // (x,fx) / {|fx|+epsilon}

      _rr = _mm_div_ps(_mm_mul_ps(_xp,_xp),_mm_add_ps(*_fp,_o));
      _rr = _mm_add_ps(_rr,_mm_div_ps(_mm_mul_ps(_xx,_xx),_mm_add_ps(*_fx,_o)));

      _mm_storeu_ps(cpol,_cc);					   // cos
      _mm_storeu_ps(spol,_ss);					   // sin
      _mm_storeu_ps(rpol,_rr);                   		   // (x,x);        

      _CC = _mm_div_ps(_XP,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (X,f+) / {|f+|+epsilon}
      _SS = _mm_div_ps(_XX,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (X,fx) / {|fx|+epsilon}

      _RR = _mm_div_ps(_mm_mul_ps(_XP,_XP),_mm_add_ps(*_fp,_o));
      _RR = _mm_add_ps(_RR,_mm_div_ps(_mm_mul_ps(_XX,_XX),_mm_add_ps(*_fx,_o)));

      _mm_storeu_ps(CPOL,_CC);					   // cos
      _mm_storeu_ps(SPOL,_SS);					   // sin
      _mm_storeu_ps(RPOL,_RR);                   		   // (X,X);        

      for(int n=0;n<4;n++) {
        r[m] = sqrt(rpol[n]);                        		   // |x|
        a[m] = atan2(spol[n],cpol[n]);               		   // atan2(spol,cpol)
        R[m] = sqrt(RPOL[n]);                        		   // |X|
        A[m] = atan2(SPOL[n],CPOL[n]);               		   // atan2(SPOL,CPOL)
        m++;
      }

// PnP - Projection to network Plane

      _cc = _mm_div_ps(_cc,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (x,f+) / {|f+|^2+epsilon}
      _ss = _mm_div_ps(_ss,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (x,fx) / {|fx|^2+epsilon}

      _CC = _mm_div_ps(_CC,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (X,f+) / {|f+|^2+epsilon}
      _SS = _mm_div_ps(_SS,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (X,fx) / {|fx|^2+epsilon}

      NETX(*_p0 = _mm_add_ps(_mm_mul_ps(*_f0,_cc),_mm_mul_ps(*_F0,_ss));
	   *_q0 = _mm_add_ps(_mm_mul_ps(*_f0,_CC),_mm_mul_ps(*_F0,_SS)); ,
           *_p1 = _mm_add_ps(_mm_mul_ps(*_f1,_cc),_mm_mul_ps(*_F1,_ss));
	   *_q1 = _mm_add_ps(_mm_mul_ps(*_f1,_CC),_mm_mul_ps(*_F1,_SS)); ,
           *_p2 = _mm_add_ps(_mm_mul_ps(*_f2,_cc),_mm_mul_ps(*_F2,_ss));
	   *_q2 = _mm_add_ps(_mm_mul_ps(*_f2,_CC),_mm_mul_ps(*_F2,_SS)); ,
           *_p3 = _mm_add_ps(_mm_mul_ps(*_f3,_cc),_mm_mul_ps(*_F3,_ss));
	   *_q3 = _mm_add_ps(_mm_mul_ps(*_f3,_CC),_mm_mul_ps(*_F3,_SS)); ,
           *_p4 = _mm_add_ps(_mm_mul_ps(*_f4,_cc),_mm_mul_ps(*_F4,_ss));
	   *_q4 = _mm_add_ps(_mm_mul_ps(*_f4,_CC),_mm_mul_ps(*_F4,_SS)); ,
           *_p5 = _mm_add_ps(_mm_mul_ps(*_f5,_cc),_mm_mul_ps(*_F5,_ss));
	   *_q5 = _mm_add_ps(_mm_mul_ps(*_f5,_CC),_mm_mul_ps(*_F5,_SS)); ,
           *_p6 = _mm_add_ps(_mm_mul_ps(*_f6,_cc),_mm_mul_ps(*_F6,_ss));
	   *_q6 = _mm_add_ps(_mm_mul_ps(*_f6,_CC),_mm_mul_ps(*_F6,_SS)); ,
           *_p7 = _mm_add_ps(_mm_mul_ps(*_f7,_cc),_mm_mul_ps(*_F7,_ss));
	   *_q7 = _mm_add_ps(_mm_mul_ps(*_f7,_CC),_mm_mul_ps(*_F7,_SS)); )

// DSP - Dual Stream Phase Transform

      __m128 _N = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_cc,_cc),_mm_mul_ps(_CC,_CC)));
      
      _cc = _mm_div_ps(_cc,_mm_add_ps(_N,_o)); 			  // cos_dsp = N * (x,f+)/|f+|^2
      _CC = _mm_div_ps(_CC,_mm_add_ps(_N,_o)); 			  // sin_dsp = N * (X,f+)/|f+|^2

      __m128 _y,_Y;

      NETX(_y = _mm_add_ps(_mm_mul_ps(*_p0,_cc),_mm_mul_ps(*_q0,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q0,_cc),_mm_mul_ps(*_p0,_CC)); *_p0 = _y; *_q0 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p1,_cc),_mm_mul_ps(*_q1,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q1,_cc),_mm_mul_ps(*_p1,_CC)); *_p1 = _y; *_q1 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p2,_cc),_mm_mul_ps(*_q2,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q2,_cc),_mm_mul_ps(*_p2,_CC)); *_p2 = _y; *_q2 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p3,_cc),_mm_mul_ps(*_q3,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q3,_cc),_mm_mul_ps(*_p3,_CC)); *_p3 = _y; *_q3 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p4,_cc),_mm_mul_ps(*_q4,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q4,_cc),_mm_mul_ps(*_p4,_CC)); *_p4 = _y; *_q4 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p5,_cc),_mm_mul_ps(*_q5,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q5,_cc),_mm_mul_ps(*_p5,_CC)); *_p5 = _y; *_q5 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p6,_cc),_mm_mul_ps(*_q6,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q6,_cc),_mm_mul_ps(*_p6,_CC)); *_p6 = _y; *_q6 = _Y; ,
           _y = _mm_add_ps(_mm_mul_ps(*_p7,_cc),_mm_mul_ps(*_q7,_CC));
	   _Y = _mm_sub_ps(_mm_mul_ps(*_q7,_cc),_mm_mul_ps(*_p7,_CC)); *_p7 = _y; *_q7 = _Y; )


// Increment pointers

      NETX(                                                 
           _p0++;_q0++;_f0++;_F0++;	,
           _p1++;_q1++;_f1++;_F1++;	,
           _p2++;_q2++;_f2++;_F2++;	,
           _p3++;_q3++;_f3++;_F3++;	,
           _p4++;_q4++;_f4++;_F4++;	,
           _p5++;_q5++;_f5++;_F5++;	,
           _p6++;_q6++;_f6++;_F6++;	,
           _p7++;_q7++;_f7++;_F7++;	)

      _fp++;_fx++;
   }

   return; 
} 


static inline float _avx_loadata_ps(float** p, float** q, 
				    float** u, float** v, float En,
				    std::vector<float*> &pAVX, int I) {
// load data vectors p,q into temporary arrays u,v
// calculate total energy of network pixels stored in pAVX[5]
// apply energy threshold En, initialize pixel mask stored in pAVX[24]
// returns total energy         
// in likelihoodWP these arrays should be stored exactly in the same order.
// this function increments pointers stored in tne input pointer arrays.  
   __m128 _aa, _AA, _aA;

   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _o = _mm_set1_ps(1.e-12);
   static const __m128 sm = _mm_set1_ps(-0.f);          // -0.f = 1 << 31
   static const __m128 _En= _mm_set1_ps(En);

   __m128* _et = (__m128*)pAVX[0];
   __m128* _mk = (__m128*)pAVX[1];
   __m128  _ee = _mm_setzero_ps();
   __m128  _EE = _mm_setzero_ps();
   __m128  _NN = _mm_setzero_ps();

   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; , 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; , 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; , 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; , 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; , 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; , 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; , 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; ) 

   NETX(__m128* _u0 = (__m128*)u[0]; __m128* _v0 = (__m128*)v[0]; , 
	__m128* _u1 = (__m128*)u[1]; __m128* _v1 = (__m128*)v[1]; , 
	__m128* _u2 = (__m128*)u[2]; __m128* _v2 = (__m128*)v[2]; , 
	__m128* _u3 = (__m128*)u[3]; __m128* _v3 = (__m128*)v[3]; , 
	__m128* _u4 = (__m128*)u[4]; __m128* _v4 = (__m128*)v[4]; , 
	__m128* _u5 = (__m128*)u[5]; __m128* _v5 = (__m128*)v[5]; , 
	__m128* _u6 = (__m128*)u[6]; __m128* _v6 = (__m128*)v[6]; , 
	__m128* _u7 = (__m128*)u[7]; __m128* _v7 = (__m128*)v[7]; ) 

   for(int i=0; i<I; i+=4) { 
      NETX(
	   _aa = _mm_mul_ps(*_p0,*_p0);                 *_u0++ = *_p0++;        
	   _AA = _mm_mul_ps(*_q0,*_q0);                 *_v0++ = *_q0++; ,
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p1,*_p1)); *_u1++ = *_p1++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q1,*_q1)); *_v1++ = *_q1++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p2,*_p2)); *_u2++ = *_p2++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q2,*_q2)); *_v2++ = *_q2++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p3,*_p3)); *_u3++ = *_p3++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q3,*_q3)); *_v3++ = *_q3++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p4,*_p4)); *_u4++ = *_p4++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q4,*_q4)); *_v4++ = *_q4++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p5,*_p5)); *_u5++ = *_p5++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q5,*_q5)); *_v5++ = *_q5++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p6,*_p6)); *_u6++ = *_p6++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q6,*_q6)); *_v6++ = *_q6++; ,                    
	   _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p7,*_p7)); *_u7++ = *_p7++;                    
	   _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q7,*_q7)); *_v7++ = *_q7++; )                    

       	 
// calculate data network orthogonalization sin and cos

      *_et = _mm_add_ps(_mm_add_ps(_aa,_AA),_o);          // total energy
      *_mk = _mm_and_ps(_mm_cmpgt_ps(*_et,_En),_1);       // 1 if et>En or 0 if et<En
       _NN = _mm_add_ps(_NN,*_mk);                        // total energy threshold
       _ee = _mm_add_ps(_ee,*_et);                        // total energy threshold
      *_et = _mm_mul_ps(*_et,*_mk++);                     // apply En threshold 
       _EE = _mm_add_ps(_EE,*_et++);                      // total energy above threshold
   } 
   pAVX[1][I+1] = _wat_hsum(_NN);                         // store number of pixels
   return _wat_hsum(_EE)/2;                               // total energy in one quadrature
} 

static inline void _avx_loadNULL_ps(float** n, float** N, 
				    float** d, float** D,
				    float** h, float** H, int I) { 
// load NULL packet amplitudes for all detectors and pixels
// these amplitudes are used for reconstruction of data time searies
// now works only for <4 detector
   NETX(__m128* _n0 = (__m128*)n[0]; __m128* _N0 = (__m128*)N[0]; , 
	__m128* _n1 = (__m128*)n[1]; __m128* _N1 = (__m128*)N[1]; , 
	__m128* _n2 = (__m128*)n[2]; __m128* _N2 = (__m128*)N[2]; , 
	__m128* _n3 = (__m128*)n[3]; __m128* _N3 = (__m128*)N[3]; , 
	__m128* _n4 = (__m128*)n[4]; __m128* _N4 = (__m128*)N[4]; , 
	__m128* _n5 = (__m128*)n[5]; __m128* _N5 = (__m128*)N[5]; , 
	__m128* _n6 = (__m128*)n[6]; __m128* _N6 = (__m128*)N[6]; , 
	__m128* _n7 = (__m128*)n[7]; __m128* _N7 = (__m128*)N[7]; ) 

   NETX(__m128* _d0 = (__m128*)d[0]; __m128* _D0 = (__m128*)D[0]; , 
	__m128* _d1 = (__m128*)d[1]; __m128* _D1 = (__m128*)D[1]; , 
	__m128* _d2 = (__m128*)d[2]; __m128* _D2 = (__m128*)D[2]; , 
	__m128* _d3 = (__m128*)d[3]; __m128* _D3 = (__m128*)D[3]; , 
	__m128* _d4 = (__m128*)d[4]; __m128* _D4 = (__m128*)D[4]; , 
	__m128* _d5 = (__m128*)d[5]; __m128* _D5 = (__m128*)D[5]; , 
	__m128* _d6 = (__m128*)d[6]; __m128* _D6 = (__m128*)D[6]; , 
	__m128* _d7 = (__m128*)d[7]; __m128* _D7 = (__m128*)D[7]; ) 
 
   NETX(__m128* _h0 = (__m128*)h[0]; __m128* _H0 = (__m128*)H[0]; , 
	__m128* _h1 = (__m128*)h[1]; __m128* _H1 = (__m128*)H[1]; , 
	__m128* _h2 = (__m128*)h[2]; __m128* _H2 = (__m128*)H[2]; , 
	__m128* _h3 = (__m128*)h[3]; __m128* _H3 = (__m128*)H[3]; , 
	__m128* _h4 = (__m128*)h[4]; __m128* _H4 = (__m128*)H[4]; , 
	__m128* _h5 = (__m128*)h[5]; __m128* _H5 = (__m128*)H[5]; , 
	__m128* _h6 = (__m128*)h[6]; __m128* _H6 = (__m128*)H[6]; , 
	__m128* _h7 = (__m128*)h[7]; __m128* _H7 = (__m128*)H[7]; ) 

   for(int i=0; i<I; i+=4) {
      NETX(*_n0++ = _mm_sub_ps(*_d0++,*_h0++); *_N0++ = _mm_sub_ps(*_D0++,*_H0++); ,
           *_n1++ = _mm_sub_ps(*_d1++,*_h1++); *_N1++ = _mm_sub_ps(*_D1++,*_H1++); ,
           *_n2++ = _mm_sub_ps(*_d2++,*_h2++); *_N2++ = _mm_sub_ps(*_D2++,*_H2++); ,
           *_n3++ = _mm_sub_ps(*_d3++,*_h3++); *_N3++ = _mm_sub_ps(*_D3++,*_H3++); ,
           *_n4++ = _mm_sub_ps(*_d4++,*_h4++); *_N4++ = _mm_sub_ps(*_D4++,*_H4++); ,
           *_n5++ = _mm_sub_ps(*_d5++,*_h5++); *_N5++ = _mm_sub_ps(*_D5++,*_H5++); ,
           *_n6++ = _mm_sub_ps(*_d6++,*_h6++); *_N6++ = _mm_sub_ps(*_D6++,*_H6++); ,
           *_n7++ = _mm_sub_ps(*_d7++,*_h7++); *_N7++ = _mm_sub_ps(*_D7++,*_H7++); )
   }
   return;
} 


static inline float _avx_packet_ps(float** p, float** q,
				   std::vector<float*> &pAVX, int I) {
// calculates packet rotation sin/cos, amplitudes and unit vectors
// initialize unit vector arrays stored in pAVX in format used in  
// in likelihoodWP these arrays should be stored exactly in the same order.
// this function increments pointers stored in tne input pointer arrays.

   int II = abs(I)*2;
   __m128 _a, _A, _aa, _AA, _aA, _x, _cc, _ss, _nn, _cn, _sn, _mk;

   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _o = _mm_set1_ps(0.0001);
   static const __m128 sm = _mm_set1_ps(-0.f);          // -0.f = 1 << 31

   float  a[8] __attribute__((aligned(32))); __m128* _am = (__m128*)a;
   float  A[8] __attribute__((aligned(32))); __m128* _AM = (__m128*)A;
   float  s[8] __attribute__((aligned(32))); __m128* _si = (__m128*)s;
   float  c[8] __attribute__((aligned(32))); __m128* _co = (__m128*)c;

   __m128* _MK = (__m128*)(pAVX[1]);
   __m128 aa[8], AA[8], aA[8], si[8], co[8];
   for(int i=0; i<8; i++) {
      aa[i] = _mm_setzero_ps();
      AA[i] = _mm_setzero_ps();
      aA[i] = _mm_setzero_ps();
   }

   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; , 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; , 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; , 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; , 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; , 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; , 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; , 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; ) 

   for(int i=0; i<I; i+=4) { 
      _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1);        // event mask
      NETX(
	   aa[0] = _mm_add_ps(aa[0],_mm_mul_ps(_mm_mul_ps(*_p0,*_p0),    _mk));                  
	   AA[0] = _mm_add_ps(AA[0],_mm_mul_ps(_mm_mul_ps(*_q0,*_q0),    _mk));                  
	   aA[0] = _mm_add_ps(aA[0],_mm_mul_ps(_mm_mul_ps(*_p0++,*_q0++),_mk));,	     
	       	 		    	       	       	     	     	   	      	  
	   aa[1] = _mm_add_ps(aa[1],_mm_mul_ps(_mm_mul_ps(*_p1,*_p1),    _mk));                   
	   AA[1] = _mm_add_ps(AA[1],_mm_mul_ps(_mm_mul_ps(*_q1,*_q1),    _mk));                   
	   aA[1] = _mm_add_ps(aA[1],_mm_mul_ps(_mm_mul_ps(*_p1++,*_q1++),_mk));,
	     	  	            	       	       	     	     	   	    	          
	   aa[2] = _mm_add_ps(aa[2],_mm_mul_ps(_mm_mul_ps(*_p2,*_p2),    _mk));                   
	   AA[2] = _mm_add_ps(AA[2],_mm_mul_ps(_mm_mul_ps(*_q2,*_q2),    _mk));                   
	   aA[2] = _mm_add_ps(aA[2],_mm_mul_ps(_mm_mul_ps(*_p2++,*_q2++),_mk));,
	     	  	            	       	       	     	     	   	    	          
	   aa[3] = _mm_add_ps(aa[3],_mm_mul_ps(_mm_mul_ps(*_p3,*_p3),    _mk));                   
	   AA[3] = _mm_add_ps(AA[3],_mm_mul_ps(_mm_mul_ps(*_q3,*_q3),    _mk));                   
	   aA[3] = _mm_add_ps(aA[3],_mm_mul_ps(_mm_mul_ps(*_p3++,*_q3++),_mk));,
	     	  	            	       	       	     	     	   	    	          
	   aa[4] = _mm_add_ps(aa[4],_mm_mul_ps(_mm_mul_ps(*_p4,*_p4),    _mk));                   
	   AA[4] = _mm_add_ps(AA[4],_mm_mul_ps(_mm_mul_ps(*_q4,*_q4),    _mk));                   
	   aA[4] = _mm_add_ps(aA[4],_mm_mul_ps(_mm_mul_ps(*_p4++,*_q4++),_mk));,
	     	  	            	       	       	     	     	   	    	          
	   aa[5] = _mm_add_ps(aa[5],_mm_mul_ps(_mm_mul_ps(*_p5,*_p5),    _mk));                   
	   AA[5] = _mm_add_ps(AA[5],_mm_mul_ps(_mm_mul_ps(*_q5,*_q5),    _mk));                   
	   aA[5] = _mm_add_ps(aA[5],_mm_mul_ps(_mm_mul_ps(*_p5++,*_q5++),_mk));,
	                             	        	       	            	  
	   aa[6] = _mm_add_ps(aa[6],_mm_mul_ps(_mm_mul_ps(*_p6,*_p6),    _mk));                   
	   AA[6] = _mm_add_ps(AA[6],_mm_mul_ps(_mm_mul_ps(*_q6,*_q6),    _mk));                   
	   aA[6] = _mm_add_ps(aA[6],_mm_mul_ps(_mm_mul_ps(*_p6++,*_q6++),_mk));,
	                             	        	       	            	  
	   aa[7] = _mm_add_ps(aa[7],_mm_mul_ps(_mm_mul_ps(*_p7,*_p7),    _mk));                   
	   AA[7] = _mm_add_ps(AA[7],_mm_mul_ps(_mm_mul_ps(*_q7,*_q7),    _mk));                   
	   aA[7] = _mm_add_ps(aA[7],_mm_mul_ps(_mm_mul_ps(*_p7++,*_q7++),_mk));)  
   }    

// packet amplitudes and sin/cos for detectors 0-3
    _a = _mm_hadd_ps(aa[0],aa[1]); _A = _mm_hadd_ps(aa[2],aa[3]); _aa = _mm_hadd_ps(_a,_A);
    _a = _mm_hadd_ps(AA[0],AA[1]); _A = _mm_hadd_ps(AA[2],AA[3]); _AA = _mm_hadd_ps(_a,_A);
    _a = _mm_hadd_ps(aA[0],aA[1]); _A = _mm_hadd_ps(aA[2],aA[3]); _aA = _mm_hadd_ps(_a,_A);

   *_si = _mm_mul_ps(_aA,_2);                              // rotation 2*sin*cos*norm
   *_co = _mm_sub_ps(_aa,_AA);                             // rotation (cos^2-sin^2)*norm
     _x = _mm_add_ps(_mm_add_ps(_aa,_AA),_o);              // total energy
    _cc = _mm_mul_ps(*_co,*_co);
    _ss = _mm_mul_ps(*_si,*_si);
    _nn = _mm_sqrt_ps(_mm_add_ps(_cc,_ss));                // co/si norm
   *_am = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_x,_nn),_2));  // first component amplitude 
   *_AM = _mm_div_ps(_mm_sub_ps(_x,_nn),_2);               // second component energy
   *_AM = _mm_sqrt_ps(_mm_andnot_ps(sm,*_AM));             // make sure |AM|>0
    _cc = _mm_div_ps(*_co,_mm_add_ps(_nn,_o));             // cos(2p)
    _nn = _mm_and_ps(_mm_cmpgt_ps(*_si,_0),_1);            // 1 if sin(2p)>0. or 0 if sin(2p)<0.  
    _ss = _mm_sub_ps(_mm_mul_ps(_2,_nn),_1);               // 1 if sin(2p)>0. or-1 if sin(2p)<0.  
   *_si = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(_1,_cc),_2));  // |sin(p)|
   *_co = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_1,_cc),_2));  // |cos(p)|
   *_co = _mm_mul_ps(*_co,_ss);                            // cos(p)

// packet amplitudes and sin/cos for detectors 4-7
   _am++; _AM++; _si++; _co++;       
    _a = _mm_hadd_ps(aa[4],aa[5]); _A = _mm_hadd_ps(aa[6],aa[7]); _aa = _mm_hadd_ps(_a,_A);
    _a = _mm_hadd_ps(AA[4],AA[5]); _A = _mm_hadd_ps(AA[6],AA[7]); _AA = _mm_hadd_ps(_a,_A);
    _a = _mm_hadd_ps(aA[4],aA[5]); _A = _mm_hadd_ps(aA[6],aA[7]); _aA = _mm_hadd_ps(_a,_A);

   *_si = _mm_mul_ps(_aA,_2);                              // rotation 2*sin*cos*norm
   *_co = _mm_sub_ps(_aa,_AA);                             // rotation (cos^2-sin^2)*norm
     _x = _mm_add_ps(_mm_add_ps(_aa,_AA),_o);              // total energy
    _cc = _mm_mul_ps(*_co,*_co);
    _ss = _mm_mul_ps(*_si,*_si);
    _nn = _mm_sqrt_ps(_mm_add_ps(_cc,_ss));                // co/si norm
   *_am = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_x,_nn),_2));  // first component amplitude 
   *_AM = _mm_div_ps(_mm_sub_ps(_x,_nn),_2);               // second component energy
   *_AM = _mm_sqrt_ps(_mm_andnot_ps(sm,*_AM));             // make sure |AM|>0
    _cc = _mm_div_ps(*_co,_mm_add_ps(_nn,_o));             // cos(2p)
    _nn = _mm_and_ps(_mm_cmpgt_ps(*_si,_0),_1);            // 1 if sin(2p)>0. or 0 if sin(2p)<0.  
    _ss = _mm_sub_ps(_mm_mul_ps(_2,_nn),_1);               // 1 if sin(2p)>0. or-1 if sin(2p)<0.  
   *_si = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(_1,_cc),_2));  // |sin(p)|
   *_co = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_1,_cc),_2));  // |cos(p)|
   *_co = _mm_mul_ps(*_co,_ss);                            // cos(p)

   //cout<<"packet1: "<<a[0]<<" "<<A[0]<<" "<<a[1]<<" "<<A[1]<<" "<<a[2]<<" "<<A[2]<<" ";
   //cout<<((a[0]+A[0])*(a[0]+A[0])/2+(a[1]+A[1])*(a[1]+A[1])/2+(a[2]+A[2])*(a[2]+A[2])/2)<<"\n";

   NETX(q[0][II]=a[0]; q[0][II+1]=A[0]; q[0][II+2]=s[0]; q[0][II+3]=c[0]; q[0][II+5]=1;,
	q[1][II]=a[1]; q[1][II+1]=A[1]; q[1][II+2]=s[1]; q[1][II+3]=c[1]; q[1][II+5]=1;,
	q[2][II]=a[2]; q[2][II+1]=A[2]; q[2][II+2]=s[2]; q[2][II+3]=c[2]; q[2][II+5]=1;,
	q[3][II]=a[3]; q[3][II+1]=A[3]; q[3][II+2]=s[3]; q[3][II+3]=c[3]; q[3][II+5]=1;,
	q[4][II]=a[4]; q[4][II+1]=A[4]; q[4][II+2]=s[4]; q[4][II+3]=c[4]; q[4][II+5]=1;,
	q[5][II]=a[5]; q[5][II+1]=A[5]; q[5][II+2]=s[5]; q[5][II+3]=c[5]; q[5][II+5]=1;,
	q[6][II]=a[6]; q[6][II+1]=A[6]; q[6][II+2]=s[6]; q[6][II+3]=c[6]; q[6][II+5]=1;,
	q[7][II]=a[7]; q[7][II+1]=A[7]; q[7][II+2]=s[7]; q[7][II+3]=c[7]; q[7][II+5]=1;)

   float Ep = 0;
   NETX(q[0][II+4]=(a[0]+A[0]); Ep+=q[0][II+4]*q[0][II+4]/2; _p0 = (__m128*)p[0]; _q0 = (__m128*)q[0];,
	q[1][II+4]=(a[1]+A[1]); Ep+=q[1][II+4]*q[1][II+4]/2; _p1 = (__m128*)p[1]; _q1 = (__m128*)q[1];,
	q[2][II+4]=(a[2]+A[2]); Ep+=q[2][II+4]*q[2][II+4]/2; _p2 = (__m128*)p[2]; _q2 = (__m128*)q[2];,
	q[3][II+4]=(a[3]+A[3]); Ep+=q[3][II+4]*q[3][II+4]/2; _p3 = (__m128*)p[3]; _q3 = (__m128*)q[3];,
	q[4][II+4]=(a[4]+A[4]); Ep+=q[4][II+4]*q[4][II+4]/2; _p4 = (__m128*)p[4]; _q4 = (__m128*)q[4];,
	q[5][II+4]=(a[5]+A[5]); Ep+=q[5][II+4]*q[5][II+4]/2; _p5 = (__m128*)p[5]; _q5 = (__m128*)q[5];,
	q[6][II+4]=(a[6]+A[6]); Ep+=q[6][II+4]*q[6][II+4]/2; _p6 = (__m128*)p[6]; _q6 = (__m128*)q[6];,
	q[7][II+4]=(a[7]+A[7]); Ep+=q[7][II+4]*q[7][II+4]/2; _p7 = (__m128*)p[7]; _q7 = (__m128*)q[7];)
  
   *_am = _mm_div_ps(_1,_mm_add_ps(*_am,_o)); _am--;
   *_am = _mm_div_ps(_1,_mm_add_ps(*_am,_o));
   *_AM = _mm_div_ps(_1,_mm_add_ps(*_AM,_o)); _AM--;
   *_AM = _mm_div_ps(_1,_mm_add_ps(*_AM,_o));

   //cout<<"packet2: "<<a[0]<<" "<<A[0]<<" "<<a[1]<<" "<<A[1]<<" "<<a[2]<<" "<<A[2]<<" ";


   NETX(aa[0]=_mm_set1_ps(a[0]); AA[0]=_mm_set1_ps(A[0]); si[0]=_mm_set1_ps(s[0]); co[0]=_mm_set1_ps(c[0]); ,
	aa[1]=_mm_set1_ps(a[1]); AA[1]=_mm_set1_ps(A[1]); si[1]=_mm_set1_ps(s[1]); co[1]=_mm_set1_ps(c[1]); ,
	aa[2]=_mm_set1_ps(a[2]); AA[2]=_mm_set1_ps(A[2]); si[2]=_mm_set1_ps(s[2]); co[2]=_mm_set1_ps(c[2]); ,
	aa[3]=_mm_set1_ps(a[3]); AA[3]=_mm_set1_ps(A[3]); si[3]=_mm_set1_ps(s[3]); co[3]=_mm_set1_ps(c[3]); ,
	aa[4]=_mm_set1_ps(a[4]); AA[4]=_mm_set1_ps(A[4]); si[4]=_mm_set1_ps(s[4]); co[4]=_mm_set1_ps(c[4]); ,
	aa[5]=_mm_set1_ps(a[5]); AA[5]=_mm_set1_ps(A[5]); si[5]=_mm_set1_ps(s[5]); co[5]=_mm_set1_ps(c[5]); ,
	aa[6]=_mm_set1_ps(a[6]); AA[6]=_mm_set1_ps(A[6]); si[6]=_mm_set1_ps(s[6]); co[6]=_mm_set1_ps(c[6]); ,
	aa[7]=_mm_set1_ps(a[7]); AA[7]=_mm_set1_ps(A[7]); si[7]=_mm_set1_ps(s[7]); co[7]=_mm_set1_ps(c[7]); )

   _MK = (__m128*)(pAVX[1]);
   for(int i=0; i<I; i+=4) {                                // calculate and store unit vector components 
      _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1);         // event mask
      NETX(_a=_mm_add_ps(_mm_mul_ps(*_p0,co[0]),_mm_mul_ps(*_q0,si[0]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q0,co[0]),_mm_mul_ps(*_p0,si[0]));
           *_p0++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[0])); 
	   *_q0++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[0])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p1,co[1]),_mm_mul_ps(*_q1,si[1]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q1,co[1]),_mm_mul_ps(*_p1,si[1]));
           *_p1++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[1])); 
	   *_q1++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[1])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p2,co[2]),_mm_mul_ps(*_q2,si[2]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q2,co[2]),_mm_mul_ps(*_p2,si[2]));
           *_p2++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[2])); 
	   *_q2++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[2])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p3,co[3]),_mm_mul_ps(*_q3,si[3]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q3,co[3]),_mm_mul_ps(*_p3,si[3]));
           *_p3++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[3])); 
	   *_q3++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[3])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p4,co[4]),_mm_mul_ps(*_q4,si[4]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q4,co[4]),_mm_mul_ps(*_p4,si[4]));
           *_p4++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[4])); 
	   *_q4++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[4])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p5,co[5]),_mm_mul_ps(*_q5,si[5]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q5,co[5]),_mm_mul_ps(*_p5,si[5]));
           *_p5++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[5])); 
	   *_q5++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[5])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p6,co[6]),_mm_mul_ps(*_q6,si[6]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q6,co[6]),_mm_mul_ps(*_p6,si[6]));
           *_p6++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[6])); 
	   *_q6++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[6])); ,

           _a=_mm_add_ps(_mm_mul_ps(*_p7,co[7]),_mm_mul_ps(*_q7,si[7]));
 	   _A=_mm_sub_ps(_mm_mul_ps(*_q7,co[7]),_mm_mul_ps(*_p7,si[7]));
           *_p7++ = _mm_mul_ps(_mk,_mm_mul_ps(_a,aa[7])); 
	   *_q7++ = _mm_mul_ps(_mk,_mm_mul_ps(_A,AA[7])); )
   }	   	 
   return Ep/2;    // returm packet energy p[er quadrature
} 

static inline float _avx_setAMP_ps(float** p, float** q, 
				   std::vector<float*> &pAVX, int I) {
// set packet amplitudes for waveform reconstruction
// returns number of degrees of freedom - effective # of pixels per detector   
   int II = I*2;
   int I2 = II+2;
   int I3 = II+3;
   int I4 = II+4;
   int I5 = II+5;
   float k = pAVX[1][I];              // number of detectors

   __m128 _a, _A, _n, _mk, _nn;
   static const __m128 _o = _mm_set1_ps(1.e-9);
   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _4 = _mm_set1_ps(4);
   static const __m128 o5 = _mm_set1_ps(0.5);

   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; __m128* _n0 = (__m128*)(q[0]+I); , 
	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; __m128* _n1 = (__m128*)(q[1]+I); , 
	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; __m128* _n2 = (__m128*)(q[2]+I); , 
	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; __m128* _n3 = (__m128*)(q[3]+I); , 
	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; __m128* _n4 = (__m128*)(q[4]+I); , 
	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; __m128* _n5 = (__m128*)(q[5]+I); , 
	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; __m128* _n6 = (__m128*)(q[6]+I); , 
	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; __m128* _n7 = (__m128*)(q[7]+I); ) 

   NETX(__m128 a0=_mm_set1_ps(q[0][I4]); __m128 s0=_mm_set1_ps(q[0][I2]); __m128 c0=_mm_set1_ps(q[0][I3]); ,
	__m128 a1=_mm_set1_ps(q[1][I4]); __m128 s1=_mm_set1_ps(q[1][I2]); __m128 c1=_mm_set1_ps(q[1][I3]); ,
	__m128 a2=_mm_set1_ps(q[2][I4]); __m128 s2=_mm_set1_ps(q[2][I2]); __m128 c2=_mm_set1_ps(q[2][I3]); ,
	__m128 a3=_mm_set1_ps(q[3][I4]); __m128 s3=_mm_set1_ps(q[3][I2]); __m128 c3=_mm_set1_ps(q[3][I3]); ,
	__m128 a4=_mm_set1_ps(q[4][I4]); __m128 s4=_mm_set1_ps(q[4][I2]); __m128 c4=_mm_set1_ps(q[4][I3]); ,
	__m128 a5=_mm_set1_ps(q[5][I4]); __m128 s5=_mm_set1_ps(q[5][I2]); __m128 c5=_mm_set1_ps(q[5][I3]); ,
	__m128 a6=_mm_set1_ps(q[6][I4]); __m128 s6=_mm_set1_ps(q[6][I2]); __m128 c6=_mm_set1_ps(q[6][I3]); ,
	__m128 a7=_mm_set1_ps(q[7][I4]); __m128 s7=_mm_set1_ps(q[7][I2]); __m128 c7=_mm_set1_ps(q[7][I3]); )

   __m128* _MK = (__m128*)pAVX[1];
   __m128* _fp = (__m128*)pAVX[2];       
   __m128* _fx = (__m128*)pAVX[3];
   __m128* _ee = (__m128*)pAVX[15];
   __m128* _EE = (__m128*)pAVX[16];
   __m128* _gn = (__m128*)pAVX[20];

   __m128  _Np = _mm_setzero_ps();        // number of effective pixels per detector
     
   for(int i=0; i<I; i+=4) {  //  packet amplitudes 
      _mk = _mm_mul_ps(o5,_mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1));                  // event mask
      NETX(
	   _n = _mm_mul_ps(_mm_mul_ps(a0,_mk),*_n0); _a=*_p0; _A=*_q0; _nn=*_n0++;
	   *_p0++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c0),_mm_mul_ps(_A,s0))); 
	   *_q0++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c0),_mm_mul_ps(_a,s0))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a1,_mk),*_n1); _a=*_p1; _A=*_q1; _nn=_mm_add_ps(_nn,*_n1++);
	   *_p1++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c1),_mm_mul_ps(_A,s1))); 
	   *_q1++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c1),_mm_mul_ps(_a,s1))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a2,_mk),*_n2); _a=*_p2; _A=*_q2; _nn=_mm_add_ps(_nn,*_n2++);
	   *_p2++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c2),_mm_mul_ps(_A,s2))); 
	   *_q2++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c2),_mm_mul_ps(_a,s2))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a3,_mk),*_n3); _a=*_p3; _A=*_q3; _nn=_mm_add_ps(_nn,*_n3++);
	   *_p3++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c3),_mm_mul_ps(_A,s3))); 
	   *_q3++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c3),_mm_mul_ps(_a,s3))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a4,_mk),*_n4); _a=*_p4; _A=*_q4; _nn=_mm_add_ps(_nn,*_n4++);
	   *_p4++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c4),_mm_mul_ps(_A,s4))); 
	   *_q4++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c4),_mm_mul_ps(_a,s4))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a5,_mk),*_n5); _a=*_p5; _A=*_q5; _nn=_mm_add_ps(_nn,*_n5++);
	   *_p5++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c5),_mm_mul_ps(_A,s5))); 
	   *_q5++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c5),_mm_mul_ps(_a,s5))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a6,_mk),*_n6); _a=*_p6; _A=*_q6; _nn=_mm_add_ps(_nn,*_n6++);
	   *_p6++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c6),_mm_mul_ps(_A,s6))); 
	   *_q6++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c6),_mm_mul_ps(_a,s6))); ,
	   _n = _mm_mul_ps(_mm_mul_ps(a7,_mk),*_n7); _a=*_p7; _A=*_q7; _nn=_mm_add_ps(_nn,*_n7++);
	   *_p7++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c7),_mm_mul_ps(_A,s7))); 
	   *_q7++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c7),_mm_mul_ps(_a,s7))); )

      _nn = _mm_mul_ps(_nn,_mk);
      _Np = _mm_add_ps(_Np,_nn);                         // Dof * k/4
   }
   return _wat_hsum(_Np)*4/k;
} 

static inline float _avx_ort_ps(float** p, float** q,
				std::vector<float*> &pAVX, int I) {
// orthogonalize data vectors p and q
// calculate norms of orthogonal vectors and rotation sin & cos
// p/q - array of pointers to 00/90 phase detector data
// returns signal energy       
// in likelihoodWP these arrays should be stored exactly in the same order.
// this function increments pointers stored in tne input pointer arrays. 
   __m128 _a,_A,_aa,_AA,_aA,_et,_cc,_ss,_nn,_cn,_sn,_mk;

   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _o = _mm_set1_ps(1.e-21);
   static const __m128 sm = _mm_set1_ps(-0.f);          // -0.f = 1 << 31

   __m128* _MK = (__m128*)pAVX[1];
   __m128* _si = (__m128*)pAVX[4];
   __m128* _co = (__m128*)pAVX[5];
   __m128* _ee = (__m128*)pAVX[15];
   __m128* _EE = (__m128*)pAVX[16];
   __m128   _e = _mm_setzero_ps();
   __m128   _E = _mm_setzero_ps();

   NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; , 
        __m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; , 
        __m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; , 
        __m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; , 
        __m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; , 
        __m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; , 
        __m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; , 
        __m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; ) 

   for(int i=0; i<I; i+=4) { 
      NETX(
           _aa=_mm_mul_ps(*_p0,*_p0);                       
           _AA=_mm_mul_ps(*_q0,*_q0);                       
           _aA=_mm_mul_ps(*_p0++,*_q0++);                  ,         
                                                                     
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p1,*_p1));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q1,*_q1));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p1++,*_q1++)); ,
                                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p2,*_p2));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q2,*_q2));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p2++,*_q2++)); ,
                                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p3,*_p3));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q3,*_q3));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p3++,*_q3++)); ,
                                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p4,*_p4));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q4,*_q4));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p4++,*_q4++)); ,
                                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p5,*_p5));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q5,*_q5));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p5++,*_q5++)); ,
                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p6,*_p6));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q6,*_q6));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p6++,*_q6++)); ,
                                                             
           _aa = _mm_add_ps(_aa,_mm_mul_ps(*_p7,*_p7));                   
           _AA = _mm_add_ps(_AA,_mm_mul_ps(*_q7,*_q7));                   
           _aA = _mm_add_ps(_aA,_mm_mul_ps(*_p7++,*_q7++)); )
         
// calculate data network orthogonalization sin and cos

      _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK,_0),_1);         // event mask
      *_si = _mm_mul_ps(_aA,_2);                          // rotation 2*sin*cos*norm
      *_co = _mm_sub_ps(_aa,_AA);                         // rotation (cos^2-sin^2)*norm
       _et = _mm_add_ps(_mm_add_ps(_aa,_AA),_o);          // total energy
       _cc = _mm_mul_ps(*_co,*_co);
       _ss = _mm_mul_ps(*_si,*_si);
       _nn = _mm_sqrt_ps(_mm_add_ps(_cc,_ss));            // co/si norm
      *_ee = _mm_div_ps(_mm_add_ps(_et,_nn),_2);          // first component energy
      *_EE = _mm_div_ps(_mm_sub_ps(_et,_nn),_2);          // second component energy
       _cc = _mm_div_ps(*_co,_mm_add_ps(_nn,_o));             // cos(2p)
       _nn = _mm_and_ps(_mm_cmpgt_ps(*_si,_0),_1);            // 1 if sin(2p)>0. or 0 if sin(2p)<0. 
       _ss = _mm_sub_ps(_mm_mul_ps(_2,_nn),_1);               // 1 if sin(2p)>0. or-1 if sin(2p)<0. 
      *_si = _mm_sqrt_ps(_mm_div_ps(_mm_sub_ps(_1,_cc),_2));  // |sin(p)|
      *_co = _mm_sqrt_ps(_mm_div_ps(_mm_add_ps(_1,_cc),_2));  // |cos(p)|
      *_co = _mm_mul_ps(*_co,_ss);                            // cos(p)
         
      _e = _mm_add_ps(_e,_mm_mul_ps(_mk,*_ee));
      _E = _mm_add_ps(_E,_mm_mul_ps(_mk,*_EE));
      _si++; _co++; _MK++; _ee++; _EE++;
   } 
   return _wat_hsum(_e)+_wat_hsum(_E);
} 


static inline __m256 _avx_stat_ps(float** x, float** X,
				  float** s, float** S,
				  std::vector<float*> &pAVX, int I) {
// returns coherent statistics in the format {cc,ec,ed,gn}
// ei - incoherent energy: sum{s[i]^4}/|s|^2 + sum{S[i]^4}/|S|^2
// ec - coherent energy: |s|^2+|S|^2 - ei
// ed - energy disbalance: 0.5*sum_k{(x[k]*s[k]-s[k]*s[k])^2} * (s,s)/(x,s)^2
//      + its 90-degrees phase value
// cc - reduced network correlation coefficient
// p/q - pointers to data/signal amplitudes
// nIFO - number of detectors.
// I - number of pixels
   float k = pAVX[1][I];              // number of detectors
   __m128 _a,_A,_x,_X,_c,_C,_s,_S,_r,_R,_rr,_RR,_xs,_XS;
   __m128 _ll,_ei,_cc,_mk,_mm,_ss,_SS;
   static const __m128 _o = _mm_set1_ps(0.001); 
   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _k = _mm_set1_ps(2*(1-k)); 
   static const __m128 sm = _mm_set1_ps(-0.f);          // -0.f = 1 << 31

   __m128* _et = (__m128*)pAVX[0];
   __m128* _MK = (__m128*)pAVX[1];
   __m128* _si = (__m128*)pAVX[4];
   __m128* _co = (__m128*)pAVX[5];
   __m128* _ec = (__m128*)pAVX[19];
   __m128* _gn = (__m128*)pAVX[20];
   __m128* _ed = (__m128*)pAVX[21];
   __m128* _rn = (__m128*)pAVX[22];

   __m128  _LL = _mm_setzero_ps();
   __m128  _Lr = _mm_setzero_ps();
   __m128  _EC = _mm_setzero_ps();
   __m128  _GN = _mm_setzero_ps();
   __m128  _RN = _mm_setzero_ps();
   __m128  _NN = _mm_setzero_ps();

   NETX(__m128* _x0 = (__m128*)x[0]; __m128* _X0 = (__m128*)X[0]; , 
        __m128* _x1 = (__m128*)x[1]; __m128* _X1 = (__m128*)X[1]; , 
        __m128* _x2 = (__m128*)x[2]; __m128* _X2 = (__m128*)X[2]; , 
        __m128* _x3 = (__m128*)x[3]; __m128* _X3 = (__m128*)X[3]; , 
        __m128* _x4 = (__m128*)x[4]; __m128* _X4 = (__m128*)X[4]; , 
        __m128* _x5 = (__m128*)x[5]; __m128* _X5 = (__m128*)X[5]; , 
        __m128* _x6 = (__m128*)x[6]; __m128* _X6 = (__m128*)X[6]; , 
        __m128* _x7 = (__m128*)x[7]; __m128* _X7 = (__m128*)X[7]; ) 

   NETX(__m128* _s0 = (__m128*)s[0]; __m128* _S0 = (__m128*)S[0]; , 
        __m128* _s1 = (__m128*)s[1]; __m128* _S1 = (__m128*)S[1]; , 
        __m128* _s2 = (__m128*)s[2]; __m128* _S2 = (__m128*)S[2]; , 
        __m128* _s3 = (__m128*)s[3]; __m128* _S3 = (__m128*)S[3]; , 
        __m128* _s4 = (__m128*)s[4]; __m128* _S4 = (__m128*)S[4]; , 
        __m128* _s5 = (__m128*)s[5]; __m128* _S5 = (__m128*)S[5]; , 
        __m128* _s6 = (__m128*)s[6]; __m128* _S6 = (__m128*)S[6]; , 
        __m128* _s7 = (__m128*)s[7]; __m128* _S7 = (__m128*)S[7]; ) 

   for(int i=0; i<I; i+=4) { 
      NETX(
	   _s=_mm_add_ps(_mm_mul_ps(*_s0,*_co),_mm_mul_ps(*_S0,*_si));     _r=_mm_sub_ps(*_s0,*_x0); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x0,*_co),_mm_mul_ps(*_X0,*_si));     _R=_mm_sub_ps(*_S0,*_X0);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S0++,*_co),_mm_mul_ps(*_s0++,*_si)); _a=_mm_mul_ps(_s,_x); _xs=_a;
	   _X=_mm_sub_ps(_mm_mul_ps(*_X0++,*_co),_mm_mul_ps(*_x0++,*_si)); _A=_mm_mul_ps(_S,_X); _XS=_A;
	   _c=_mm_mul_ps(_a,_a); _ss=_mm_mul_ps(_s,_s); _rr=_mm_mul_ps(_r,_r); 
	   _C=_mm_mul_ps(_A,_A); _SS=_mm_mul_ps(_S,_S); _RR=_mm_mul_ps(_R,_R); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s1,*_co),  _mm_mul_ps(*_S1,*_si));   _r=_mm_sub_ps(*_s1,*_x1); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x1,*_co),  _mm_mul_ps(*_X1,*_si));   _R=_mm_sub_ps(*_S1,*_X1);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S1++,*_co),_mm_mul_ps(*_s1++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X1++,*_co),_mm_mul_ps(*_x1++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s2,*_co),  _mm_mul_ps(*_S2,*_si));   _r=_mm_sub_ps(*_s2,*_x2); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x2,*_co),  _mm_mul_ps(*_X2,*_si));   _R=_mm_sub_ps(*_S2,*_X2);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S2++,*_co),_mm_mul_ps(*_s2++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X2++,*_co),_mm_mul_ps(*_x2++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s3,*_co),  _mm_mul_ps(*_S3,*_si));   _r=_mm_sub_ps(*_s3,*_x3); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x3,*_co),  _mm_mul_ps(*_X3,*_si));   _R=_mm_sub_ps(*_S3,*_X3);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S3++,*_co),_mm_mul_ps(*_s3++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X3++,*_co),_mm_mul_ps(*_x3++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s4,*_co),  _mm_mul_ps(*_S4,*_si));   _r=_mm_sub_ps(*_s4,*_x4); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x4,*_co),  _mm_mul_ps(*_X4,*_si));   _R=_mm_sub_ps(*_S4,*_X4);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S4++,*_co),_mm_mul_ps(*_s4++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X4++,*_co),_mm_mul_ps(*_x4++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s5,*_co),  _mm_mul_ps(*_S5,*_si));   _r=_mm_sub_ps(*_s5,*_x5); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x5,*_co),  _mm_mul_ps(*_X5,*_si));   _R=_mm_sub_ps(*_S5,*_X5);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S5++,*_co),_mm_mul_ps(*_s5++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X5++,*_co),_mm_mul_ps(*_x5++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s6,*_co),  _mm_mul_ps(*_S6,*_si));   _r=_mm_sub_ps(*_s6,*_x6); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x6,*_co),  _mm_mul_ps(*_X6,*_si));   _R=_mm_sub_ps(*_S6,*_X6);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S6++,*_co),_mm_mul_ps(*_s6++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X6++,*_co),_mm_mul_ps(*_x6++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); ,

	   _s=_mm_add_ps(_mm_mul_ps(*_s7,*_co),  _mm_mul_ps(*_S7,*_si));   _r=_mm_sub_ps(*_s7,*_x7); 
	   _x=_mm_add_ps(_mm_mul_ps(*_x7,*_co),  _mm_mul_ps(*_X7,*_si));   _R=_mm_sub_ps(*_S7,*_X7);
	   _S=_mm_sub_ps(_mm_mul_ps(*_S7++,*_co),_mm_mul_ps(*_s7++,*_si)); _a=_mm_mul_ps(_s,_x);  _xs=_mm_add_ps(_xs,_a);
	   _X=_mm_sub_ps(_mm_mul_ps(*_X7++,*_co),_mm_mul_ps(*_x7++,*_si)); _A=_mm_mul_ps(_S,_X);  _XS=_mm_add_ps(_XS,_A);
	   _c=_mm_add_ps(_c,_mm_mul_ps(_a,_a)); _ss=_mm_add_ps(_ss,_mm_mul_ps(_s,_s)); _rr=_mm_add_ps(_rr,_mm_mul_ps(_r,_r)); 
	   _C=_mm_add_ps(_C,_mm_mul_ps(_A,_A)); _SS=_mm_add_ps(_SS,_mm_mul_ps(_S,_S)); _RR=_mm_add_ps(_RR,_mm_mul_ps(_R,_R)); )

 
      _mk = _mm_and_ps(_mm_cmpge_ps(*_MK,_0),_1);                             // event mask				  
      _c  = _mm_div_ps(_c,_mm_add_ps(_mm_mul_ps(_xs,_xs),_o));                // first component incoherent energy	  
      _C  = _mm_div_ps(_C,_mm_add_ps(_mm_mul_ps(_XS,_XS),_o));                // second component incoherent energy	  
      _ll = _mm_mul_ps(_mk,_mm_add_ps(_ss,_SS));         		      // signal energy        		  
      _ss = _mm_mul_ps(_ss,_mm_sub_ps(_1,_c));                  	      // 00 coherent energy                     
      _SS = _mm_mul_ps(_SS,_mm_sub_ps(_1,_C));                  	      // 90 coherent energy                     
     *_ec = _mm_mul_ps(_mk,_mm_add_ps(_ss,_SS));         		      // coherent energy			  
     *_gn = _mm_mul_ps(_mk,_mm_mul_ps(_2,*_MK));         		      // G-noise correction			  
     *_rn = _mm_mul_ps(_mk,_mm_add_ps(_rr,_RR));                              // residual noise in TF domain		  
									                                               
       _a = _mm_mul_ps(_2,_mm_andnot_ps(sm,*_ec));                            // 2*|ec|
       _A = _mm_add_ps(*_rn,_mm_add_ps(_o,*_gn));        		      // NULL 				  
      _cc = _mm_div_ps(*_ec,_mm_add_ps(_a,_A));           		      // correlation coefficient		  
      _Lr = _mm_add_ps(_Lr,_mm_mul_ps(_ll,_cc));         		      // reduced likelihood			  
      _mm = _mm_and_ps(_mm_cmpgt_ps(*_ec,_o),_1);                             // coherent energy mask				  
									                                               
      _LL = _mm_add_ps(_LL, _ll);                        		      // total signal energy			  
      _GN = _mm_add_ps(_GN,*_gn);                        		      // total G-noise correction		  
      _EC = _mm_add_ps(_EC,*_ec);                        		      // total coherent energy		  
      _RN = _mm_add_ps(_RN,*_rn);                        		      // residual noise in TF domain		  
      _NN = _mm_add_ps(_NN, _mm);                                             // number of pixel in TF domain with Ec>0          

      _si++; _co++; _MK++; _ec++; _gn++; _ed++; _rn++; 
   }
   float cc = _wat_hsum(_Lr)/(_wat_hsum(_LL)+0.001);     // network correlation coefficient
   float ec = _wat_hsum(_EC);                            // total coherent energy
   float rn = _wat_hsum(_GN)+_wat_hsum(_RN);             // total noise x 2
   float nn = _wat_hsum(_NN);                            // number of pixels
   float ch = rn/(k*nn+sqrt(nn))/2;                      // chi2 in TF domain
   return _mm256_set_ps(0,0,0,0,rn/2,nn,ec,2*cc);        // reversed order
} 


static inline __m256 _avx_noise_ps(float** p, float** q, std::vector<float*> &pAVX, int I) {
// q - pointer to pixel norms 
// returns noise correction
// I - number of pixels
   float k = pAVX[1][I];              // number of detectors
   __m128 _nx,_ns,_nm,_mk,_gg,_rc,_nn;
   __m128* _et = (__m128*)pAVX[0];
   __m128* _MK = (__m128*)pAVX[1];
   __m128* _ec = (__m128*)pAVX[19];
   __m128* _gn = (__m128*)pAVX[20];
   __m128* _rn = (__m128*)pAVX[22];
   __m128  _GN = _mm_setzero_ps();
   __m128  _RC = _mm_setzero_ps();
   __m128  _ES = _mm_setzero_ps();
   __m128  _EH = _mm_setzero_ps();
   __m128  _EC = _mm_setzero_ps();
   __m128  _SC = _mm_setzero_ps();
   __m128  _NC = _mm_setzero_ps();
   __m128  _NS = _mm_setzero_ps();

   static const __m128 _0 = _mm_set1_ps(0);
   static const __m128 o5 = _mm_set1_ps(0.5);
   static const __m128 _o = _mm_set1_ps(1.e-9);
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _k = _mm_set1_ps(1./k);

   NETX(__m128* _s0 = (__m128*)(p[0]+I);, __m128* _s1 = (__m128*)(p[1]+I);,
	__m128* _s2 = (__m128*)(p[2]+I);, __m128* _s3 = (__m128*)(p[3]+I);, 
	__m128* _s4 = (__m128*)(p[4]+I);, __m128* _s5 = (__m128*)(p[5]+I);,
	__m128* _s6 = (__m128*)(p[6]+I);, __m128* _s7 = (__m128*)(p[7]+I);)

   NETX(__m128* _x0 = (__m128*)(q[0]+I);, __m128* _x1 = (__m128*)(q[1]+I);,
	__m128* _x2 = (__m128*)(q[2]+I);, __m128* _x3 = (__m128*)(q[3]+I);, 
	__m128* _x4 = (__m128*)(q[4]+I);, __m128* _x5 = (__m128*)(q[5]+I);,
	__m128* _x6 = (__m128*)(q[6]+I);, __m128* _x7 = (__m128*)(q[7]+I);)

   for(int i=0; i<I; i+=4) { 
      NETX(_ns =*_s0++;,                  _ns = _mm_add_ps(*_s1++,_ns); , 
           _ns = _mm_add_ps(*_s2++,_ns);, _ns = _mm_add_ps(*_s3++,_ns); , 
           _ns = _mm_add_ps(*_s4++,_ns);, _ns = _mm_add_ps(*_s5++,_ns); , 
           _ns = _mm_add_ps(*_s6++,_ns);, _ns = _mm_add_ps(*_s7++,_ns); )
	 
      NETX(_nx =*_x0++;,                  _nx = _mm_add_ps(*_x1++,_nx); , 
           _nx = _mm_add_ps(*_x2++,_nx);, _nx = _mm_add_ps(*_x3++,_nx); , 
           _nx = _mm_add_ps(*_x4++,_nx);, _nx = _mm_add_ps(*_x5++,_nx); , 
           _nx = _mm_add_ps(*_x6++,_nx);, _nx = _mm_add_ps(*_x7++,_nx); )
	 	 
      _ns = _mm_mul_ps(_ns,_k);
      _nx = _mm_mul_ps(_nx,_k);
      _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK,_0),_1);        // event mask
      _nm = _mm_and_ps(_mm_cmpgt_ps(_nx,_0),_1);         // data norm mask
      _nm = _mm_mul_ps(_mk,_nm);                         // norm x event mask      
      _EC = _mm_add_ps(_EC,_mm_mul_ps(_nm,*_ec));        // coherent energy
      _NC = _mm_add_ps(_NC,_nm);                         // number of core pixels

      _nm = _mm_sub_ps(_mk,_nm);                         // halo mask
      _ES = _mm_add_ps(_ES,_mm_mul_ps(_nm,*_rn));        // residual sattelite noise

      _rc = _mm_and_ps(_mm_cmplt_ps(*_gn,_2),_1);        // rc=1 if gn<2, or rc=0 if gn>=2
      _nn = _mm_mul_ps(*_gn,_mm_sub_ps(_1,_rc));         // _nn = [_1 - _rc] * _gn
      _rc = _mm_add_ps(_rc,_mm_mul_ps(_nn,o5));          // Ec normalization
      _rc = _mm_div_ps(*_ec,_mm_add_ps(_rc,_o));         // normalized EC

      _nm = _mm_and_ps(_mm_cmpgt_ps(_ns,_0),_1);         // signal norm mask
      _nm = _mm_mul_ps(_mk,_nm);                         // norm x event mask      
     *_gn = _mm_mul_ps(_mk,_mm_mul_ps(*_gn,_nx));        // normalize Gaussian noise ecorrection
      _SC = _mm_add_ps(_SC,_mm_mul_ps(_nm,*_ec));        // signal coherent energy
      _RC = _mm_add_ps(_RC,_mm_mul_ps(_nm, _rc));        // total normalized EC
      _GN = _mm_add_ps(_GN,_mm_mul_ps(_nm,*_gn));        // G-noise correction in time domain
      _NS = _mm_add_ps(_NS,_nm);                         // number of signal pixels

      _nm = _mm_sub_ps(_mk,_nm);                         // satellite mask
      _EH = _mm_add_ps(_EH,_mm_mul_ps(_nm,*_et));        // halo energy in TF domain

      _MK++; _gn++; _et++; _ec++; _rn++;
   }
   float es =  _wat_hsum(_ES)/2;                         // residual satellite energy in time domain 
   float eh =  _wat_hsum(_EH)/2;                         // halo energy in TF domain
   float gn =  _wat_hsum(_GN);                           // G-noise correction
   float nc =  _wat_hsum(_NC);                           // number of core pixels
   float ns =  _wat_hsum(_NS);                           // number of signal pixels
   float rc =  _wat_hsum(_RC);                           // normalized EC x 2 
   float ec =  _wat_hsum(_EC);                           // signal coherent energy x 2
   float sc =  _wat_hsum(_SC);                           // core coherent energy x 2
   return _mm256_set_ps(ns,nc,es,eh,rc/(sc+0.01),sc-ec,ec,gn);
} 


//   vec.push_back(m[j]*(fp*(a+A)*(u[j]*cu-U[j]*su)/N[0] + fx*(b+B)*(V[j]*cv+v[j]*sv)/N[1]));
//   vec.push_back(m[j]*(fp*(a+A)*(U[j]*cu+u[j]*su)/N[0] + fx*(b+B)*(v[j]*cv-V[j]*sv)/N[1]));


#endif // WATAVX_HH

















