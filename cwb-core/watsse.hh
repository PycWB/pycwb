/*
# Copyright (C) 2019 Sergey Klimenko
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

#ifndef WATSSE_HH
#define WATSSE_HH

#include <xmmintrin.h>
#include <pmmintrin.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <TMath.h>
#include "wat.hh"

// WAT SSE functions


static inline void _sse_print_ps(__m128* _p) {
   float x[4];
   _NET(_mm_storeu_ps(x,_p[0]); printf("%e %e %e %e ",x[0],x[1],x[2],x[3]);,
        _mm_storeu_ps(x,_p[1]); printf("%e %e %e %e ",x[0],x[1],x[2],x[3]);)
      cout<<endl;
} 

static inline void _sse_zero_ps(__m128* _p) {
   _NET(_p[0] = _mm_setzero_ps();,
        _p[1] = _mm_setzero_ps();)
      return;
} 

static inline void _sse_load_ps(__m128* _p, float a) {
   _NET(_p[0] = _mm_load1_ps(&a);,
        _p[1] = _mm_load1_ps(&a);)
      return;
} 

static inline void _sse_mul_ps(__m128* _a, float b) {
  __m128 _b = _mm_load1_ps(&b);
  _NET(_a[0] = _mm_mul_ps(_a[0],_b);,
       _a[1] = _mm_mul_ps(_a[1],_b);)
}

static inline void _sse_mul_ps(__m128* _a, __m128* _b) {
  _NET(_a[0] = _mm_mul_ps(_a[0],_b[0]);,
       _a[1] = _mm_mul_ps(_a[1],_b[1]);)
}

static inline float _sse_mul_ps(__m128* _a, __m128* _b, __m128* _o) {
   float x[4];
   float out;
  _NET(_o[0] = _mm_mul_ps(_a[0],_b[0]);_mm_storeu_ps(x,_o[0]); out = XSUM(x);,
       _o[1] = _mm_mul_ps(_a[1],_b[1]);_mm_storeu_ps(x,_o[1]); out+= YSUM(x);)
      return out;
}

static inline void _sse_mul4_ps(__m128* _am, __m128 _c) {
   float c[4]; _mm_storeu_ps(c,_c); 
  __m128* _a = _am;
   _NET(_c=_mm_load1_ps( c );*_a=_mm_mul_ps(*_a,_c);_a++;, *_a=_mm_mul_ps(*_a,_c);_a++;)
   _NET(_c=_mm_load1_ps(c+1);*_a=_mm_mul_ps(*_a,_c);_a++;, *_a=_mm_mul_ps(*_a,_c);_a++;)
   _NET(_c=_mm_load1_ps(c+2);*_a=_mm_mul_ps(*_a,_c);_a++;, *_a=_mm_mul_ps(*_a,_c);_a++;)
   _NET(_c=_mm_load1_ps(c+3);*_a=_mm_mul_ps(*_a,_c);_a++;, *_a=_mm_mul_ps(*_a,_c);_a++;)
}

static inline void _sse_hard4_ps(__m128* _uu, __m128* _am, __m128* _AM, __m128 _c) {
// calculate hard responses
// replace standard with hard responses if _c=0;
// uu - unity vector along f+
// am, AM - 00 and 90 phase responses.
   float c[4]; _mm_storeu_ps(c,_c); 
  __m128* _u = _uu;
  __m128* _a = _am;
  __m128* _A = _AM;
  __m128 _r, _R;
  _NET(
       _r = _mm_set1_ps(c[0]); _R = _mm_set1_ps(1-c[0]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;
       )                                        
  _NET(
       _r = _mm_set1_ps(c[1]); _R = _mm_set1_ps(1-c[1]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;
       )                                        
  _NET(
       _r = _mm_set1_ps(c[2]); _R = _mm_set1_ps(1-c[2]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;
       )                                        
  _NET(
       _r = _mm_set1_ps(c[3]); _R = _mm_set1_ps(1-c[3]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_r),_mm_mul_ps(_mm_mul_ps(*_u++,*_a),_R));_a++;*_A = _mm_mul_ps(*_A,_r);_A++;
       )                                        
}

static inline void _sse_ifcp4_ps(__m128* _aa, __m128* _bb,  __m128 _c) {
// sabstutute vector _aa with _bb  if _c=0;
   float c[4]; _mm_storeu_ps(c,_c); 
  __m128* _a = _aa;
  __m128* _b = _bb;
  __m128 _1, _0;
  _NET(_1 = _mm_set1_ps(c[0]); _0 = _mm_set1_ps(1-c[0]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;)
  _NET(_1 = _mm_set1_ps(c[1]); _0 = _mm_set1_ps(1-c[1]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;)
  _NET(_1 = _mm_set1_ps(c[2]); _0 = _mm_set1_ps(1-c[2]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;) 
  _NET(_1 = _mm_set1_ps(c[3]); _0 = _mm_set1_ps(1-c[3]);
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;,
       *_a = _mm_add_ps(_mm_mul_ps(*_a,_1),_mm_mul_ps(*_b++,_0));_a++;)                                        
}


static inline float _sse_abs_ps(__m128* _a) {
  float x[4];
  float out;
  _NET(
       _mm_storeu_ps(x,_mm_mul_ps(_a[0],_a[0])); out = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(_a[1],_a[1])); out+= YSUM(x);
       )
     return out;
}

static inline float _sse_abs_ps(__m128* _a, __m128* _A) {
  float x[4];
  float out;
  _NET(
       _mm_storeu_ps(x,_mm_add_ps(_mm_mul_ps(_a[0],_a[0]),_mm_mul_ps(_A[0],_A[0]))); out = XSUM(x);,
       _mm_storeu_ps(x,_mm_add_ps(_mm_mul_ps(_a[1],_a[1]),_mm_mul_ps(_A[1],_A[1]))); out+= YSUM(x);
       )
     return out;
}

static inline __m128 _sse_abs4_ps(__m128* _p) {
// return |p|^2
  float x[4];
  float o[4];
  __m128* _a = _p;
  _NET(
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[0] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[0]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[1] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[1]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[2] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[2]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[3] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[3]+= YSUM(x); 
       )                                        
     return _mm_load_ps(o);
}

static inline __m128 _sse_div4_ps(__m128* _v, __m128* _u) {
// returns |v|/|u| 
   return _mm_sqrt_ps(_mm_div_ps(_sse_abs4_ps(_v),
				 _mm_add_ps(_sse_abs4_ps(_u),
					    _mm_set1_ps(1.e-24))));
}


static inline __m128 _sse_rnorm4_ps(__m128* _p) {
// return reciprocical norm: 1/|p|
  float x[4];
  float o[4];
  __m128* _a = _p;
  _NET(
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[0] = XSUM(x)+1.e-24;,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[0]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[1] = XSUM(x)+1.e-24;,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[1]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[2] = XSUM(x)+1.e-24;,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[2]+= YSUM(x); 
       )                                       
  _NET(                                      
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[3] = XSUM(x)+1.e-24;,
       _mm_storeu_ps(x,_mm_mul_ps(*_a,*_a));_a++;o[3]+= YSUM(x); 
       )                                        
     return _mm_div_ps(_mm_set1_ps(1.),_mm_sqrt_ps(_mm_load_ps(o)));
}

static inline float _sse_dot_ps(__m128* _a, __m128* _b) {
  float x[4];
  float out;
  _NET(
       _mm_storeu_ps(x,_mm_mul_ps(_a[0],_b[0])); out = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(_a[1],_b[1])); out+= YSUM(x);
       )
     return out;
}

static inline __m128 _sse_dot4_ps(__m128* _p, __m128* _q) {
  float x[4];
  float o[4];
  __m128* _o = (__m128*) o;
  __m128* _a = _p;
  __m128* _b = _q;
  _NET(
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[0] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[0]+= YSUM(x); 
       )                                        
  _NET(                                         
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[1] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[1]+= YSUM(x); 
       )                                        
  _NET(                                         
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[2] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[2]+= YSUM(x); 
       )                                        
  _NET(                                         
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[3] = XSUM(x);,
       _mm_storeu_ps(x,_mm_mul_ps(*_a++,*_b++));o[3]+= YSUM(x); 
       )
     return *_o;
}

static inline void _sse_add_ps(__m128* _a, __m128* _b) {
// _a += _b   
   _NET(_a[0] = _mm_add_ps(_a[0],_b[0]);,
        _a[1] = _mm_add_ps(_a[1],_b[1]);)
      return;
}

static inline void _sse_add_ps(__m128* _a, __m128* _b, __m128 _c) {
// _a += _b *_c  
   _NET(_a[0] = _mm_add_ps(_a[0],_mm_mul_ps(_b[0],_c));,
        _a[1] = _mm_add_ps(_a[1],_mm_mul_ps(_b[1],_c));)
      return;
}

static inline void _sse_add4_ps(__m128* _a, __m128* _b, __m128 _c) {
// _a++ += _b++ *c[0]  
// _a++ += _b++ *c[1]  
// _a++ += _b++ *c[2]  
// _a++ += _b++ *c[3]
   float c[4]; _mm_storeu_ps(c,_c); 
   __m128* _p = _a;
   __m128* _q = _b;
   _NET(*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps( c ))); _p++;,
	*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps( c ))); _p++;)
   _NET(*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+1))); _p++;,
	*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+1))); _p++;)
   _NET(*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+2))); _p++;,
	*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+2))); _p++;)
   _NET(*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+3))); _p++;,
	*_p = _mm_add_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+3))); _p++;)
   return;
}

static inline void _sse_sub_ps(__m128* _a, __m128* _b) {
// _a -= _b   
   _NET(_a[0] = _mm_sub_ps(_a[0],_b[0]);,
        _a[1] = _mm_sub_ps(_a[1],_b[1]);)
      return;
}

static inline void _sse_sub4_ps(__m128* _a, __m128* _b, __m128 _c) {
// _a++ -= _b++ *c[0]  
// _a++ -= _b++ *c[1]  
// _a++ -= _b++ *c[2]  
// _a++ -= _b++ *c[3]
   float c[4]; _mm_storeu_ps(c,_c); 
   __m128* _p = _a;
   __m128* _q = _b;
   _NET(*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps( c ))); _p++;,
	*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps( c ))); _p++;)
   _NET(*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+1))); _p++;,
	*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+1))); _p++;)
   _NET(*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+2))); _p++;,
	*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+2))); _p++;)
   _NET(*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+3))); _p++;,
	*_p = _mm_sub_ps(*_p,_mm_mul_ps(*_q++,_mm_load1_ps(c+3))); _p++;)
   return;
}

static inline void _sse_cpf_ps(float* a, __m128* _p) {
   _NET(_mm_storeu_ps(a,*_p);,
        _mm_storeu_ps(a+4,*(_p+1));)
      return;
}

static inline void _sse_cpf_ps(__m128* _a, __m128* _p) {
   _NET(*_a = *_p;, *(_a+1) = *(_p+1);)
}

static inline void _sse_cpf4_ps(__m128* _aa, __m128* _pp) {
   __m128* _a = _aa;
   __m128* _p = _pp;
   _NET(*_a++ = *_p++;, *_a++ = *_p++;)
   _NET(*_a++ = *_p++;, *_a++ = *_p++;)
   _NET(*_a++ = *_p++;, *_a++ = *_p++;)
   _NET(*_a++ = *_p++;, *_a++ = *_p++;)
}


static inline void _sse_cpf_ps(float* a, __m128* _p, float b) {
  __m128 _b = _mm_load1_ps(&b);
  _NET(_mm_storeu_ps(a,_mm_mul_ps(*_p,_b));,
       _mm_storeu_ps(a+4,_mm_mul_ps(*(_p+1),_b));)
     return;
}

static inline void _sse_cpf4_ps(float* aa, __m128* _pp) {
// copy data from pp to aa for 4 consecutive pixels 
   float*   a =  aa;
   __m128* _p = _pp;
   
  _NET(_mm_storeu_ps(a,*_p++); a+=4;,
       _mm_storeu_ps(a,*_p++); a+=4;)
  _NET(_mm_storeu_ps(a,*_p++); a+=4;,
       _mm_storeu_ps(a,*_p++); a+=4;)
  _NET(_mm_storeu_ps(a,*_p++); a+=4;,
       _mm_storeu_ps(a,*_p++); a+=4;)
  _NET(_mm_storeu_ps(a,*_p++); a+=4;,
       _mm_storeu_ps(a,*_p++); a+=4;)

     return;
}

static inline void _sse_cpf4_ps(__m128* _aa, __m128* _pp, __m128 _c) {
// multiply p by c and copy data from p to a for 4 consecutive pixels 
// calculate _a[0]=_p[0]*c[0] 
// calculate _a[1]=_p[1]*c[1] 
// calculate _a[2]=_p[2]*c[2] 
// calculate _a[3]=_p[3]*c[3]
   float c[4]; _mm_storeu_ps(c,_c); 
   __m128* _a = _aa;
   __m128* _p = _pp;
   
   _NET(*_a++ = _mm_mul_ps(*_p++,_mm_load1_ps( c ));,
        *_a++ = _mm_mul_ps(*_p++,_mm_load1_ps( c ));)
   _NET(*_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+1));,
        *_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+1));)
   _NET(*_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+2));,
        *_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+2));)
   _NET(*_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+3));,
        *_a++ = _mm_mul_ps(*_p++,_mm_load1_ps(c+3));)
     return;
}

static inline float _sse_nrg_ps(__m128* _u, float c, __m128* _v, float s, __m128* _a) {
// calculate b = (a - u*c - v*s) and return b*b
  float x[4];
  float out;
  __m128 _b;
  __m128 _c = _mm_load1_ps(&c);
  __m128 _s = _mm_load1_ps(&s);

  _NET(_b = _mm_sub_ps(_a[0], _mm_add_ps(_mm_mul_ps(*_u,_c), _mm_mul_ps(*_v,_s))); 
       _mm_storeu_ps(x,_mm_mul_ps(_b,_b)); out = XSUM(x);,
       _b = _mm_sub_ps(_a[1], _mm_add_ps(_mm_mul_ps(*(_u+1),_c), _mm_mul_ps(*(_v+1), _s))); 
       _mm_storeu_ps(x,_mm_mul_ps(_b,_b)); out+= YSUM(x);)

  return out/2.;
}    

static inline void _sse_rotadd_ps(__m128* _u, float c, __m128* _v, float s, __m128* _a) {
// calculate a += u*c + v*s
  __m128 _c = _mm_load1_ps(&c);
  __m128 _s = _mm_load1_ps(&s);
  _NET(
       _a[0] = _mm_add_ps(_a[0], _mm_add_ps(_mm_mul_ps(_u[0],_c), _mm_mul_ps(_v[0],_s)));, 
       _a[1] = _mm_add_ps(_a[1], _mm_add_ps(_mm_mul_ps(_u[1],_c), _mm_mul_ps(_v[1],_s)));
       ) 
     return;
}    

static inline float _sse_rotsub_ps(__m128* _u, float c, __m128* _v, float s, __m128* _a) {
// calculate a -= u*c + v*s and return a*a
  float x[4];
  float out;
  __m128 _c = _mm_load1_ps(&c);
  __m128 _s = _mm_load1_ps(&s);

  _NET(
       _a[0] = _mm_sub_ps(_a[0], _mm_add_ps(_mm_mul_ps(_u[0],_c), _mm_mul_ps(_v[0],_s))); 
       _mm_storeu_ps(x,_mm_mul_ps(_a[0],_a[0])); out = XSUM(x);,
       _a[1] = _mm_sub_ps(_a[1], _mm_add_ps(_mm_mul_ps(_u[1],_c), _mm_mul_ps(_v[1], _s))); 
       _mm_storeu_ps(x,_mm_mul_ps(_a[1],_a[1])); out+= YSUM(x);
       )

  return out;
}    

static inline void _sse_rotp_ps(__m128* u, float* c, __m128* v, float* s, __m128* a) {
// calculate a = u*c + v*s
  _NET(
       a[0] = _mm_add_ps(_mm_mul_ps(u[0],_mm_load1_ps(c)), _mm_mul_ps(v[0],_mm_load1_ps(s)));, 
       a[1] = _mm_add_ps(_mm_mul_ps(u[1],_mm_load1_ps(c)), _mm_mul_ps(v[1],_mm_load1_ps(s))); 
       ) 
}    

static inline void _sse_rotm_ps(__m128* u, float* c, __m128* v, float* s, __m128* a) {
// calculate a = u*c - v*s
  _NET(
       a[0] = _mm_sub_ps(_mm_mul_ps(u[0],_mm_load1_ps(c)), _mm_mul_ps(v[0],_mm_load1_ps(s)));, 
       a[1] = _mm_sub_ps(_mm_mul_ps(u[1],_mm_load1_ps(c)), _mm_mul_ps(v[1],_mm_load1_ps(s))); 
       ) 
}    

static inline __m128 _sse_rotp_ps(__m128 _u, __m128 _c, __m128 _v, __m128 _s) {
// return u*c + v*s
   return _mm_add_ps(_mm_mul_ps(_u,_c), _mm_mul_ps(_v,_s));
}    

static inline __m128 _sse_rotm_ps(__m128 _u, __m128 _c, __m128 _v, __m128 _s) {
// return u*c - v*s
   return _mm_sub_ps(_mm_mul_ps(_u,_c), _mm_mul_ps(_v,_s));
}    

static inline void _sse_rot4p_ps(__m128* _u, __m128* _c, __m128* _v, __m128* _s, __m128* _a) {
// calculate a[0] = u[0]*c[0] + v[0]*s[0]
// calculate a[1] = u[1]*c[1] + v[1]*s[1]
// calculate a[2] = u[2]*c[2] + v[2]*s[2]
// calculate a[3] = u[3]*c[3] + v[3]*s[3]
   float c[4];
   float s[4];
   _mm_storeu_ps(c,*_c);
   _mm_storeu_ps(s,*_s);
   __m128* u = _u;
   __m128* v = _v;
   __m128* a = _a;
  _NET(
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps( c )), _mm_mul_ps(*v++,_mm_load1_ps( s )));, 
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps( c )), _mm_mul_ps(*v++,_mm_load1_ps( s ))); 
       ) 
  _NET(
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+1)), _mm_mul_ps(*v++,_mm_load1_ps(s+1)));, 
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+1)), _mm_mul_ps(*v++,_mm_load1_ps(s+1))); 
       ) 
  _NET(
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+2)), _mm_mul_ps(*v++,_mm_load1_ps(s+2)));, 
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+2)), _mm_mul_ps(*v++,_mm_load1_ps(s+2))); 
       ) 
  _NET(
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+3)), _mm_mul_ps(*v++,_mm_load1_ps(s+3)));, 
       *a++ = _mm_add_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+3)), _mm_mul_ps(*v++,_mm_load1_ps(s+3))); 
       ) 
}    

static inline void _sse_rot4m_ps(__m128* _u, __m128* _c, __m128* _v, __m128* _s, __m128* _a) {
// calculate a[0] = u[0]*c[0] - v[0]*s[0]
// calculate a[1] = u[1]*c[1] - v[1]*s[1]
// calculate a[2] = u[2]*c[2] - v[2]*s[2]
// calculate a[3] = u[3]*c[3] - v[3]*s[3]
   float c[4];
   float s[4];
   _mm_storeu_ps(c,*_c);
   _mm_storeu_ps(s,*_s);
   __m128* u = _u;
   __m128* v = _v;
   __m128* a = _a;
  _NET(
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps( c )), _mm_mul_ps(*v++,_mm_load1_ps( s )));, 
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps( c )), _mm_mul_ps(*v++,_mm_load1_ps( s ))); 
       ) 
  _NET(
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+1)), _mm_mul_ps(*v++,_mm_load1_ps(s+1)));, 
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+1)), _mm_mul_ps(*v++,_mm_load1_ps(s+1))); 
       ) 
  _NET(
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+2)), _mm_mul_ps(*v++,_mm_load1_ps(s+2)));, 
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+2)), _mm_mul_ps(*v++,_mm_load1_ps(s+2))); 
       ) 
  _NET(
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+3)), _mm_mul_ps(*v++,_mm_load1_ps(s+3)));, 
       *a++ = _mm_sub_ps(_mm_mul_ps(*u++,_mm_load1_ps(c+3)), _mm_mul_ps(*v++,_mm_load1_ps(s+3))); 
       ) 
}    

static inline void _sse_point_ps(__m128** _p, float** p, short** m, int l, int n) {
// point 0-7 __m128 pointers to first network pixel
   NETX(_p[0] = (__m128*) (p[0] + m[0][l]*n);,
        _p[1] = (__m128*) (p[1] + m[1][l]*n);,
        _p[2] = (__m128*) (p[2] + m[2][l]*n);,
        _p[3] = (__m128*) (p[3] + m[3][l]*n);,
        _p[4] = (__m128*) (p[4] + m[4][l]*n);,
        _p[5] = (__m128*) (p[5] + m[5][l]*n);,
        _p[6] = (__m128*) (p[6] + m[6][l]*n);,
        _p[7] = (__m128*) (p[7] + m[7][l]*n);)
      return;
}   

static inline __m128 _sse_sum_ps(__m128** _p) {
   __m128 _q = _mm_setzero_ps();
   NETX(_q = _mm_add_ps(_q, *_p[0]);,
        _q = _mm_add_ps(_q, *_p[1]);,
        _q = _mm_add_ps(_q, *_p[2]);,
        _q = _mm_add_ps(_q, *_p[3]);,
        _q = _mm_add_ps(_q, *_p[4]);,
        _q = _mm_add_ps(_q, *_p[5]);,
        _q = _mm_add_ps(_q, *_p[6]);,
        _q = _mm_add_ps(_q, *_p[7]);)
      return _q;
} 

static inline __m128 _sse_cut_ps(__m128* _pE, __m128** _pe, __m128 _Es, __m128 _cmp) {
   NETX(_cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[0]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[1]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[2]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[3]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[4]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[5]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[6]++),_Es));,
        _cmp = _mm_and_ps(_cmp,_mm_cmpge_ps(_mm_sub_ps(*_pE, *_pe[7]++),_Es));)
      return _cmp;
}

static inline void _sse_minSNE_ps(__m128* _pE, __m128** _pe, __m128* _es) {
// put pixel minimum subnetwork energy in _es
// input _es should be initialized to _pE before call
   NETX(*_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[0]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[1]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[2]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[3]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[4]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[5]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[6]++));,
        *_es = _mm_min_ps(*_es,_mm_sub_ps(*_pE, *_pe[7]++));
        )
}

static inline float _sse_maxE_ps(__m128* _a, __m128* _A) {
// given input 00 and 90 data vectors
// returns energy of dominant detector (max energy)
   float out;
   float x[4];
   __m128 _o1;
   __m128 _o2 = _mm_setzero_ps();
   _NET(
        _o1 = _mm_add_ps(_mm_mul_ps(_a[0],_a[0]),_mm_mul_ps(_A[0],_A[0]));,
        _o2 = _mm_add_ps(_mm_mul_ps(_a[1],_a[1]),_mm_mul_ps(_A[1],_A[1])); 
        )
   _o1 = _mm_max_ps(_o1,_o2); _mm_storeu_ps(x,_o1); out=x[0];
   if(out<x[1]) out=x[1];
   if(out<x[2]) out=x[2];
   if(out<x[3]) out=x[3];
   return out;
}

static inline void _sse_ort4_ps(__m128* _u, __m128* _v, __m128* _s, __m128* _c) {
// orthogonalize vectors _u and _v: take vectors u and v, 
// make them orthogonal, calculate rotation phase
// fill in sin and cos in _s and _c respectively 
   static const __m128 sm = _mm_set1_ps(-0.f);                           // -0.f = 1 << 31
   static const __m128 _o = _mm_set1_ps(1.e-24); 
   static const __m128 _0 = _mm_set1_ps(0.); 
   static const __m128 _1 = _mm_set1_ps(1.); 
   static const __m128 _2 = _mm_set1_ps(2.); 
   __m128 _n,_m,gI,gR,_p,_q;
   gI = _mm_mul_ps(_sse_dot4_ps(_u,_v),_2);                              // ~sin(2*psi) or 2*u*v
   gR = _mm_sub_ps(_sse_dot4_ps(_u,_u),_sse_dot4_ps(_v,_v));             // u^2-v^2
   _p = _mm_and_ps(_mm_cmpge_ps(gR,_0),_1);                              // 1 if gR>0. or 0 if gR<0.  
   _q = _mm_sub_ps(_1,_p);                                               // 0 if gR>0. or 1 if gR<0.  
   _n = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(gI,gI),_mm_mul_ps(gR,gR)));    // gc
   gR = _mm_add_ps(_mm_andnot_ps(sm,gR),_mm_add_ps(_n,_o));              // gc+|gR|+eps
   _n = _mm_add_ps(_mm_mul_ps(_2,_n),_o);                                // 2*gc + eps
   gI = _mm_div_ps(gI,_n);                                               // sin(2*psi)
   _n = _mm_sqrt_ps(_mm_div_ps(gR,_n));                                  // sqrt((gc+|gR|)/(2gc+eps))
   _m = _mm_and_ps(_mm_cmpge_ps(gI,_0),_1);                              // 1 if gI>0. or 0 if gI<0.  
   _m = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(_m,_2),_1),_n);                 // _n if gI>0 or -_n if gI<0   
  *_s = _mm_add_ps(_mm_mul_ps(_q,_m),_mm_mul_ps(_p,_mm_div_ps(gI,_n)));  // sin(psi)
   gI = _mm_andnot_ps(sm,gI);                                            // |gI|
  *_c = _mm_add_ps(_mm_mul_ps(_p,_n),_mm_mul_ps(_q,_mm_div_ps(gI,_n)));  // cos(psi)
   return;
}

static inline void _sse_ort4_ps(__m128* _s, __m128* _c, __m128 _r) {
// solves equation sin(2a) * _c - cos(2a) * _s = 0
// input:  _s ~ sin(2a), _c ~ cos(2a) (not normalized), _r - regulator
// regularized  _C = |_c| + _r*eps
// norm:        _n =  sqrt(_s*_s+_C*_C) 
//         cos(2a) = _C/_n 
//         sin(2a) = _s/_n
// output  _s = sin(a),  _c = cos(a)
// algorithm:
// _c>0:   cos(a) = sqrt([1+cos(2a)]/2);           sin(a) = sin(2a)/2/cos(a)
// _c<0:   sin(a) = sign[_s]*sqrt([1+cos(2a)]/2);  cos(a) = |sin(2a)|/2/|sin(a)|
   static const __m128 sm = _mm_set1_ps(-0.f);                           // -0.f = 1 << 31
   static const __m128 _0 = _mm_set1_ps(0.); 
   static const __m128 _5 = _mm_set1_ps(0.5); 
   static const __m128 _1 = _mm_set1_ps(1.); 
   static const __m128 _2 = _mm_set1_ps(2.); 
   __m128 _n,_m,_C,_S,_p,_q;
   _r = _mm_mul_ps(_mm_add_ps(_1,_r),_mm_set1_ps(1.e-6));                // regulator
   _m = _mm_and_ps(_mm_cmpge_ps(*_s,_0),_1);                             // 1 if _S>0. or 0 if _S<0.  
   _m = _mm_sub_ps(_mm_mul_ps(_m,_2),_1);                                // sign(_s)   
   _p = _mm_and_ps(_mm_cmpge_ps(*_c,_0),_1);                             // 1 if _c>0. or 0 if _c<0.  
   _q = _mm_sub_ps(_1,_p);                                               // 0 if _c>0. or 1 if _c<0.  
   _C = _mm_add_ps(_mm_andnot_ps(sm,*_c),_r);                            // |_c|+regulator
   _n = _mm_add_ps(_mm_mul_ps(*_s,*_s),_mm_mul_ps(_C,_C));               // norm^2 for sin(2a) and cos(2a) 
   _n = _mm_div_ps(_1,_mm_mul_ps(_mm_sqrt_ps(_n),_2));                   // 0.5/norm for sin(2a) and cos(2a) 
   _C = _mm_sqrt_ps(_mm_add_ps(_5,_mm_mul_ps(_C,_n)));                   // |cos(a)| 
   _S = _mm_div_ps(_mm_mul_ps(*_s,_n),_C);                               // sin(a)*cos(a)/|cos(a)|
  *_s = _mm_add_ps(_mm_mul_ps(_p,_S),_mm_mul_ps(_q,_mm_mul_ps(_C,_m)));  // sin(psi)
  *_c = _mm_add_ps(_mm_mul_ps(_p,_C),_mm_mul_ps(_q,_mm_mul_ps(_S,_m)));  // cos(psi)
   return;
}

static inline void _sse_dpf4_ps(__m128* _Fp, __m128* _Fx, __m128* _fp, __m128* _fx) {
// transformation to DPF for 4 consecutive pixels.
// rotate vectors Fp and Fx into DPF: fp and fx
   __m128 _c, _s;
   _sse_ort4_ps(_Fp,_Fx,&_s,&_c);                                        // get sin and cos
   _sse_rot4p_ps(_Fp,&_c,_Fx,&_s,_fp);                                   // get fp=Fp*c+Fx*s  
   _sse_rot4m_ps(_Fx,&_c,_Fp,&_s,_fx);                                   // get fx=Fx*c-Fp*s 
   return;
}

static inline void _sse_pnp4_ps(__m128* _fp, __m128* _fx, __m128* _am, __m128* _AM, __m128* _u, __m128* _v) {
// projection to network plane (pnp)
// _fp and _fx must be in DPF
// project vectors _am and _AM on the network plane _fp,_fx
// returns square of the network alignment factor 
// c = xp/gp - cos of rotation to PCF
// s = xx/gx - sin of rotation to PCF
   static const __m128 _o = _mm_set1_ps(1.e-24);
   static const __m128 _1 = _mm_set1_ps(1.0);
   __m128 gp = _mm_div_ps(_1,_mm_add_ps(_sse_dot4_ps(_fp,_fp),_o)); // 1/fp*fp
   _sse_sub4_ps(_fx,_fp,_mm_mul_ps(gp,_sse_dot4_ps(_fp,_fx)));      // check fx _|_ to fp
   __m128 gx = _mm_div_ps(_1,_mm_add_ps(_sse_dot4_ps(_fx,_fx),_o)); // 1/fx*fx  
   __m128 cc = _mm_mul_ps(_sse_dot4_ps(_fp,_am),gp);                // cos
   __m128 ss = _mm_mul_ps(_sse_dot4_ps(_fx,_am),gx);                // sin
   _sse_rot4p_ps(_fp,&cc,_fx,&ss,_u);                               // get vector u   
          cc = _mm_mul_ps(_sse_dot4_ps(_fp,_AM),gp);                // cos
          ss = _mm_mul_ps(_sse_dot4_ps(_fx,_AM),gx);                // sin
   _sse_rot4p_ps(_fp,&cc,_fx,&ss,_v);                               // get vector v   
   return;
}

static inline void _sse_dsp4_ps(__m128* u, __m128* v, __m128* _am, __m128* _AM, __m128* _u, __m128* _v) {
// dual stream phase (dsp) transformation
// take projection vectors uu and vu, 
// make them orthogonal, calculate dual stream phase
// apply phase transformation both to data and projections 
   __m128 _c, _s;
   _sse_ort4_ps(u,v,&_s,&_c);            // get sin and cos
   _sse_rot4p_ps(u,&_c,v,&_s,_u);        // get 00 response  
   _sse_rot4m_ps(v,&_c,u,&_s,_v);        // get 90 response 
   _sse_rot4p_ps(_am,&_c,_AM,&_s,u);     // get 00 data vector 
   _sse_rot4m_ps(_AM,&_c,_am,&_s,v);     // get 90 data vector
   _sse_cpf4_ps(_am,u);
   _sse_cpf4_ps(_AM,v);
   return;
}

static inline __m128 _sse_ei4_ps(__m128* _u, __m128 _L) {
// returns incoherent energy for vector u
// calculates sum: u[k]*u[k]*u[k]*u[k]/L
// where L should be |u|^2  
  float x[4];
  float o[4];
  __m128* _a = _u;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[0]+= YSUM(x);
       )                                                               
  _NET(                                                                
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[1]+= YSUM(x);
       )                                                               
  _NET(                                                                
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[2]+= YSUM(x);
       )                                                               
  _NET(                                                                
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));_a++;o[3]+= YSUM(x);
       )                                        
     return _mm_div_ps(_mm_load_ps(o),_mm_add_ps(_L,_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_ei4xx_ps(__m128* _x, __m128* _u, __m128 _L) {
// returns incoherent energy for vectors p,q
// _x - data vector
// _u - projection vector on the network plane
// calculates sum: x[k]*x[k]*u[k]*u[k]/L  
  float x[4];
  float o[4];
  __m128* _a = _x;
  __m128* _b = _u;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3]+= YSUM(x);
       )                                        
     return _mm_div_ps(_mm_load_ps(o),_mm_add_ps(_L,_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_ei4xu_ps(__m128* _x, __m128* _u, __m128 _L) {
// returns incoherent energy for vectors p,q
// _x - data vector
// _u - projection vector on the network plane
// calculates sum: x[k]*u[k]*u[k]*u[k]/L  
  float x[4];
  float o[4];
  __m128* _a = _x;
  __m128* _b = _u;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[0]+= YSUM(x);
       )                                                                               
  _NET(                                                                                
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[1]+= YSUM(x);
       )                                                                               
  _NET(                                                                                
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[2]+= YSUM(x);
       )                                                                               
  _NET(                                                                                
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_b,*_b);_mm_storeu_ps(x,_mm_mul_ps(_c,_mm_mul_ps(*_a++,*_b++)));o[3]+= YSUM(x);
       )                                        
     return _mm_div_ps(_mm_load_ps(o),_mm_add_ps(_L,_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_null4_ps(__m128* _p, __m128* _q) {
// returns global NULL energy: |E-L-Ei+Li| + |Ei-Li|
// E = |p|^2; Ei = sum{p[k]^4}/|p|^2
// L = |q|^2; Li = sum{q[k]^4}/|q|^2
  __m128 _sm = _mm_set1_ps(-0.f);            // sign mask: -0.f = 1 << 31
  __m128 _pe = _sse_abs4_ps(_p);             // get total energy for _p
  __m128 _pi = _sse_ei4_ps(_p,_pe);          // get incoherent energy for _p
  __m128 _qe = _sse_abs4_ps(_q);             // get total energy for _q
  __m128 _qi = _sse_ei4_ps(_q,_qe);          // get incoherent energy for _q
  _pi = _mm_sub_ps(_pi,_qi);                 // Ei-Li
  _pe = _mm_sub_ps(_mm_sub_ps(_pe,_qe),_pi); // (E-L) - (Ei-Li)
  return _mm_add_ps(_mm_andnot_ps(_sm,_pi),_mm_andnot_ps(_sm,_pe));
}


static inline __m128 _sse_ecoh4_ps(__m128* _p, __m128* _q, __m128 _L) {
// returns coherent energy given
// _p - data vector
// _q - network response
// calculates vector: u[k] = p[k]*q[k]
// returns:           L - u[k]*u[k]/L,
// where L should be (q,q)^2  
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128* _b = _q;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3]+= YSUM(x);
       )                                        
     return _mm_sub_ps(_L,_mm_div_ps(_mm_load_ps(o),_mm_add_ps(_L,_mm_set1_ps(1.e-12))));
}

static inline __m128 _sse_ind4_ps(__m128* _p, __m128 _L) {
// returns vector index
// _p - data vector  and L is its norm^2
// calculates vector: u[k] = p[k]*p[k]
// returns:           u[k]*u[k]/L/L,
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_a,*_a); _a++; _mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3]+= YSUM(x);
       )                                        
     _c = _mm_add_ps(_mm_mul_ps(_L,_L),_mm_set1_ps(1.e-16));
     return _mm_div_ps(_mm_load_ps(o),_c);
}

static inline __m128 _sse_ecoh4_ps(__m128* _p, __m128* _q) {
// returns coherent part of (p,q)^2
// calculates reduced unity vector: u[k] = q * |q| /(p,q)
// returns:  (p,u)^2 - sum_k{p[k]*p[k]*u[k]*u[k]} 
//      =    |q|^2 - p[k]^2 * q[k]^2 * |q|^2/(p,q)^2
//      =    |q|^2 (1 - sum_k{p[k]^2*q[k]^2}/(p,q)^2  
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128* _b = _q;
  __m128  _c;
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[0]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[1]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[2]+= YSUM(x);
       )                                        
  _NET(
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3] = XSUM(x);,
       _c = _mm_mul_ps(*_a++,*_b++);_mm_storeu_ps(x,_mm_mul_ps(_c,_c));o[3]+= YSUM(x);
       ) 

  _c = _sse_dot4_ps(_p,_q);                                   // scalar product
  _c = _mm_add_ps(_mm_set1_ps(1.e-12),_mm_mul_ps(_c,_c));     // (p,q)^2+1.e-12
  _c = _mm_div_ps(_mm_load_ps(o),_c);                         // get incoherent part
  
  return _mm_mul_ps(_sse_abs4_ps(_q),_mm_sub_ps(_mm_set1_ps(1.),_c));
}

static inline __m128 _sse_ed4_ps(__m128* _p, __m128* _q, __m128 _L) {
// returns energy disbalance
// _p - data vector
// _q - network response
// calculates sum: 0.5*(p[k]*q[k]-q[k]*q[k])^2  / L
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128* _b = _q;
  __m128  _aa;
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[0] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[0]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[1] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[1]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[2] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[2]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[3] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[3]+= YSUM(x);
       )
     _aa = _mm_mul_ps(_mm_load_ps(o),_mm_set1_ps(0.5));
     return _mm_div_ps(_aa,_mm_add_ps(_L,_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_ed4_ps(__m128* _p, __m128* _q) {
// returns energy disbalance
// _p - data vector
// _q - network response (may not be a projection)
// calculates sum: 0.5*sum_k{(p[k]*q[k]-q[k]*q[k])^2} * (q,q)/(p,q)^2
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128* _b = _q;
  __m128  _aa;
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[0] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[0]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[1] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[1]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[2] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[2]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[3] = XSUM(x);,
       _aa=_mm_sub_ps(_mm_mul_ps(*_a,*_b),_mm_mul_ps(*_b,*_b));
       _mm_storeu_ps(x,_mm_mul_ps(_aa,_aa));_a++;_b++;o[3]+= YSUM(x);
       )
  _aa = _sse_dot4_ps(_p,_q);                                   // scalar product
  _aa = _mm_add_ps(_mm_set1_ps(1.e-12),_mm_mul_ps(_aa,_aa));   // (p,q)^2+1.e-12
  _aa = _mm_div_ps(_sse_abs4_ps(_q),_aa);                      // (q,q)/(p,q)^2
  
  return _mm_mul_ps(_aa,_mm_mul_ps(_mm_load_ps(o),_mm_set1_ps(0.5)));
}

static inline __m128 _sse_ed4i_ps(__m128* _p, __m128* _q, __m128 _L) {
// returns incoherent part of energy disbalance
// _p - projection amplitude on the network
// calculates vector: v[k] = p[k]*p[k] * (q[k]*q[k]-p[k]*p[k])
// returns:           Sum_k{v[k]/L}  
  float x[4];
  float o[4];
  __m128* _a = _p;
  __m128* _b = _q;
  __m128  _aa,_bb;
  _NET(
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[0] = XSUM(x);,
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[0]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[1] = XSUM(x);,
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[1]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[2] = XSUM(x);,
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[2]+= YSUM(x);
       )                                        
  _NET(
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[3] = XSUM(x);,
       _aa=_mm_mul_ps(*_a,*_b); _bb=_mm_mul_ps(*_b,*_b);
       _mm_storeu_ps(x,_mm_mul_ps(_bb,_mm_sub_ps(_aa,_bb))); 
       _a++; _b++; o[3]+= YSUM(x);
       )                                        
     _aa = _mm_mul_ps(_mm_load_ps(o),_mm_set1_ps(2.));
     return _mm_div_ps(_aa,_mm_add_ps(_L,_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_like4_ps(__m128* _f, __m128* _a, __m128* _A) {
// input ff - antenna pattern (f+ or fx) in DPF
// input am,AM - network amplitude vectors
// returns: (xp*xp+XP*XP)/|f+|^2 or (xx*xx+XX*XX)/|fx|^2
   __m128 xx = _sse_dot4_ps(_f,_a);                                 // fp*am
   __m128 XX = _sse_dot4_ps(_f,_A);                                 // fp*AM
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
   return _mm_div_ps(xx,_mm_add_ps(_sse_dot4_ps(_f,_f),_mm_set1_ps(1.e-12)));
}

static inline __m128 _sse_like4_ps(__m128* fp, __m128* fx, __m128* am, __m128* AM, __m128 _D) {
// input fp,fx - antenna patterns in DPF
// input am,AM - network amplitude vectors
// input _D - (delta) - hard regulator
// returns: (xp*xp+XP*XP)/|f+|^2+(xx*xx+XX*XX)/(|fx|^2+delta)
   __m128 xp = _sse_dot4_ps(fp,am);                                 // fp*am
   __m128 XP = _sse_dot4_ps(fp,AM);                                 // fp*AM
   __m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
   __m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
   __m128 gp = _mm_add_ps(_sse_dot4_ps(fp,fp),_mm_set1_ps(1.e-12)); // fx*fx + epsilon 
   __m128 gx = _mm_add_ps(_sse_dot4_ps(fx,fx),_D);                  // fx*fx + delta 
          xp = _mm_add_ps(_mm_mul_ps(xp,xp),_mm_mul_ps(XP,XP));     // xp=xp*xp+XP*XP
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
   return _mm_add_ps(_mm_div_ps(xp,gp),_mm_div_ps(xx,gx));          // regularized projected energy
}

static inline __m128 _sse_like4_ps(__m128* fp, __m128* fx, __m128* am, __m128* AM) {
// input fp,fx - antenna patterns in DPF
// input am,AM - network amplitude vectors
// returns: (xp*xp+XP*XP)/|f+|^2+(xx*xx+XX*XX)/(|fx|^2)
   __m128 xp = _sse_dot4_ps(fp,am);                                 // fp*am
   __m128 XP = _sse_dot4_ps(fp,AM);                                 // fp*AM
   __m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
   __m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
   __m128 gp = _mm_add_ps(_sse_dot4_ps(fp,fp),_mm_set1_ps(1.e-12)); // fx*fx + epsilon 
   __m128 gx = _mm_add_ps(_sse_dot4_ps(fx,fx),_mm_set1_ps(1.e-12)); // fx*fx + epsilon 
          xp = _mm_add_ps(_mm_mul_ps(xp,xp),_mm_mul_ps(XP,XP));     // xp=xp*xp+XP*XP
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
   return _mm_add_ps(_mm_div_ps(xp,gp),_mm_div_ps(xx,gx));          // regularized projected energy
}

static inline __m128 _sse_like4w_ps(__m128* fp, __m128* fx, __m128* am, __m128* AM) {
// input fp,fx - antenna patterns in DPF
// input am,AM - network amplitude vectors
// returns: [(xp*xp+XP*XP)+(xx*xx+XX*XX)]/|f+|^2
   __m128 xp = _sse_dot4_ps(fp,am);                                 // fp*am
   __m128 XP = _sse_dot4_ps(fp,AM);                                 // fp*AM
   __m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
   __m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
   __m128 gp = _mm_add_ps(_sse_dot4_ps(fp,fp),_mm_set1_ps(1.e-9));  // fp*fp + epsilon 
          xp = _mm_add_ps(_mm_mul_ps(xp,xp),_mm_mul_ps(XP,XP));     // xp=xp*xp+XP*XP
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
   return _mm_div_ps(_mm_add_ps(xp,xx),gp);                         // regularized projected energy
}

static inline __m128 _sse_like4_ps(__m128* am, __m128* AM) {
// input am,AM - network projection vectors
   return _mm_add_ps(_mm_add_ps(_sse_abs4_ps(am),_sse_abs4_ps(AM)),_mm_set1_ps(1.e-12));
}

static inline __m128 _sse_reg4x_ps(__m128 _L, __m128* fx, __m128* am, __m128* AM, __m128 _D) {
// x regulator (incoherent)
// input _L - non-regulated likelihood
// input fx - antenna pattern in DPF
// input am,AM - network amplitude vectors
// input _D - (delta) - regulator
// returns: (delta*Lx/L-|fx^2|)/|fx^2+delta|
   static const __m128 _o = _mm_set1_ps(1.e-12);
   __m128 FF = _mm_add_ps(_sse_dot4_ps(fx,fx),_o);                  // fx*fx
   __m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
   __m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
          xx = _mm_div_ps(_mm_mul_ps(xx,_D),_mm_mul_ps(_L,FF));     // (Lx/L)*delta
   return _mm_div_ps(_mm_sub_ps(FF,xx),_mm_add_ps(FF,_D));          // [|fx|^2-(Lx/L)*delta]/|fx^2+delta|
}

static inline __m128 _sse_nind4_ps(__m128* _am, __m128* _AM) {
// calculates network index
// input fx - antenna pattern in DPF
// input am,AM - network projection vectors
   __m128 _ll = _sse_abs4_ps(_am);                            // L00
   __m128 _LL = _sse_abs4_ps(_AM);                            // L00
   __m128 _ei = _sse_ei4_ps(_am,_ll);                         // 00 incoherent
   __m128 _EI = _sse_ei4_ps(_AM,_LL);                         // 90 incoherent
   _ll = _mm_add_ps(_mm_add_ps(_ll,_LL),_mm_set1_ps(1.e-12)); // standard likelihood
   return _mm_div_ps(_mm_add_ps(_ei,_EI),_ll);                // network index
}

static inline void _sse_pol4_ps(__m128* _fp, __m128* _fx, __m128* _v, double* r, double* a) {
// calculates the polar coordinates of the input vector v in the DPF frame 
// input _fp,_fx - antenna patterns in DPF
// input _v      - network projection vector pointer
// input r,a     - polar coordinates (r : radius, a : angle in radians)
//                 r,a are pointers to 4 dimensional float arrays

   __m128 _ss,_cc;
   __m128 _oo = _mm_set1_ps(1.e-12);
   float rpol[4],cpol[4],spol[4];

   _cc = _sse_abs4_ps(_fp);                       // |fp|^2
   _cc = _mm_add_ps(_mm_sqrt_ps(_cc),_oo);        // |fp|
   _cc = _mm_div_ps(_sse_dot4_ps(_fp,_v),_cc);    // (fp,v)/|fp|
   _mm_storeu_ps(cpol,_cc);

   _ss = _sse_abs4_ps(_fx);                       // |fx|^2
   _ss = _mm_add_ps(_mm_sqrt_ps(_ss),_oo);        // |fx|
   _ss = _mm_div_ps(_sse_dot4_ps(_fx,_v),_ss);    // (fx,v)/|fx|
   _mm_storeu_ps(spol,_ss);

   _mm_storeu_ps(rpol,_sse_abs4_ps(_v));          // |v|^2

   for(int n=0;n<4;n++) {
     r[n] = sqrt(rpol[n]);                        // |v|
     a[n] = atan2(spol[n],cpol[n]);               // atan2(spol,cpol)
   }

   return;
}

/*
static inline __m128 _sse_reg4_ps(__m128* fp, __m128* fx, __m128* am, __m128* AM, __m128 _D, __m128 _G) {
// input fp,fx - antenna patterns in DPF
// input am,AM - network amplitude vectors
// input _D - (delta) - Tikhonov constraint
// input _G - (gamma) - polarization constraint
// calculate S^2 = sin(psi)^2, where psi is the angle between f+ and (xi+XI).
// xi is the projection of x00 (_am) on the f+,fx plane
// XI is the projection of x90 (_AM) on the f+,fx plane
// is asymmetry s/(s+S)<T, return 0  otherwise return |xi|^2+|XI|^2
// where  s^2 = 4*|f+|*|fx|/(|f+|^2+|fx|^2)^2 and
// and    S^2 = 1-(|f+,xi|-|f+,XI|)^2/|f+|^2/(x00^2+x90^2)
// the actual condition checked: s^2 < G*S^2

   static const __m128 sm = _mm_set1_ps(-0.f);                      // sign mask: -0.f = 1 << 31
   static const __m128 _1 = _mm_set1_ps(1);
   static const __m128 _2 = _mm_set1_ps(2);
   static const __m128 _o = _mm_set1_ps(1.e-12);                    // epsilon
   __m128 xp = _sse_dot4_ps(fp,am);                                 // fp*am
   __m128 XP = _sse_dot4_ps(fp,AM);                                 // fp*AM
   __m128 xx = _sse_dot4_ps(fx,am);                                 // fx*am
   __m128 XX = _sse_dot4_ps(fx,AM);                                 // fx*AM
   __m128 cc = _mm_sub_ps(_mm_mul_ps(xx,XP),_mm_mul_ps(xp,XX));     // cc=xx*XP-xp*XX
   __m128 ss = _mm_add_ps(_mm_mul_ps(xx,xp),_mm_mul_ps(XX,XP));     // ss=xx*xp+XX*XP
          cc = _mm_andnot_ps(sm,_mm_mul_ps(_mm_mul_ps(cc,ss),_2));  // cc=|2*cc*ss| (chk-ed)
          xp = _mm_add_ps(_mm_mul_ps(xp,xp),_mm_mul_ps(XP,XP));     // xp=xp*xp+XP*XP
          xx = _mm_add_ps(_mm_mul_ps(xx,xx),_mm_mul_ps(XX,XX));     // xx=xx*xx+XX*XX
   __m128 gp = _sse_dot4_ps(fp,fp);                                 // fp*fp
   __m128 gx = _mm_add_ps(_sse_dot4_ps(fx,fx),_o);                  // fx*fx + epsilon 
          XX = _mm_add_ps(_mm_mul_ps(xx,gp),_mm_mul_ps(xp,gx));     // XX=xx*gp+xp*gx
          ss = _mm_add_ps(_sse_dot4_ps(am,am),_sse_dot4_ps(AM,AM)); // total energy
          ss = _mm_add_ps(ss,_D);                                   // total energy + delta
          cc = _mm_div_ps(cc,_mm_add_ps(_mm_mul_ps(xx,ss),_o));     // second psi term - added
          ss = _mm_div_ps(xp,_mm_add_ps(_mm_mul_ps(gp,ss),_o));     // first psi term - subtracted
          ss = _mm_mul_ps(_G,_mm_add_ps(_mm_sub_ps(_1,ss),cc));     // G*sin(psi)^2
          cc = _mm_div_ps(_2,_mm_add_ps(gp,gx));                    // cc=2*(1-T)/(gp+gx)
          cc = _mm_mul_ps(_mm_mul_ps(cc,cc),_mm_mul_ps(gp,gx));     // right is ready for comparison
          //cc = _mm_sqrt_ps(cc); 
          //ss = _mm_sqrt_ps(ss);
          //cc = _mm_div_ps(cc,_mm_add_ps(ss,cc));
          cc = _mm_and_ps(_mm_cmpge_ps(cc,ss),_1);                  // 1 if cc>ss or 0 if cc<ss  
          XX = _mm_div_ps(XX,_mm_mul_ps(_mm_add_ps(gp,_D),gx));     // std L with Tikhonov regulator
   return XX;                                        // final projected energy
   return _mm_mul_ps(XX,cc);                                        // final projected energy
}


static inline __m128 _sse_asy4_ps(__m128* _fp, __m128* _fx, __m128* _aa, __m128* _AA) {
// calculate sample network asymmetry
// _a - alignment factors
// _e - ellipticity factors
// _c - cos between f+ and 00 projection
// return network asymmetry
   static const __m128 sm = _mm_set1_ps(-0.f);                       // sign mask: -0.f = 1 << 31
   static const __m128 _1 = _mm_set1_ps(1.);
   static const __m128 _2 = _mm_set1_ps(2.);
   static const __m128 _o = _mm_set1_ps(1.e-4);
   __m128 _a = _sse_dot4_ps(_fp,_fp);                                // |fp|^2
   __m128 _e = _sse_dot4_ps(_aa,_aa);                                // |x00|^2
   __m128 _c = _sse_dot4_ps(_fp,_aa);                                // (fp,aa)
          _c = _mm_div_ps(_mm_mul_ps(_c,_c),_mm_mul_ps(_a,_e));      // cos^2(psi)
          _a = _mm_add_ps(_mm_div_ps(_sse_dot4_ps(_fx,_fx),_a),_o);  // a = (|fx|/|fp|)^2
          _e = _mm_div_ps(_sse_dot4_ps(_AA,_AA),_mm_mul_ps(_a,_e));  // (|x90|/|x00|)^2/a
          _e = _mm_div_ps(_1,_mm_add_ps(_1,_mm_sqrt_ps(_e)));        // ee = 1/(1+sqrt(ee))
          _c = _mm_mul_ps(_mm_sub_ps(_1,_c),_mm_add_ps(_1,_a));      // _c = s*s*(1+a)
          _a = _mm_div_ps(_c,_mm_mul_ps(_2,_a));                     // _a = s*s*(1+a)/(2*a)
          _a = _mm_div_ps(_1,_mm_add_ps(_1,_mm_sqrt_ps(_a)));        // 1/(1+sqrt(aa))
          _a = _mm_div_ps(_mm_add_ps(_a,_e),_2);                     // (aa+ee)/2
          _e = _mm_andnot_ps(sm,_mm_sub_ps(_a,_e));                  // |aa-ee|/2
   return _mm_sub_ps(_a,_e);                                         // (aa-ee)/2-|aa-ee|/2
}


static inline float _sse_snc4_ps(__m128* _pe, float* _ne, float* out, int V4) {
// sub net cut (snc)
// _pe - pointer to energy vectors
// _rE - pointer to network energy array
// out - output array
//  V4 - number of iterations (pixels)
   __m128 _Etot = _mm_setzero_ps();             // total energy
   __m128 _Esub = _mm_setzero_ps();             // energy after subnet cut
   __m128* _rE  = (__m128*) ne;                 // m128 pointer to energy array     

         for(j=0; j<V4; j+=4) {                       // loop over selected pixels 
            *_rE = _net_sum_ps(_pe);
                        
            __m128 _cmp = _mm_cmpge_ps(*_rE,_En);     // E>En  
            __m128 _msk = _mm_and_ps(_cmp,_one);      // 0/1 mask

            *_rE = _mm_mul_ps(*_rE,_msk);             // zero sub-threshold pixels 
           _Etot = _mm_add_ps(_Etot,*_rE);

            _cmp = _net_cut_ps(_rE, _pe, _Es, _cmp);  // apply subnetwork cut

            _msk  = _mm_and_ps(_cmp,_one);            // 0/1 mask
            _Esub = _mm_add_ps(_Esub,_mm_mul_ps(*_rE++,_msk));
            *_pE++ = _mm_setzero_ps();
         }

         _mm_storeu_ps(etot,_Etot);
} 
*/
/*
    gx = NET.dotx(Fx,Fx)+1.e-24;          
    gI = NET.dotx(Fp,Fx);                 
    xp = NET.dotx(Fp,am);                 
    xx = NET.dotx(Fx,am);                       
    XP = NET.dotx(Fp,AM);                 
    XX = NET.dotx(Fx,AM);                       

// find weak vector 00

    uc = (xp*gx - xx*gI);                 // u cos of rotation to PCF
    us = (xx*gp - xp*gI);                 // u sin of rotation to PCF
    um = NET.rotx(Fp,uc,Fx,us,u);         // calculate u and return its norm
    et = NET.dotx(am,am);     
    hh = NET.dotx(am,u,e);
    ec = (hh*hh - NET.dotx(e,e))/um;

    if(nn==0) SM.set(n,0.1*hh*hh/NET.dotx(e,e));
    else      SM.add(n,0.1*hh*hh/NET.dotx(e,e));

         NET.rotx(u,hh/um,e,0.,aa);       // normalize aa
//    ec = NET.dotx(aa,aa);

// find weak vector 90

    uc = (XP*gx - XX*gI);                 // u cos of rotation to PCF
    us = (XX*gp - XP*gI);                 // u sin of rotation to PCF
    UM = NET.rotx(Fp,uc,Fx,us,U);        // calculate u and return its norm
    ET = NET.dotx(AM,AM);    
    HH = NET.dotx(AM,U,e);  
    EC = (HH*HH - NET.dotx(e,e))/UM;
         NET.rotx(U,HH/UM,e,0.,AA);        // normalize AA
//    EC = NET.dotx(AA,AA);
//    g2->Fill((ec+EC)/(et+ET));

//  transformation to DDF

    gp = NET.dotx(aa,aa)+1.e-24;          // fp^2
    gx = NET.dotx(AA,AA)+1.e-24;          // fx^2
    gI = NET.dotx(aa,AA);                 // fp*fx
    gR = (gp-gx)/2.; 
    gr = (gp+gx)/2.;
    gc = sqrt(gR*gR+gI*gI);               // norm of complex antenna pattern
     b = (gr-gc)/(gr+gc);    

//    g0->Fill(sqrt(b));

    cc = sqrt((gc+gR)*(gc+gR)+gI*gI);
    ss = NET.rotx(aa,(gc+gR)/cc,AA,gI/cc,s);   // s[k] 
    xx = NET.rotx(am,(gc+gR)/cc,AM,gI/cc,x);   // x[k] 
    cc = sqrt((gc-gR)*(gc-gR)+gI*gI);
    SS = NET.rotx(aa,-(gc-gR)/cc,AA,gI/cc,S)+1.e-24; // S[k]
    XX = NET.rotx(am,-(gc-gR)/cc,AM,gI/cc,X)+1.e-24; // X[k]

//    cout<<ss<<" "<<NET.dotx(x,s)*NET.dotx(x,s)/ss<<endl;

// Principle components

    hh = NET.dotx(x,s,e);
    ec = (hh*hh - NET.dotx(e,e))/ss;
    HH = NET.dotx(X,S,e);
    EC = (HH*HH - NET.dotx(e,e))/SS;

*/



#endif // WATSSE_HH

















