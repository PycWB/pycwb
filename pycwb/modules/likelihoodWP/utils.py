from math import sqrt

import numpy as np
from numba import njit, float32

# static inline __m256 _avx_noise_ps(float** p, float** q, std::vector<float*> &pAVX, int I) {
# // q - pointer to pixel norms 
# // returns noise correction
# // I - number of pixels
#    float k = pAVX[1][I];              // number of detectors
#    __m128 _nx,_ns,_nm,_mk,_gg,_rc,_nn;
#    __m128* _et = (__m128*)pAVX[0];
#    __m128* _MK = (__m128*)pAVX[1];
#    __m128* _ec = (__m128*)pAVX[19];
#    __m128* _gn = (__m128*)pAVX[20];
#    __m128* _rn = (__m128*)pAVX[22];
#    __m128  _GN = _mm_setzero_ps();
#    __m128  _RC = _mm_setzero_ps();
#    __m128  _ES = _mm_setzero_ps();
#    __m128  _EH = _mm_setzero_ps();
#    __m128  _EC = _mm_setzero_ps();
#    __m128  _SC = _mm_setzero_ps();
#    __m128  _NC = _mm_setzero_ps();
#    __m128  _NS = _mm_setzero_ps();

#    static const __m128 _0 = _mm_set1_ps(0);
#    static const __m128 o5 = _mm_set1_ps(0.5);
#    static const __m128 _o = _mm_set1_ps(1.e-9);
#    static const __m128 _1 = _mm_set1_ps(1);
#    static const __m128 _2 = _mm_set1_ps(2);
#    static const __m128 _k = _mm_set1_ps(1./k);

#    NETX(__m128* _s0 = (__m128*)(p[0]+I);, __m128* _s1 = (__m128*)(p[1]+I);,
# 	__m128* _s2 = (__m128*)(p[2]+I);, __m128* _s3 = (__m128*)(p[3]+I);, 
# 	__m128* _s4 = (__m128*)(p[4]+I);, __m128* _s5 = (__m128*)(p[5]+I);,
# 	__m128* _s6 = (__m128*)(p[6]+I);, __m128* _s7 = (__m128*)(p[7]+I);)

#    NETX(__m128* _x0 = (__m128*)(q[0]+I);, __m128* _x1 = (__m128*)(q[1]+I);,
# 	__m128* _x2 = (__m128*)(q[2]+I);, __m128* _x3 = (__m128*)(q[3]+I);, 
# 	__m128* _x4 = (__m128*)(q[4]+I);, __m128* _x5 = (__m128*)(q[5]+I);,
# 	__m128* _x6 = (__m128*)(q[6]+I);, __m128* _x7 = (__m128*)(q[7]+I);)

#    for(int i=0; i<I; i+=4) { 
#       NETX(_ns =*_s0++;,                  _ns = _mm_add_ps(*_s1++,_ns); , 
#            _ns = _mm_add_ps(*_s2++,_ns);, _ns = _mm_add_ps(*_s3++,_ns); , 
#            _ns = _mm_add_ps(*_s4++,_ns);, _ns = _mm_add_ps(*_s5++,_ns); , 
#            _ns = _mm_add_ps(*_s6++,_ns);, _ns = _mm_add_ps(*_s7++,_ns); )
	 
#       NETX(_nx =*_x0++;,                  _nx = _mm_add_ps(*_x1++,_nx); , 
#            _nx = _mm_add_ps(*_x2++,_nx);, _nx = _mm_add_ps(*_x3++,_nx); , 
#            _nx = _mm_add_ps(*_x4++,_nx);, _nx = _mm_add_ps(*_x5++,_nx); , 
#            _nx = _mm_add_ps(*_x6++,_nx);, _nx = _mm_add_ps(*_x7++,_nx); )
	 	 
#       _ns = _mm_mul_ps(_ns,_k);
#       _nx = _mm_mul_ps(_nx,_k);
#       _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK,_0),_1);        // event mask
#       _nm = _mm_and_ps(_mm_cmpgt_ps(_nx,_0),_1);         // data norm mask
#       _nm = _mm_mul_ps(_mk,_nm);                         // norm x event mask      
#       _EC = _mm_add_ps(_EC,_mm_mul_ps(_nm,*_ec));        // coherent energy
#       _NC = _mm_add_ps(_NC,_nm);                         // number of core pixels

#       _nm = _mm_sub_ps(_mk,_nm);                         // halo mask
#       _ES = _mm_add_ps(_ES,_mm_mul_ps(_nm,*_rn));        // residual sattelite noise

#       _rc = _mm_and_ps(_mm_cmplt_ps(*_gn,_2),_1);        // rc=1 if gn<2, or rc=0 if gn>=2
#       _nn = _mm_mul_ps(*_gn,_mm_sub_ps(_1,_rc));         // _nn = [_1 - _rc] * _gn
#       _rc = _mm_add_ps(_rc,_mm_mul_ps(_nn,o5));          // Ec normalization
#       _rc = _mm_div_ps(*_ec,_mm_add_ps(_rc,_o));         // normalized EC

#       _nm = _mm_and_ps(_mm_cmpgt_ps(_ns,_0),_1);         // signal norm mask
#       _nm = _mm_mul_ps(_mk,_nm);                         // norm x event mask      
#      *_gn = _mm_mul_ps(_mk,_mm_mul_ps(*_gn,_nx));        // normalize Gaussian noise ecorrection
#       _SC = _mm_add_ps(_SC,_mm_mul_ps(_nm,*_ec));        // signal coherent energy
#       _RC = _mm_add_ps(_RC,_mm_mul_ps(_nm, _rc));        // total normalized EC
#       _GN = _mm_add_ps(_GN,_mm_mul_ps(_nm,*_gn));        // G-noise correction in time domain
#       _NS = _mm_add_ps(_NS,_nm);                         // number of signal pixels

#       _nm = _mm_sub_ps(_mk,_nm);                         // satellite mask
#       _EH = _mm_add_ps(_EH,_mm_mul_ps(_nm,*_et));        // halo energy in TF domain

#       _MK++; _gn++; _et++; _ec++; _rn++;
#    }
#    float es =  _wat_hsum(_ES)/2;                         // residual satellite energy in time domain 
#    float eh =  _wat_hsum(_EH)/2;                         // halo energy in TF domain
#    float gn =  _wat_hsum(_GN);                           // G-noise correction
#    float nc =  _wat_hsum(_NC);                           // number of core pixels
#    float ns =  _wat_hsum(_NS);                           // number of signal pixels
#    float rc =  _wat_hsum(_RC);                           // normalized EC x 2 
#    float ec =  _wat_hsum(_EC);                           // signal coherent energy x 2
#    float sc =  _wat_hsum(_SC);                           // core coherent energy x 2
#    return _mm256_set_ps(ns,nc,es,eh,rc/(sc+0.01),sc-ec,ec,gn);
# } 
def avx_noise_ps(p, q, et, MK, ec, gn, rn):
    """
    get G-noise correction

    Parameters
    ----------
    p : np.ndarray
        The p component of the signal.
    q : np.ndarray
        pointer to pixel norms 
    et : np.ndarray
        The total energy
    MK : np.ndarray
        The mask indicating valid pixels.
    ec : np.ndarray
        The coherent energy.
    gn : np.ndarray
        The G-noise component.
    rn : np.ndarray
        The residual noise.

    Returns
    -------
    tuple
        - NS : float
    """
    n_pixels = len(p[0])
    n_ifos = len(p)

    # TODO: Simplify the computation
    EC = 0
    NC = 0
    ES = 0
    SC = 0
    RC = 0
    GN = 0
    EH = 0
    NS = 0

    for i in range(n_pixels):
        ns = 0
        nx = 0
            
        for j in range(n_ifos):
            ns += p[j][i]
            nx += q[j][i]
        ns /= n_ifos
        nx /= n_ifos
        # # DEBUG:
        # if i in [392, 393, 394, 395]:
        #     print(f"Pixel {i}: ns = {ns}, nx = {nx}")
        # if i == 393:
        #     ns = -ns
        #     nx = -nx
    
        mk = 1.0 if MK[i] > 0 else 0.0  # event mask
        nm = 1.0 if nx > 0 else 0.0  # data norm mask
        nm = mk * nm  # norm x event mask

        EC += nm * ec[i]  # coherent energy
        NC += nm

        nm = mk - nm                    # halo mask
        ES += nm * rn[i]                # residual sattelite noise
        rc = 1.0 if gn[i] < 2 else 0.0  # rc=1 if gn<2, or rc=0 if gn>=2
        nn = gn[i] * (1 - rc)           # _nn = [_1 - _rc] * _gn
        rc += nn * 0.5                  # Ec normalization
        rc = ec[i] / (rc + 1e-9)        # normalized EC

        nm = 1.0 if ns > 0 else 0.0  # signal norm mask
        nm = mk * nm  # norm x event mask
        gn[i] = mk * gn[i] * nx  # normalize Gaussian noise ecorrection
        SC += nm * ec[i]  # signal coherent energy
        RC += nm * rc  # total normalized EC
        GN += nm * gn[i]  # G-noise correction in time domain
        NS += nm # number of signal pixels

        nm = mk - nm  # satellite mask
        # # DEBUG:
        # if nm > 0:
        #     print(f"Warning: nm > 0 for pixel {i}, nm = {nm}, et = {et[i]}")
        #     print(f"ns = {ns}, nx = {nx}, mk = {mk}, ec = {ec[i]}, gn = {gn[i]}")
        EH += nm * et[i]  # halo energy in TF domain

    es = ES / 2  # residual satellite energy in time domain
    eh = EH / 2  # halo energy in TF domain
    gn = GN  # G-noise correction
    nc = NC  # number of core pixels
    ns = NS  # number of signal pixels
    rc = RC  # normalized EC x 2
    ec = EC  # signal coherent energy x 2
    sc = SC  # core coherent energy x 2

    return gn, ec, sc - ec, rc / (sc + 0.01), eh, es, nc, ns


# static inline float _avx_setAMP_ps(float** p, float** q, 
# 				   std::vector<float*> &pAVX, int I) {
# // set packet amplitudes for waveform reconstruction
# // returns number of degrees of freedom - effective # of pixels per detector   
#    int II = I*2;
#    int I2 = II+2;
#    int I3 = II+3;
#    int I4 = II+4;
#    int I5 = II+5;
#    float k = pAVX[1][I];              // number of detectors

#    __m128 _a, _A, _n, _mk, _nn;
#    static const __m128 _o = _mm_set1_ps(1.e-9);
#    static const __m128 _0 = _mm_set1_ps(0);
#    static const __m128 _1 = _mm_set1_ps(1);
#    static const __m128 _4 = _mm_set1_ps(4);
#    static const __m128 o5 = _mm_set1_ps(0.5);

#    NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0]; __m128* _n0 = (__m128*)(q[0]+I); , 
# 	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1]; __m128* _n1 = (__m128*)(q[1]+I); , 
# 	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2]; __m128* _n2 = (__m128*)(q[2]+I); , 
# 	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3]; __m128* _n3 = (__m128*)(q[3]+I); , 
# 	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4]; __m128* _n4 = (__m128*)(q[4]+I); , 
# 	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5]; __m128* _n5 = (__m128*)(q[5]+I); , 
# 	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6]; __m128* _n6 = (__m128*)(q[6]+I); , 
# 	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7]; __m128* _n7 = (__m128*)(q[7]+I); ) 

#    NETX(__m128 a0=_mm_set1_ps(q[0][I4]); __m128 s0=_mm_set1_ps(q[0][I2]); __m128 c0=_mm_set1_ps(q[0][I3]); ,
# 	__m128 a1=_mm_set1_ps(q[1][I4]); __m128 s1=_mm_set1_ps(q[1][I2]); __m128 c1=_mm_set1_ps(q[1][I3]); ,
# 	__m128 a2=_mm_set1_ps(q[2][I4]); __m128 s2=_mm_set1_ps(q[2][I2]); __m128 c2=_mm_set1_ps(q[2][I3]); ,
# 	__m128 a3=_mm_set1_ps(q[3][I4]); __m128 s3=_mm_set1_ps(q[3][I2]); __m128 c3=_mm_set1_ps(q[3][I3]); ,
# 	__m128 a4=_mm_set1_ps(q[4][I4]); __m128 s4=_mm_set1_ps(q[4][I2]); __m128 c4=_mm_set1_ps(q[4][I3]); ,
# 	__m128 a5=_mm_set1_ps(q[5][I4]); __m128 s5=_mm_set1_ps(q[5][I2]); __m128 c5=_mm_set1_ps(q[5][I3]); ,
# 	__m128 a6=_mm_set1_ps(q[6][I4]); __m128 s6=_mm_set1_ps(q[6][I2]); __m128 c6=_mm_set1_ps(q[6][I3]); ,
# 	__m128 a7=_mm_set1_ps(q[7][I4]); __m128 s7=_mm_set1_ps(q[7][I2]); __m128 c7=_mm_set1_ps(q[7][I3]); )

#    __m128* _MK = (__m128*)pAVX[1];
#    __m128* _fp = (__m128*)pAVX[2];       
#    __m128* _fx = (__m128*)pAVX[3];
#    __m128* _ee = (__m128*)pAVX[15];
#    __m128* _EE = (__m128*)pAVX[16];
#    __m128* _gn = (__m128*)pAVX[20];

#    __m128  _Np = _mm_setzero_ps();        // number of effective pixels per detector
     
#    for(int i=0; i<I; i+=4) {  //  packet amplitudes 
#       _mk = _mm_mul_ps(o5,_mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1));                  // event mask
#       NETX(
# 	   _n = _mm_mul_ps(_mm_mul_ps(a0,_mk),*_n0); _a=*_p0; _A=*_q0; _nn=*_n0++;
# 	   *_p0++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c0),_mm_mul_ps(_A,s0))); 
# 	   *_q0++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c0),_mm_mul_ps(_a,s0))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a1,_mk),*_n1); _a=*_p1; _A=*_q1; _nn=_mm_add_ps(_nn,*_n1++);
# 	   *_p1++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c1),_mm_mul_ps(_A,s1))); 
# 	   *_q1++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c1),_mm_mul_ps(_a,s1))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a2,_mk),*_n2); _a=*_p2; _A=*_q2; _nn=_mm_add_ps(_nn,*_n2++);
# 	   *_p2++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c2),_mm_mul_ps(_A,s2))); 
# 	   *_q2++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c2),_mm_mul_ps(_a,s2))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a3,_mk),*_n3); _a=*_p3; _A=*_q3; _nn=_mm_add_ps(_nn,*_n3++);
# 	   *_p3++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c3),_mm_mul_ps(_A,s3))); 
# 	   *_q3++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c3),_mm_mul_ps(_a,s3))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a4,_mk),*_n4); _a=*_p4; _A=*_q4; _nn=_mm_add_ps(_nn,*_n4++);
# 	   *_p4++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c4),_mm_mul_ps(_A,s4))); 
# 	   *_q4++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c4),_mm_mul_ps(_a,s4))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a5,_mk),*_n5); _a=*_p5; _A=*_q5; _nn=_mm_add_ps(_nn,*_n5++);
# 	   *_p5++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c5),_mm_mul_ps(_A,s5))); 
# 	   *_q5++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c5),_mm_mul_ps(_a,s5))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a6,_mk),*_n6); _a=*_p6; _A=*_q6; _nn=_mm_add_ps(_nn,*_n6++);
# 	   *_p6++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c6),_mm_mul_ps(_A,s6))); 
# 	   *_q6++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c6),_mm_mul_ps(_a,s6))); ,
# 	   _n = _mm_mul_ps(_mm_mul_ps(a7,_mk),*_n7); _a=*_p7; _A=*_q7; _nn=_mm_add_ps(_nn,*_n7++);
# 	   *_p7++ = _mm_mul_ps(_n,_mm_sub_ps(_mm_mul_ps(_a,c7),_mm_mul_ps(_A,s7))); 
# 	   *_q7++ = _mm_mul_ps(_n,_mm_add_ps(_mm_mul_ps(_A,c7),_mm_mul_ps(_a,s7))); )

#       _nn = _mm_mul_ps(_nn,_mk);
#       _Np = _mm_add_ps(_Np,_nn);                         // Dof * k/4
#    }
#    return _wat_hsum(_Np)*4/k;
# } 
def avx_setAMP_ps(p, q, q_norm, q_si, q_co, q_a, q_A, MK):
    """
    set packet amplitudes for waveform reconstruction
    returns number of degrees of freedom - effective # of pixels per detector

    Parameters
    ----------
    p : np.ndarray
        The p component of the signal.
    q : np.ndarray
        The q component of the signal.
    q_norm : np.ndarray
        Pointer to pixel norms.
    q_si : np.ndarray
        The second component of the q signal. q_II2 : np.ndarray
    q_co : np.ndarray
        The third component of the q signal. q_II3 : np.ndarray
    q_E : np.ndarray
        The fourth component of the q signal. q_II4 : np.ndarray
    MK : np.ndarray
        The mask indicating valid pixels.

    Returns
    -------
    tuple
        - N : float
            Number of effective pixels per detector.
        - new_p : np.ndarray
            Updated p component.
        - new_q : np.ndarray
            Updated q component.
    """
    new_p = np.zeros_like(p)
    new_q = np.zeros_like(q)

    n_pix = len(p[0])
    n_ifo = len(p)

    aA = q_a + q_A
    
    _Np = 0.0  # number of effective pixels per detector

    for i in range(n_pix):
        mk = 0.5 * (1.0 if MK[i] > 0 else 0.0)  # event mask
        _nn = 0.0  # norm x event mask
        for j in range(n_ifo):
            _nn += q_norm[j][i]
            # here the q_E is different from the q_II4
            _n = aA[j] * mk * q_norm[j][i]  # norm x event mask
            new_p[j][i] = _n * (p[j][i] * q_co[j] - q[j][i] * q_si[j])
            new_q[j][i] = _n * (q[j][i] * q_co[j] + p[j][i] * q_si[j])
        
        _nn *= mk  # norm x event mask
        _Np += _nn  # accumulate effective pixels
    return _Np * 4 / n_ifo, new_p, new_q


# static inline void _avx_loadNULL_ps(float** n, float** N, 
# 				    float** d, float** D,
# 				    float** h, float** H, int I) { 
# // load NULL packet amplitudes for all detectors and pixels
# // these amplitudes are used for reconstruction of data time searies
# // now works only for <4 detector
#    NETX(__m128* _n0 = (__m128*)n[0]; __m128* _N0 = (__m128*)N[0]; , 
# 	__m128* _n1 = (__m128*)n[1]; __m128* _N1 = (__m128*)N[1]; , 
# 	__m128* _n2 = (__m128*)n[2]; __m128* _N2 = (__m128*)N[2]; , 
# 	__m128* _n3 = (__m128*)n[3]; __m128* _N3 = (__m128*)N[3]; , 
# 	__m128* _n4 = (__m128*)n[4]; __m128* _N4 = (__m128*)N[4]; , 
# 	__m128* _n5 = (__m128*)n[5]; __m128* _N5 = (__m128*)N[5]; , 
# 	__m128* _n6 = (__m128*)n[6]; __m128* _N6 = (__m128*)N[6]; , 
# 	__m128* _n7 = (__m128*)n[7]; __m128* _N7 = (__m128*)N[7]; ) 

#    NETX(__m128* _d0 = (__m128*)d[0]; __m128* _D0 = (__m128*)D[0]; , 
# 	__m128* _d1 = (__m128*)d[1]; __m128* _D1 = (__m128*)D[1]; , 
# 	__m128* _d2 = (__m128*)d[2]; __m128* _D2 = (__m128*)D[2]; , 
# 	__m128* _d3 = (__m128*)d[3]; __m128* _D3 = (__m128*)D[3]; , 
# 	__m128* _d4 = (__m128*)d[4]; __m128* _D4 = (__m128*)D[4]; , 
# 	__m128* _d5 = (__m128*)d[5]; __m128* _D5 = (__m128*)D[5]; , 
# 	__m128* _d6 = (__m128*)d[6]; __m128* _D6 = (__m128*)D[6]; , 
# 	__m128* _d7 = (__m128*)d[7]; __m128* _D7 = (__m128*)D[7]; ) 
 
#    NETX(__m128* _h0 = (__m128*)h[0]; __m128* _H0 = (__m128*)H[0]; , 
# 	__m128* _h1 = (__m128*)h[1]; __m128* _H1 = (__m128*)H[1]; , 
# 	__m128* _h2 = (__m128*)h[2]; __m128* _H2 = (__m128*)H[2]; , 
# 	__m128* _h3 = (__m128*)h[3]; __m128* _H3 = (__m128*)H[3]; , 
# 	__m128* _h4 = (__m128*)h[4]; __m128* _H4 = (__m128*)H[4]; , 
# 	__m128* _h5 = (__m128*)h[5]; __m128* _H5 = (__m128*)H[5]; , 
# 	__m128* _h6 = (__m128*)h[6]; __m128* _H6 = (__m128*)H[6]; , 
# 	__m128* _h7 = (__m128*)h[7]; __m128* _H7 = (__m128*)H[7]; ) 

#    for(int i=0; i<I; i+=4) {
#       NETX(*_n0++ = _mm_sub_ps(*_d0++,*_h0++); *_N0++ = _mm_sub_ps(*_D0++,*_H0++); ,
#            *_n1++ = _mm_sub_ps(*_d1++,*_h1++); *_N1++ = _mm_sub_ps(*_D1++,*_H1++); ,
#            *_n2++ = _mm_sub_ps(*_d2++,*_h2++); *_N2++ = _mm_sub_ps(*_D2++,*_H2++); ,
#            *_n3++ = _mm_sub_ps(*_d3++,*_h3++); *_N3++ = _mm_sub_ps(*_D3++,*_H3++); ,
#            *_n4++ = _mm_sub_ps(*_d4++,*_h4++); *_N4++ = _mm_sub_ps(*_D4++,*_H4++); ,
#            *_n5++ = _mm_sub_ps(*_d5++,*_h5++); *_N5++ = _mm_sub_ps(*_D5++,*_H5++); ,
#            *_n6++ = _mm_sub_ps(*_d6++,*_h6++); *_N6++ = _mm_sub_ps(*_D6++,*_H6++); ,
#            *_n7++ = _mm_sub_ps(*_d7++,*_h7++); *_N7++ = _mm_sub_ps(*_D7++,*_H7++); )
#    }
#    return;
# } 
def avx_loadNULL_ps(d, D, h, H):
    """
    Load NULL packet amplitudes for all detectors and pixels.
    These amplitudes are used for reconstruction of data time series.

    Parameters
    ----------
    n : np.ndarray
        The n component of the signal.
    N : np.ndarray
        The N component of the signal.
    d : np.ndarray
        The d component of the signal.
    D : np.ndarray
        The D component of the signal.
    h : np.ndarray
        The h component of the signal.
    H : np.ndarray
        The H component of the signal.

    Returns
    -------
    None
    """
    n = np.array([d[i].copy() for i in range(len(d))], dtype=np.float32)
    N = np.array([D[i].copy() for i in range(len(D))], dtype=np.float32)

    n_pix = len(d[0])

    for i in range(n_pix):
        for j in range(len(n)):
            n[j][i] = d[j][i] - h[j][i]
            N[j][i] = D[j][i] - H[j][i]

    return n, N


# static inline void _avx_pol_ps(float** p, float ** q, 
#                                wavearray<double>* pol00, wavearray<double>* pol90,
# 		               std::vector<float*> &pAPN,
# 			       std::vector<float*> &pAVX, int II) {
# // calculates the polar coordinates of the input vector v in the DPF frame 
# // p,q  - input/output - data vector 
# // pol00 - output - 00 component in polar coordinates (pol00[0] : radius, pol00[1] : angle in radians)
# // pol90 - output - 90 component in polar coordinates (pol90[0] : radius, pol90[1] : angle in radians)
# // pRMS - vector with noise rms data
# // pAVX - pixel statistics
# // II   - number of AVX pixels
# // in likelihoodWP these arrays should be stored exactly in the same order.

#    int I = abs(II);
   
#    __m128* _MK = (__m128*)pAVX[1];
#    __m128* _fp = (__m128*)pAVX[2];       
#    __m128* _fx = (__m128*)pAVX[3];

#    __m128 _xp,_XP,_xx,_XX,_rr,_RR,_mk;

#    static const __m128 _0  = _mm_set1_ps(0);
#    static const __m128 _1  = _mm_set1_ps(1);
#    static const __m128 _o  = _mm_set1_ps(1.e-5);

#    __m128 _ss,_cc,_SS,_CC;
#    float rpol[4],cpol[4],spol[4];
#    float RPOL[4],CPOL[4],SPOL[4];

#    double* r = pol00[0].data;
#    double* a = pol00[1].data;
#    double* R = pol90[0].data;
#    double* A = pol90[1].data;

#    // pointers to antenna patterns
#    NETX(__m128* _f0=(__m128*)pAPN[0]; __m128* _F0=(__m128*)(pAPN[0]+I);,
# 	__m128* _f1=(__m128*)pAPN[1]; __m128* _F1=(__m128*)(pAPN[1]+I);,
# 	__m128* _f2=(__m128*)pAPN[2]; __m128* _F2=(__m128*)(pAPN[2]+I);,
# 	__m128* _f3=(__m128*)pAPN[3]; __m128* _F3=(__m128*)(pAPN[3]+I);,
# 	__m128* _f4=(__m128*)pAPN[4]; __m128* _F4=(__m128*)(pAPN[4]+I);,
# 	__m128* _f5=(__m128*)pAPN[5]; __m128* _F5=(__m128*)(pAPN[5]+I);,
# 	__m128* _f6=(__m128*)pAPN[6]; __m128* _F6=(__m128*)(pAPN[6]+I);,
# 	__m128* _f7=(__m128*)pAPN[7]; __m128* _F7=(__m128*)(pAPN[7]+I);)

#    // pointers to data
#    NETX(__m128* _p0 = (__m128*)p[0]; __m128* _q0 = (__m128*)q[0];, 
# 	__m128* _p1 = (__m128*)p[1]; __m128* _q1 = (__m128*)q[1];, 
# 	__m128* _p2 = (__m128*)p[2]; __m128* _q2 = (__m128*)q[2];, 
# 	__m128* _p3 = (__m128*)p[3]; __m128* _q3 = (__m128*)q[3];, 
# 	__m128* _p4 = (__m128*)p[4]; __m128* _q4 = (__m128*)q[4];, 
# 	__m128* _p5 = (__m128*)p[5]; __m128* _q5 = (__m128*)q[5];, 
# 	__m128* _p6 = (__m128*)p[6]; __m128* _q6 = (__m128*)q[6];, 
# 	__m128* _p7 = (__m128*)p[7]; __m128* _q7 = (__m128*)q[7];) 

#    int m=0;
#    for(int i=0; i<I; i+=4) {                                 

# // Compute scalar products 

#       _mk = _mm_and_ps(_mm_cmpgt_ps(*_MK++,_0),_1); // event mask - apply energy threshold En

#       NETX(                                                 
# 	   _xp = _mm_mul_ps(*_f0,_mm_mul_ps(_mk,*_p0));                      // (x,f+)
# 	   _XP = _mm_mul_ps(*_f0,_mm_mul_ps(_mk,*_q0));                      // (X,f+)
# 	   _xx = _mm_mul_ps(*_F0,_mm_mul_ps(_mk,*_p0));                      // (x,fx)
# 	   _XX = _mm_mul_ps(*_F0,_mm_mul_ps(_mk,*_q0));                  ,   // (X,fx)
	   			                            
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f1,_mm_mul_ps(_mk,*_p1)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f1,_mm_mul_ps(_mk,*_q1)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F1,_mm_mul_ps(_mk,*_p1)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F1,_mm_mul_ps(_mk,*_q1)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f2,_mm_mul_ps(_mk,*_p2)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f2,_mm_mul_ps(_mk,*_q2)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F2,_mm_mul_ps(_mk,*_p2)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F2,_mm_mul_ps(_mk,*_q2)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f3,_mm_mul_ps(_mk,*_p3)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f3,_mm_mul_ps(_mk,*_q3)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F3,_mm_mul_ps(_mk,*_p3)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F3,_mm_mul_ps(_mk,*_q3)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f4,_mm_mul_ps(_mk,*_p4)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f4,_mm_mul_ps(_mk,*_q4)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F4,_mm_mul_ps(_mk,*_p4)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F4,_mm_mul_ps(_mk,*_q4)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f5,_mm_mul_ps(_mk,*_p5)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f5,_mm_mul_ps(_mk,*_q5)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F5,_mm_mul_ps(_mk,*_p5)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F5,_mm_mul_ps(_mk,*_q5)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f6,_mm_mul_ps(_mk,*_p6)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f6,_mm_mul_ps(_mk,*_q6)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F6,_mm_mul_ps(_mk,*_p6)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F6,_mm_mul_ps(_mk,*_q6)));  ,   // (X,fx)
	   		                          
# 	   _xp = _mm_add_ps(_xp,_mm_mul_ps(*_f7,_mm_mul_ps(_mk,*_p7)));      // (x,f+)
# 	   _XP = _mm_add_ps(_XP,_mm_mul_ps(*_f7,_mm_mul_ps(_mk,*_q7)));      // (X,f+)
# 	   _xx = _mm_add_ps(_xx,_mm_mul_ps(*_F7,_mm_mul_ps(_mk,*_p7)));      // (x,fx)
# 	   _XX = _mm_add_ps(_XX,_mm_mul_ps(*_F7,_mm_mul_ps(_mk,*_q7)));  )   // (X,fx)
	 
# // 00/90 components in polar coordinates (pol00/90[0] : radius, pol00/90[1] : angle in radians)

#       _cc = _mm_div_ps(_xp,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));     // (x,f+) / {|f+|+epsilon}
#       _ss = _mm_div_ps(_xx,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));     // (x,fx) / {|fx|+epsilon}

#       _rr = _mm_div_ps(_mm_mul_ps(_xp,_xp),_mm_add_ps(*_fp,_o));
#       _rr = _mm_add_ps(_rr,_mm_div_ps(_mm_mul_ps(_xx,_xx),_mm_add_ps(*_fx,_o)));

#       _mm_storeu_ps(cpol,_cc);					   // cos
#       _mm_storeu_ps(spol,_ss);					   // sin
#       _mm_storeu_ps(rpol,_rr);                   		   // (x,x);        

#       _CC = _mm_div_ps(_XP,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (X,f+) / {|f+|+epsilon}
#       _SS = _mm_div_ps(_XX,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (X,fx) / {|fx|+epsilon}

#       _RR = _mm_div_ps(_mm_mul_ps(_XP,_XP),_mm_add_ps(*_fp,_o));
#       _RR = _mm_add_ps(_RR,_mm_div_ps(_mm_mul_ps(_XX,_XX),_mm_add_ps(*_fx,_o)));

#       _mm_storeu_ps(CPOL,_CC);					   // cos
#       _mm_storeu_ps(SPOL,_SS);					   // sin
#       _mm_storeu_ps(RPOL,_RR);                   		   // (X,X);        

#       for(int n=0;n<4;n++) {
#         r[m] = sqrt(rpol[n]);                        		   // |x|
#         a[m] = atan2(spol[n],cpol[n]);               		   // atan2(spol,cpol)
#         R[m] = sqrt(RPOL[n]);                        		   // |X|
#         A[m] = atan2(SPOL[n],CPOL[n]);               		   // atan2(SPOL,CPOL)
#         m++;
#       }

# // PnP - Projection to network Plane

#       _cc = _mm_div_ps(_cc,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (x,f+) / {|f+|^2+epsilon}
#       _ss = _mm_div_ps(_ss,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (x,fx) / {|fx|^2+epsilon}

#       _CC = _mm_div_ps(_CC,_mm_add_ps(_mm_sqrt_ps(*_fp),_o));      // (X,f+) / {|f+|^2+epsilon}
#       _SS = _mm_div_ps(_SS,_mm_add_ps(_mm_sqrt_ps(*_fx),_o));      // (X,fx) / {|fx|^2+epsilon}

#       NETX(*_p0 = _mm_add_ps(_mm_mul_ps(*_f0,_cc),_mm_mul_ps(*_F0,_ss));
# 	   *_q0 = _mm_add_ps(_mm_mul_ps(*_f0,_CC),_mm_mul_ps(*_F0,_SS)); ,
#            *_p1 = _mm_add_ps(_mm_mul_ps(*_f1,_cc),_mm_mul_ps(*_F1,_ss));
# 	   *_q1 = _mm_add_ps(_mm_mul_ps(*_f1,_CC),_mm_mul_ps(*_F1,_SS)); ,
#            *_p2 = _mm_add_ps(_mm_mul_ps(*_f2,_cc),_mm_mul_ps(*_F2,_ss));
# 	   *_q2 = _mm_add_ps(_mm_mul_ps(*_f2,_CC),_mm_mul_ps(*_F2,_SS)); ,
#            *_p3 = _mm_add_ps(_mm_mul_ps(*_f3,_cc),_mm_mul_ps(*_F3,_ss));
# 	   *_q3 = _mm_add_ps(_mm_mul_ps(*_f3,_CC),_mm_mul_ps(*_F3,_SS)); ,
#            *_p4 = _mm_add_ps(_mm_mul_ps(*_f4,_cc),_mm_mul_ps(*_F4,_ss));
# 	   *_q4 = _mm_add_ps(_mm_mul_ps(*_f4,_CC),_mm_mul_ps(*_F4,_SS)); ,
#            *_p5 = _mm_add_ps(_mm_mul_ps(*_f5,_cc),_mm_mul_ps(*_F5,_ss));
# 	   *_q5 = _mm_add_ps(_mm_mul_ps(*_f5,_CC),_mm_mul_ps(*_F5,_SS)); ,
#            *_p6 = _mm_add_ps(_mm_mul_ps(*_f6,_cc),_mm_mul_ps(*_F6,_ss));
# 	   *_q6 = _mm_add_ps(_mm_mul_ps(*_f6,_CC),_mm_mul_ps(*_F6,_SS)); ,
#            *_p7 = _mm_add_ps(_mm_mul_ps(*_f7,_cc),_mm_mul_ps(*_F7,_ss));
# 	   *_q7 = _mm_add_ps(_mm_mul_ps(*_f7,_CC),_mm_mul_ps(*_F7,_SS)); )

# // DSP - Dual Stream Phase Transform

#       __m128 _N = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(_cc,_cc),_mm_mul_ps(_CC,_CC)));
      
#       _cc = _mm_div_ps(_cc,_mm_add_ps(_N,_o)); 			  // cos_dsp = N * (x,f+)/|f+|^2
#       _CC = _mm_div_ps(_CC,_mm_add_ps(_N,_o)); 			  // sin_dsp = N * (X,f+)/|f+|^2

#       __m128 _y,_Y;

#       NETX(_y = _mm_add_ps(_mm_mul_ps(*_p0,_cc),_mm_mul_ps(*_q0,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q0,_cc),_mm_mul_ps(*_p0,_CC)); *_p0 = _y; *_q0 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p1,_cc),_mm_mul_ps(*_q1,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q1,_cc),_mm_mul_ps(*_p1,_CC)); *_p1 = _y; *_q1 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p2,_cc),_mm_mul_ps(*_q2,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q2,_cc),_mm_mul_ps(*_p2,_CC)); *_p2 = _y; *_q2 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p3,_cc),_mm_mul_ps(*_q3,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q3,_cc),_mm_mul_ps(*_p3,_CC)); *_p3 = _y; *_q3 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p4,_cc),_mm_mul_ps(*_q4,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q4,_cc),_mm_mul_ps(*_p4,_CC)); *_p4 = _y; *_q4 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p5,_cc),_mm_mul_ps(*_q5,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q5,_cc),_mm_mul_ps(*_p5,_CC)); *_p5 = _y; *_q5 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p6,_cc),_mm_mul_ps(*_q6,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q6,_cc),_mm_mul_ps(*_p6,_CC)); *_p6 = _y; *_q6 = _Y; ,
#            _y = _mm_add_ps(_mm_mul_ps(*_p7,_cc),_mm_mul_ps(*_q7,_CC));
# 	   _Y = _mm_sub_ps(_mm_mul_ps(*_q7,_cc),_mm_mul_ps(*_p7,_CC)); *_p7 = _y; *_q7 = _Y; )


# // Increment pointers

#       NETX(                                                 
#            _p0++;_q0++;_f0++;_F0++;	,
#            _p1++;_q1++;_f1++;_F1++;	,
#            _p2++;_q2++;_f2++;_F2++;	,
#            _p3++;_q3++;_f3++;_F3++;	,
#            _p4++;_q4++;_f4++;_F4++;	,
#            _p5++;_q5++;_f5++;_F5++;	,
#            _p6++;_q6++;_f6++;_F6++;	,
#            _p7++;_q7++;_f7++;_F7++;	)

#       _fp++;_fx++;
#    }

#    return; 
# } 
def avx_pol_ps(p, q, MK, fp, fx, f, F):
    """
    Calculates the polar coordinates of the input vector v in the DPF frame.

    Parameters
    ----------
    p : np.ndarray
        The p component of the signal.
    q : np.ndarray
        The q component of the signal.
    pol00 : list
        Output for 00 component in polar coordinates (pol00[0] : radius, pol00[1] : angle in radians).
    pol90 : list
        Output for 90 component in polar coordinates (pol90[0] : radius, pol90[1] : angle in radians).
    MK : np.ndarray
        Event mask array.
    fp : np.ndarray
        The FP component.
    fx : np.ndarray
        The FX component.
    f : np.ndarray
        Array of antenna patterns for each detector.
    F : np.ndarray
        Array of antenna patterns for each detector.

    Returns
    -------
    None
    """
    n_ifo = len(p)
    n_pix = len(p[0])

    new_p = np.empty((n_ifo, n_pix), dtype=np.float32)
    new_q = np.empty((n_ifo, n_pix), dtype=np.float32)
    
    r = np.empty(n_pix, dtype=np.float32)
    a = np.empty(n_pix, dtype=np.float32)
    R = np.empty(n_pix, dtype=np.float32)
    A = np.empty(n_pix, dtype=np.float32)

    _o = float(1.e-5)

    for i in range(n_pix):
        mk = float(1.0) if MK[i] > 0 else float(0.0)

        xp = 0.
        XP = 0.
        xx = 0.
        XX = 0.

        for j in range(n_ifo):
            xp += f[i][j] * (mk * p[j][i])
            XP += f[i][j] * (mk * q[j][i])
            xx += F[i][j] * (mk * p[j][i])
            XX += F[i][j] * (mk * q[j][i])

        # 00/90 components in polar coordinates (pol00/90[0] : radius, pol00/90[1] : angle in radians)
        cpol = xp / (np.sqrt(fp[i]) + _o)
        spol = xx / (np.sqrt(fx[i]) + _o)
        rpol = (xp * xp) / (fp[i] + _o) + (xx * xx) / (fx[i] + _o)

        CPOL = XP / (np.sqrt(fp[i]) + _o)
        SPOL = XX / (np.sqrt(fx[i]) + _o)
        RPOL = (XP * XP) / (fp[i] + _o) + (XX * XX) / (fx[i] + _o)

        r[i] = np.sqrt(rpol)
        a[i] = np.arctan2(spol, cpol)
        R[i] = np.sqrt(RPOL)
        A[i] = np.arctan2(SPOL, CPOL)

    # PnP - Projection to network Plane
    cpol /= (np.sqrt(fp[i]) + _o)   # (x,f+) / {|f+|^2+epsilon}
    spol /= (np.sqrt(fx[i]) + _o)  # (x,fx) / {|fx|^2+epsilon}
    CPOL /= (np.sqrt(fp[i]) + _o)  # (X,f+) / {|f+|^2+epsilon}
    SPOL /= (np.sqrt(fx[i]) + _o)  # (X,fx) / {|fx|^2+epsilon}  

    for j in range(n_ifo):
        new_p[j][i] = (f[i][j] * cpol + F[i][j] * spol)
        new_q[j][i] = (f[i][j] * CPOL + F[i][j] * SPOL)

    # DSP - Dual Stream Phase Transform
    N = np.sqrt(cpol * cpol + CPOL * CPOL)
    cpol /= (N + _o)  # cos_dsp = N * (x,f+)/|f+|^2
    CPOL /= (N + _o)  # sin_dsp = N * (X,f+)/|f+|^2

    for j in range(n_ifo):
        new_p[j][i] = new_p[j][i] * cpol + new_q[j][i] * CPOL
        new_q[j][i] = new_q[j][i] * cpol - new_p[j][i] * CPOL

    return new_p, new_q, (r, a), (R, A)


@njit(cache=True)
def avx_packet_ps(v00, v90, mask):
    """
    calculates packet rotation sin/cos, amplitudes and unit vectors, initialize unit vector arrays

    Parameters
    ----------
    v00 : np.ndarray
        The v00 component of the packet. v00[ifo][pixel]
    v90 : np.ndarray
        The v90 component of the packet. v90[ifo][pixel]
    mask : np.ndarray
        The mask indicating valid pixels. mask[pixel]

    Returns
    -------
    tuple
        - Ep : float
            The packet energy p[er quadrature
        - v00_updated : np.ndarray
            Updated v00 component of the packet after normalization and rotation.
        - v90_updated : np.ndarray
            Updated v90 component of the packet after normalization and rotation.
        - E : np.ndarray
            Energy of the packet for each interferometer.
        - si : np.ndarray
            Sine component of the rotation for each interferometer.
        - co : np.ndarray
            Cosine component of the rotation for each interferometer.
        - a_save : np.ndarray
            Amplitude of the first component for each interferometer.
        - A_save : np.ndarray
            Amplitude of the second component for each interferometer.

    """
    n_ifo = len(v00)
    n_pix = len(v00[0])
    _o = float(0.0001)

    mk = np.empty(n_pix, dtype=np.float32)
    aa = np.zeros(n_ifo, dtype=np.float32)
    AA = np.zeros(n_ifo, dtype=np.float32)
    aA = np.zeros(n_ifo, dtype=np.float32)

    si = np.empty(n_ifo, dtype=np.float32)
    co = np.empty(n_ifo, dtype=np.float32)
    a = np.empty(n_ifo, dtype=np.float32)
    A = np.empty(n_ifo, dtype=np.float32)
    a_save = np.empty(n_ifo, dtype=np.float32)
    A_save = np.empty(n_ifo, dtype=np.float32)

    for i in range(n_pix):
        mk[i] = float32(1.0) if mask[i] > 0 else float32(0.)

    for j in range(n_ifo):
        for i in range(n_pix):
            aa[j] += mk[i] * (v00[j][i] * v00[j][i])
            AA[j] += mk[i] * (v90[j][i] * v90[j][i])
            aA[j] += mk[i] * (v00[j][i] * v90[j][i])

    E = np.empty(n_ifo, dtype=np.float32)
    for i in range(n_ifo):
        _si = float32(2.) * aA[i] # rotation 2*sin*cos*norm
        _co = aa[i] - AA[i]       # rotation (cos^2-sin^2)*norm
        # print(f"_si[{i}]: ", _si, f"_co[{i}]: ", _co)
        _x = aa[i] + AA[i] + _o   # total energy
        _cc = _co * _co           # cos^2
        _ss = _si * _si           # sin^2
        _nn = sqrt(_cc + _ss)     # co/si norm
        a[i] = sqrt((_x + _nn) / float32(2.))      # first component amplitude
        A[i] = sqrt(abs((_x - _nn) / float32(2.))) # second component energy
        _cc = _co / (_nn + _o)    # cos(2p)
        _ss = float32(1.) if _si > float32(0.) else float32(-1.)  # 1 if sin(2p)>0. or-1 if sin(2p)<0.
        si[i] = sqrt((float32(1.) - _cc) / float32(2.))           # |sin(p)|
        co[i] = sqrt((float32(1.) + _cc) / float32(2.)) * _ss     # cos(p)

        E[i] = (a[i] + A[i]) ** 2 / float32(2.)
        a_save[i] = a[i]
        A_save[i] = A[i]
        a[i] = float(1.0) / (a[i] + _o)
        A[i] = float(1.0) / (A[i] + _o)

    Ep = 0.
    for i in range(n_ifo):
        Ep += E[i]

    v00_updated = np.empty((n_ifo, n_pix), dtype=np.float32)
    v90_updated = np.empty((n_ifo, n_pix), dtype=np.float32)
    for j in range(n_ifo):
        for i in range(n_pix):
            _a = v00[j][i] * co[j] + v90[j][i] * si[j]
            _A = v90[j][i] * co[j] - v00[j][i] * si[j]
            v00_updated[j][i] = mk[i] * _a * a[j]
            v90_updated[j][i] = mk[i] * _A * A[j]

    return Ep/float32(2.), v00_updated, v90_updated, E, si, co, a_save, A_save


@njit(cache=True)
def packet_norm_numpy(p, q, xtalks, xtalks_lookup, mk, q_E):
    """Compute the norm of a packet of pixels.

    Parameters
    ----------
    p : np.ndarray
        The p component of the packet. p[ifo][pixel]
    q : np.ndarray
        The q component of the packet. q[ifo][pixel]
    xtalks : np.ndarray
        The cross-talk matrix. xtalks[pixel]
    xtalks_lookup : np.ndarray
        Lookup table for cross-talk ranges. xtalks_lookup[pixel]
    mk : np.ndarray
        Mask indicating valid pixels. mk[pixel]
    q_E : np.ndarray
        Energy threshold for the q component. q_E[ifo]

    Returns
    -------
    tuple
        - detector_snr : np.ndarray
            The detector SNR for each interferometer.
        - norm : np.ndarray
            The norm of the packet for each interferometer. Was II + 5 in cWB
        - rn : np.ndarray
            Halo noise
        - q_norm : np.ndarray
            The 90 degree component norms for each interferometer. Was I + i in cWB
    """
    n_pixels = len(p[0])
    n_ifos = len(p)
    _o = float32(1.e-12)

    q_norm = np.zeros((n_ifos, n_pixels))
    norm = np.zeros(n_ifos)
    rn = np.zeros(n_pixels)
    for i in range(n_pixels):
        if mk[i] <= 0.:
            continue
        xtalk_range = xtalks_lookup[i]
        xtalk = xtalks[xtalk_range[0]:xtalk_range[1]]
        xtalk_indexes = xtalk[:,0].astype(np.int32)
        xtalk_cc = np.vstack((xtalk[:,4], xtalk[:,5], xtalk[:,6], xtalk[:,7]))  # 4xM matrix
        # Select elements from p and q based on xtalk_indexes
        p_vec = p[:, xtalk_indexes]  # N*M matrix
        q_vec = q[:, xtalk_indexes]  # N*M matrix
        # Compute the sums using a vectorized approach
        # x = np.sum(xtalk_cc * np.array([q_vec, p_vec, q_vec, p_vec]), axis=1)  # 4-d vector

        # h = x * np.array([q[:, i], p[:, i], q[:, i], p[:, i]])
        x = np.vstack((np.dot(p_vec, xtalk_cc[0].T),
                      np.dot(p_vec, xtalk_cc[1].T),
                      np.dot(q_vec, xtalk_cc[2].T),
                      np.dot(q_vec, xtalk_cc[3].T)))  # 4xN matrix

        # Summing all components together
        t = (x[0] * p[:, i]) + (x[1] * q[:, i]) + (x[2] * p[:, i]) + (x[3] * q[:, i])

        # if i == 0:
        #     print('xtalk_cc: ', xtalk_cc)
        #     print('xtalk: ', xtalk[0], xtalk[4], xtalk[5], xtalk[6], xtalk[7])
        #     print('x: ', x)
        #     print('t: ', t)
        #     print('p[0, i]: ', p[0, i])
        #     print('q[0, i]: ', q[0, i])


        # set t to 0 if t < 0
        norm += np.where(t < 0, 0, t)

        e = (p[:, i] * p[:, i] + q[:, i] * q[:, i]) / (t + _o)  # 1-d vector

        q_norm[:, i] = np.where(e > 1, 0, e)

        u = x[0] + x[2]
        v = x[1] + x[3]
        rn[i] = np.sum(u * u + v * v)

    # print('q: ', norm)
    e = q_E * 2   # TF-Domain SNR
    norm = np.where(norm < float32(2.), float32(2.), norm)  # set norm to 2 if norm < 2
    detector_snr = e / norm  # detector {0:NIFO} SNR

    return detector_snr, norm, rn, q_norm


@njit(cache=True)
def gw_norm_numpy(q_norm, q_E, p_E, ec):
    """
     set signal norms, return signal SNR

    Parameters
    ----------
    data_norm : np.ndarray
        The normalized data for each interferometer. data_norm[ifo][pixel]
    q_norm : np.ndarray
        The 90 degree component norms for each interferometer. q_norm[ifo][pixel]
    p_E : np.ndarray
        The energy threshold for the p component. p_E[ifo], was p[II+4]**2 / 2 in cWB
    ec : np.ndarray
        The energy component for each interferometer. ec[ifo][pixel]

    Returns
    -------
    tuple
        - total_norm : float
            The total SNR
        - norm : np.ndarray
            The detector SNR for each interferometer.
        - p_norm_II5 : np.ndarray
            Was II5 in cWB
        - p_norm : np.ndarray
            The 00 degree component norms for each interferometer.
    """
    n_pixels = len(q_norm[0])
    n_ifos = len(q_norm)

    norm = np.zeros(n_ifos)
    new_p_E = np.zeros(n_ifos)
    p_norm = np.zeros((n_ifos, n_pixels))

    for i in range(n_ifos):
        norm[i] = q_E[i]  # get data norms
        new_p_E[i] = norm[i]  # save norms
        e = p_E[i] * 2  # TF-Domain SNR, here the p_E is the p[II+4]**2 / 2 in cWB
        norm[i] = e / norm[i]  # detector {0:NIFO} SNR
        for j in range(n_pixels):
            p_norm[i][j] = np.where(ec[j] > 0, q_norm[i][j], 0)  # set p norms

    total_norm = np.sum(norm)  # total norm
    return total_norm, norm, new_p_E, p_norm


@njit
def orthogonalize_and_rotate(p, q, pAVX, length):
    event_mask = pAVX[1]
    rotation_sin = pAVX[4]
    rotation_cos = pAVX[5]
    first_component_energy = pAVX[15]
    second_component_energy = pAVX[16]
    energy_accumulated_first = np.zeros_like(p[0])
    energy_accumulated_second = np.zeros_like(q[0])

    for i in range(0, length, 4):
        accumulated_p_square = np.zeros_like(p[0])
        accumulated_q_square = np.zeros_like(q[0])
        accumulated_pq_product = np.zeros_like(p[0])

        for j in range(8):
            partial_p = p[j][i:i+4]
            partial_q = q[j][i:i+4]

            accumulated_p_square += partial_p * partial_p
            accumulated_q_square += partial_q * partial_q
            accumulated_pq_product += partial_p * partial_q

        event_occurance = (event_mask[i//4] > 0.0) * 1.0
        rotation_sin[i//4] = accumulated_pq_product * 2.0
        rotation_cos[i//4] = accumulated_p_square - accumulated_q_square

        total_energy = accumulated_p_square + accumulated_q_square + 1.e-21
        cos_square = rotation_cos[i//4]**2
        sin_square = rotation_sin[i//4]**2
        cos_sin_norm = np.sqrt(cos_square + sin_square)

        first_component_energy[i//4] = (total_energy + cos_sin_norm) / 2.0
        second_component_energy[i//4] = (total_energy - cos_sin_norm) / 2.0

        cos_divided = rotation_cos[i//4] / (cos_sin_norm + 1.e-21)
        sin_positive = (rotation_sin[i//4] > 0.0) * 1.0
        sin_value = 2.0 * sin_positive - 1.0

        rotation_sin[i//4] = np.sqrt((1.0 - cos_divided) / 2.0)
        rotation_cos[i//4] = np.sqrt((1.0 + cos_divided) / 2.0) * sin_value

        energy_accumulated_first += event_occurance * first_component_energy[i//4]
        energy_accumulated_second += event_occurance * second_component_energy[i//4]

    pAVX[1] = event_mask
    pAVX[4] = rotation_sin
    pAVX[5] = rotation_cos
    pAVX[15] = first_component_energy
    pAVX[16] = second_component_energy

    return np.sum(energy_accumulated_first) + np.sum(energy_accumulated_second), pAVX
