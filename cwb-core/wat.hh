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


#ifndef WAT_HH
#define WAT_HH

const int NIFO_MAX=8;
const int NRES_MAX=12;

namespace WAT {

// return if HEALPix is enabled/disabled
#ifdef _USE_HEALPIX
inline bool USE_HEALPIX() {return true;}
#else
inline bool USE_HEALPIX() {return false;}
#endif

// return if LAL is enabled/disabled
#ifdef _USE_LAL
inline bool USE_LAL() {return true;}
#else
inline bool USE_LAL() {return false;}
#endif

// return if EBBH is enabled/disabled
#ifdef _USE_EBBH
inline bool USE_EBBH() {return true;}
#else
inline bool USE_EBBH() {return false;}
#endif

// return if ROOT6 is enabled/disabled
#ifdef _USE_ROOT6
inline bool USE_ROOT6() {return true;}
#else
inline bool USE_ROOT6() {return false;}
#endif

// return if CPP is enabled/disabled
#ifdef _USE_CPP
inline bool USE_CPP() {return true;}
#else
inline bool USE_CPP() {return false;}
#endif

// return if ICC is enabled/disabled
#ifdef _USE_ICC
inline bool USE_ICC() {return true;}
#else
inline bool USE_ICC() {return false;}
#endif

} // end namespace

#ifndef XIFO
#define XIFO 4
#endif

#define _ALIGNED        __attribute__((aligned(32)))

#if XIFO < 5
#define NIFO 4
#endif

#if XIFO > 4
#define NIFO 8
#endif

#if XIFO < 5
#define _NET(P1,P2) \
P1                              
#endif

#if XIFO > 4
#define _NET(P1,P2) \
P1                                      \
P2                              
#endif

#if XIFO == 1
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              
#endif

#if XIFO == 2
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              
#endif

#if XIFO == 3
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              
#endif

#if XIFO == 4
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                             	  
#endif

#if XIFO == 5
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              
#endif

#if XIFO == 6
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              
#endif

#if XIFO == 7
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              	 \
P7                              
#endif

#if XIFO == 8
#define NETX(P1,P2,P3,P4,P5,P6,P7,P8) \
P1                              	 \
P2                              	 \
P3                              	 \
P4                              	 \
P5                              	 \
P6                              	 \
P7                              	 \
P8                              
#endif

#if XIFO==1
#define XSUM(X)   X[0]
#define YSUM(Y)	  0
#endif
#if XIFO==2
#define XSUM(X)   X[0]+X[1]
#define YSUM(Y)	  0
#endif
#if XIFO==3
#define XSUM(X)   X[0]+X[1]+X[2]
#define YSUM(Y)	  0
#endif
#if XIFO==4
#define XSUM(X)   X[0]+X[1]+X[2]+X[3]
#define YSUM(Y)	  0
#endif
#if XIFO==5
#define XSUM(X)   X[0]+X[1]+X[2]+X[3]
#define YSUM(Y)   Y[0]
#endif
#if XIFO==6
#define XSUM(X)   X[0]+X[1]+X[2]+X[3]
#define YSUM(Y)   Y[0]+Y[1]
#endif
#if XIFO==7
#define XSUM(X)   X[0]+X[1]+X[2]+X[3]
#define YSUM(Y)   Y[0]+Y[1]+Y[2]
#endif
#if XIFO==8
#define XSUM(X)   X[0]+X[1]+X[2]+X[3]
#define YSUM(Y)   Y[0]+Y[1]+Y[2]+Y[3]
#endif

#endif

;  // DO NOT REMOVE !!!
