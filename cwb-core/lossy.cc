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


/*-------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   lossy.cc
 * Author: S.Klimenko University of Florida
 *-------------------------------------------------------
*/

// Changes in version 2.0:
// new set of wavelet classes
// Base version number is 2.0. Uses WAT v.2.0

#include <stdio.h>
#include "waverdc.hh"
#include "Biorthogonal.hh"
#include "wseries.hh"
#include "lossy.hh"

int Compress(wavearray<double> &in, int* &out, int Lwt, int Lbt,
              const double g1, const double g2, int np1, int np2)
{
// in - input data
// Lw - decomposition depth for wavelet Tree
// Lb - decomposition depth for wavelet binary Tree
// g1, g2 - are value of losses in percents
// np1, np2 - are orders of wavelets

   int i;
   WaveRDC z;
   bool sign = (g1<0 || g2<0) ? true : false;
   bool skip = (g1>=100 && g2>=100) ? true : false;

   if (np1<=0) np1=8;
   if (np2<=0) np2=8;

   wavearray<double> *p = &in;	              // temporary pointer
   wavearray<double> wl;		      // temporary storage
   wavearray<short>  ws;		      // temporary short array

   Biorthogonal<double> B1(np1,0,B_POLYNOM);   // Lifting wavelet
   Biorthogonal<double> B2(np2,1,B_POLYNOM);   // Lifting wavelet
//   Biorthogonal<double> B1(np1,0,B_PAD_EDGE);   // Lifting wavelet
//   Biorthogonal<double> B2(np2,1,B_PAD_EDGE);   // Lifting wavelet
   WSeries<double> W1;           	       // wavelet data container
   WSeries<double> W2;           	       // wavelet data container

   double mean, rms, a;

   if(!sign && !skip){
      a=in.getStatistics(mean,rms);
      z.rmsLimit=fabs(a)<0.5 ? fabs(2*rms*a) : rms;
   }

// wavelet tree

   if(Lwt>0 && !skip){
      W1.Forward(in,B1,Lwt);
      
      if(sign){
	 W1.getLayer(wl,0);
	 p = (wavearray<double>*)&W1;
      }

      else
	 for(i=Lwt; i>=0; i--) {
	    W1.getLayer(wl,i);
	    if(Lbt>0 && i==0) break;
	    z.Compress(wl,g1);
	 }
   }

// binary tree

   if(Lbt>0 && !skip) {                       
      p = (Lwt>0) ? &wl : &in; 
      W2.Forward(*p,B2,Lbt);
      p = (wavearray<double>*)&W2;

      if(sign){
	 if(Lwt>0){
	    W1.putLayer(*p,0);
	    p = (wavearray<double>*)&W1;
	 }
      }

      else   
	 for(i=(1<<Lbt)-1; i>=0; i--) {
	    W2.getLayer(wl,i);
	    z.Compress(wl,g2);
	 }
   }
   
// no wavelets or sign transform

   if (skip || sign || ((Lwt==0) && (Lbt==0))) {  

	 if(!sign && !skip) 
	    z.Compress(in, (g1>g2) ? g1 : g2);

	 else if(sign){
	    z.getSign(*p,ws); 
	    z.Compress(ws);
	 }
	 
	 else{
	    ws.resize(p->size());
	    ws = 0;
	    z.Compress(ws);
	 }
   }

//	z.Dir(1);

/****************************************************************
 * COMPRESSED DATA OUTPUT FORMAT                                *
 ****************************************************************
 * out = { "WZVv", NN, NU, np1, np2, Lwt, Lbt,                  *
 *          sl1[Lwt], sl2[2^Lbt], z.dataz[NZ] },                *
 * NZ = NN-(4+NL) - packed data length in 32-bit words          *
 * NL = Lwt+2^Lbt - total number of compressed data layers      *
 * -------------------------------------------------------------*
 * shift|length: meaning (shift and lenght in bytes)            *
 * -------------------------------------------------------------*
 * 0|4 : WZVv - ID string with version number (ver 1.5 -> WZ15) * 
 * 4|4 : NN - length of output data in 32-bit words	        *
 * 8|4 : NU - length of unpacked data in 16-bit words           *
 * 12|4: np1, np2, Lwt, Lbt - byte-length numbers               *
 * 16|4: skew=data[end]-data[start] - floating point number     *
 * 20|4*NL : sl1 and sl2 - arrays of floating point numbers     *
 * 20+4*NL|4*NZ : dataz[NZ] - array of compressed data          *
 ****************************************************************
*/
  size_t nsw = 4;               // length of service block
  int N = nsw+z.size();

  if (!out) {
     out = (int *) malloc(N*sizeof(int));
     if (!out) cout << "Memory allocation error\n";
  }

  i = 0;

// write id "WZ16" in byte-order dependent manner
  out[0]  = ('W'<<24) + ('Z'<<16) + ('2'<<8) + ('0'); 
  out[1]  = N;		        // output data length in 32-bit words
  out[2]  = z.nSample;		// uncompressed data length in 16-bit words
  out[3]  = (Lwt & 0xFF)  << 24;
  out[3] |= (Lbt & 0xFF)  << 16;
  out[3] |= (np1 & 0xFF)  << 8;
  out[3] |= (np2 & 0xFF)  << 1;
  out[3] |= (sign) ? 1 : 0;

  for (i=nsw; i<N ; i++) {
    out[i] = z.data[i-nsw];
  } 
  return N;
}

int unCompress(int *in, wavearray<float> &out)
{
/****************************************************************
 * COMPRESSED DATA INPUT FORMAT                                 *
 ****************************************************************
 * in = { "WZVv", NN, NU, np1, np2, Lwt, Lbt,                   *
 *          sl1[Lwt], sl2[2^Lbt], z.dataz[NZ] },                *
 * NZ = NN-(4+NL) - packed data length in 32-bit words          *
 * NL = Lwt+2^Lbt - total number of compressed data layers      *
 * -------------------------------------------------------------*
 * shift|length: meaning (shift and lenght in bytes)            *
 * -------------------------------------------------------------*
 * 0|4 : WZVv - ID string with version number (ver 1.5 -> WZ15) *
 * 4|4 : NN - length of output data in 32-bit words             *
 * 8|4 : NU - length of unpacked data in 16-bit words           *
 * 12|4: Lwt, Lbt, np1, np2 - byte-length numbers               *
 * 16|4: skew=data[end]-data[start] - floating point number     *
 * 20|4*NL : sl1 and sl2 - arrays of loating point numbers      *
 * 20+4*NL|4*NZ : dataz[NZ] - array of compressed data          *
 ****************************************************************
*/
  int *p = in+4;
  int n;
  short *id, *iv;			// pointers to 2-byte id and version #
  short aid, aiv;			// actual id and ver.# taken from input
  short swp=0x1234;
  char *pswp=(char *) &swp;
//  size_t nsw = 4;                       // length of the service block

// it is assumed, that frame-reading software already changed byte order if
// compress and uncompress is performed on machines with different endness 
  if (pswp[0]==0x34) { 			// little-endian machine
    id = (short *)"ZW";
    iv = (short *)"02";
  }
  else {				// big-endian machine
    id = (short *)"WZ";
    iv = (short *)"20";
  }
  aid = short(in[0] >> 16);
  aiv = short(in[0] & 0xFFFF);

//  printf(" in[0]=%4s id=%2s, iv=%2s\n", (char*)in, (char*)id, (char*)iv);

  if (aid != *id) {
    cout << " unCompress: error, unknown input data format.\n";
    return -1;
  }
  if (aiv!= *iv) {
    cout << " unCompress: error, unsupported version of input data format.\n";
    return -1;
  }

  int i = 0;
  int n32 = in[1];	          // length of input data in 32-bit words
  if (n32<4) return -1;

  int n16 =  in[2];	          // length of uncompressed data in 16-bit words
  int Lwt = (in[3]>>24) & 0xFF;   // number of Wavelet Tree decomposition levels
  int Lbt = (in[3]>>16) & 0xFF;	  // number of Binary Tree decomposition levels
  int np1 = (in[3]>>8)  & 0xFF;
  int np2 = (in[3]>>1)  & 0x7F;
  bool sign = in[3] & 1;          // check sign bit

  n32 -= 4;                       // length of compressed data in 32-bit words
  if (n32<=0) return -1;


  int nl1=0, nl2=0;
  if (Lwt >= 0) nl1 = Lbt>0 ? Lwt : Lwt+1;
  if (Lbt >  0) nl2 = 1<<Lbt;

  WaveRDC z;
  z.resize(n32);
  z.nLayer = (1<<Lbt) + Lwt;

  for (int j=0; j<n32 ; j++) {
    z.data[j] = *(p++);
  }

  if (sign || (Lwt==0 && Lbt==0)) {
    return z.unCompress(out);
  }

  wavearray<double>* pwl;		      // temporary pointer
  wavearray<double>   wl;		      // temporary storage
  
  Biorthogonal<double> B1(np1,0,B_POLYNOM);   // Lifting wavelet
  Biorthogonal<double> B2(np2,1,B_POLYNOM);   // Lifting wavelet
//  Biorthogonal<double> B1(np1,0,B_PAD_EDGE);   // Lifting wavelet
//  Biorthogonal<double> B2(np2,1,B_PAD_EDGE);   // Lifting wavelet
  WSeries<double> W1(B1);           	      // wavelet data container
  WSeries<double> W2(B2);           	      // wavelet data container

  int nL;                        // number of layers

  if(Lbt>0){                     // binary tree processing
     nL = 1<<Lbt;
     W2.resize(n16/(1<<Lwt));
     W2.pWavelet->setLevel(Lbt);

     for(i=nL; i>0; i--) {
	n=z.unCompress(wl,Lwt+i); // the last layer is layer 0
	W2.putLayer(wl,nL-i);
     }
     W2.Inverse();
  }

  pwl = &wl;
  if(Lwt>0){                     	// wavelet tree processing
     W1.resize(n16);
     W1.pWavelet->setLevel(Lwt);
 
    for(i=Lwt; i>=0; i--) {
       if(Lbt==0 || i>0) n=z.unCompress(*pwl,Lwt-i+1);
       else pwl = (wavearray<double>*)&W2;
       W1.putLayer(*pwl,i);
    }
    W1.Inverse();
  }

  if(out.size() != (size_t)n16) out.resize(n16);

  if (Lwt==0)
     for (i=0; i<n16; i++) out.data[i] = (float)W2.data[i];
  else
     for (i=0; i<n16; i++) out.data[i] = (float)W1.data[i];
     
  return n16;
}

int Compress(float in[], int nn, int* &out, int Lwt, int Lbt,
              double g1, double g2, int np1, int np2)
{
   wavearray<double> wa(nn);
   for (int i=0; i<nn; i++) wa.data[i]=in[i];
   return Compress(wa, out, Lwt, Lbt, g1, g2, np1, np2);
}

int Compress(int in[], int nn, int* &out, int Lwt, int Lbt,
              double g1, double g2, int np1, int np2)
{
   wavearray<double> wa(nn);
   for (int i=0; i<nn; i++) wa.data[i]=in[i];
   return Compress(wa, out, Lwt, Lbt, g1, g2, np1, np2);
}

int Compress(short in[], int nn, int* &out, int Lwt, int Lbt,
              double g1, double g2, int np1, int np2)
{
   wavearray<double> wa(nn);
   for (int i=0; i<nn; i++) wa.data[i]=in[i];
   return Compress(wa, out, Lwt, Lbt, g1, g2, np1, np2);
}

int unCompress(int* in, short* &out)
{
   wavearray<float> wa;
   int nn, n16;
   double d;
   n16 = unCompress(in, wa);
   nn = wa.size();
   if(out) free(out);
   out = (short *) malloc(nn*sizeof(short));

// rounding data
   for(int i=0; i<nn; i++) {
     d=wa.data[i];
     out[i]=short(d>0.? d+0.5: d-0.5);
   }

   return n16;
}

int unCompress(int* in, float* &out)
{
   wavearray<float> wa;
   int n16=unCompress(in, wa);
   int nn = wa.size();
   if(out) free(out);
   out = (float *) malloc(nn*sizeof(float));
   for (int i=0; i<nn; i++) out[i] = wa.data[i];
   return n16;
}










