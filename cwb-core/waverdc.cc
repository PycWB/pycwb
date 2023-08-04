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
 * Package: 	Wavelet Analysis Tool
 * File name: 	waverdc.cc  :: Random Data Compression
 *-------------------------------------------------------
*/

#include "waverdc.hh"

ClassImp(WaveRDC)

// Default constructor
WaveRDC::WaveRDC() : wavearray<unsigned int>()
{
  nLayer = 0;
  nSample = 0;
  optz = 0;
}

// Destructor
WaveRDC::~WaveRDC(){ }


// Lists layers stored in compressed array,
// "v" - is verbosity level, =0 by default.

void WaveRDC::Dir(int v)
{

  if (size() <= 0) { 
    cout <<" Compressed array is empty.\n";
    return;
  }

  if (v > 0) {

    cout <<"\n Number of layers :"<< nLayer <<"\n";

    int ll=78;
    size_t ncd = 0;
    double r;
    size_t n32 = 0;
    size_t n16 = 0;

    for (int i=0; i<ll; i++) printf("-"); printf("\n");

    printf("%s%s\n",
           "|layer|compressed |uncompressed|    ratio |",
           "kBSW|sh,lo|  shift | scale |  opt |");

    for (int i=0; i<ll; i++) printf("-"); printf("\n");
 
    for (int i = 0; i < nLayer; i++) {

      if ((ncd + 2) > size()) break;

      optz = data[ncd] & 0xFFFF;

      kShort = (data[ncd] >> 26) & 0x1F;
       kLong = (data[ncd] >> 21) & 0x1F;
        kBSW = (data[ncd] >> 16) & 0x1F;
	Bias = 0;

      if(optz & 0x2 ){        // 4 bytes was used to store number of words
	 n32 = data[++ncd];   // # of 32b words occupied by compressed data
	 n16 = data[++ncd];   // # of 16b words occupied by original data
      }
      else if(optz & 0x1){    // 2 bytes was used to store number of words
	 n32 = data[++ncd] & 0xFFFF;
	 n16 = (data[ncd]>>16) & 0xFFFF;
      }
      else{                 // 1 byte was used to store number of words
	 n32 = data[++ncd] & 0xFF;
	 n16 = (data[ncd]>>8) & 0xFF;
	 Bias = short((data[ncd]>>16) & 0xFFFF);
      }

      if (n32 < getLSW(optz)) {
	 cout <<" unCompress() error: invalid layer number "<< i+1 <<"\n";
	 return;
      }

      if(!n16) break;

      Zero  = (optz & 0x4) ? 1<<(kShort-1) : 0;
      Bias  = ((optz & 0x8) && (optz & 0x3)) ? data[++ncd] & 0xFFFF : Bias;
      Scale = (optz & 0x10) ? *((float *)&data[++ncd]) : 1.;

      ncd += n32 - getLSW(optz) + 1;

      n32 *= sizeof(*data);
      n16 *= sizeof(short);
      r = double(n16)/double(n32);

      printf("|%4d |%10d |%11d |%9.3f |%3d |%2d,%2d|%8d",
             i+1, (int)n32, (int)n16, r, (int)kBSW, (int)kShort, (int)kLong, (int)Bias);
      printf("|%6.1f |%5o |\n", Scale, short(optz));

    }
    for (int i=0; i<ll; i++) printf("-"); printf("\n");
  }

  cout <<" Total compressed length   : "
       << size()*sizeof(*data) <<" bytes.\n";
  cout <<" Total uncompressed length : "
       << nSample*2 <<" bytes.\n";

  cout <<" Overall compression ratio : "
       << double(nSample*2)/double(size()*sizeof(*data))<<"\n\n";
}

WaveRDC& WaveRDC::operator=(const WaveRDC &a){
   size_t n = a.size();
  if (this != &a && n > 0) {
    if (size() != n) resize(n);
    for (size_t i=0; i < n; i++) data[i] = a.data[i];
    nSample = a.nSample;
     nLayer = a.nLayer;
       optz = a.optz;
  }
  return *this;
}

// Concatenation of comressed data
WaveRDC& WaveRDC::operator+=(const WaveRDC &a)
{
  size_t nZ = size();
  size_t nA = a.size();
  size_t i;
  unsigned int *pnew;
    
  if (a.size() > 0) {
    pnew = new unsigned int[size() + a.size()];
    for (i = 0; i < nZ; i++) pnew[i] = data[i];
    pnew += nZ;
    for (i = 0; i < nA; i++) pnew[i] = a.data[i];

    cout<<nZ<<"  "<<nA<<"\n";
    delete [] data;
    cout<<nZ<<"  "<<nA<<"\n";
    data = pnew-nZ;
    nSample += a.nSample;
     nLayer += a.nLayer; 
  }
  return *this;
}

// Dumps data array to file *fname in binary format and type "short".
// Variable 'app' select write mode, app=0 - write new file,
// app=1 - append to existent file.
int WaveRDC::DumpRDC(const char *fname, int app)
{
  const char *mode = "wb";
  if (app == 1) mode = (char*)"ab";

  FILE *fp;
  if ( (fp = fopen(fname, mode)) == NULL ) {
      cout << " DumpRDC() error: cannot open file " << fname <<". \n";
      return -1;
  }

  int n = size() * sizeof(*data);
  fwrite(data, n, 1, fp);
  fclose(fp);
  return n;
}

// Check if data will fit into short, return scale factor
 float WaveRDC::getScale(const waveDouble &F, double loss)
{
  if(!F.size()) return 0.;

  double mean, bias, rms;
  float scale;
  size_t N = F.size()-1;
  double *pF = F.data;
  double x = 0.;
  double y = pF[0];
  size_t i;

  loss = sqrt(0.12*loss);
  optz = 0;

// calculate the minimal rms and scale factor sf1
  x = F.getStatistics(mean,rms);
  if(rms > rmsLimit) rms = rmsLimit;
  scale = rms*loss;

  if(fabs(x) < 0.5){ 
     scale *= 2*fabs(x);
     if(x<0) optz |= 0x80;   // differentiation 
     if(x>0) optz |= 0x100;  // heterodine & differentiation 
  }

  if(scale <=0.) scale = 1.;
  bias = mean;

  if(optz & 0x180){       
     bias = (pF[N]-pF[0])/(N+1);

     if(optz & 0x100){        // heterodine + differentiation
	for(i=N; i>0; i--){ 
	   x = pF[i]+pF[i-1];
	   if(fabs(x) > fabs(y)) y = x;
	}
	bias -= (N&1) ? 2*pF[N]/(N+1) : 0.; 
     }
     else                    // differentiation
	for(i=N; i>0; i--){ 
	   x = pF[i]-pF[i-1];
	   if(fabs(x) > fabs(y)) y = x;
	}

   }
  else{                       // no handling
     for(i=0; i<=N; i++) 
	if(fabs(pF[i]) > fabs(y)) y = pF[i];
  }

  y -= bias;
  x = (optz & 0x180) ? pF[0] : bias;
  y = (fabs(x)>fabs(y)) ? x : y;
  y /= (scale>0.) ? scale : 1;

  bias = double(1<<(sizeof(short)*8-1))-1.;   
  scale *=  (fabs(y) > bias) ? fabs(y)/bias : 1.;   
  if(scale <=0.) scale = 1.;
  Bias = wint(mean/scale);
  Scale = scale;

//  cout<<rms<<"  scale="<<Scale<<"  bias="<<Bias<<"  ";
//  printf("%o \n", short(optz));

  return scale;
}     

// convert double to int/short
// Scale and Bias should be set before this function is executed
 void WaveRDC::getShort(const waveDouble &F, waveShort &S)
{
  if(!F.size()) return;

  if(F.size() != S.size()) S.resize(F.size());

  size_t n16 = F.size();
  size_t   N = n16 - 1;
  size_t i;
  int x;
  double *pF = F.data;
  short  *pS = S.data;
  double s = (Scale>0.) ? 1./Scale : 1.;

  int mean = wint(s*(F.data[N]-F.data[0])/n16);
   if(n16&1 && optz&0x100) 
      mean = -wint(s*(F.data[N]+F.data[0])/n16);

  if(optz & 0x180){         // differentiation

     if(optz & 0x100){         // heterodine & differentiation 
	for(i=1; i<N; i+=2){
	   x = wint(F.data[i]*s); 
	   pS[i]   = -(x + wint(s*pF[i-1]) - mean);
	   pS[i+1] =  (wint(s*pF[i+1]) + x + mean);
	}
	if(!(n16&1)) pS[N] = -wint(s*pF[N])-wint(s*pF[N-1])+mean;
     }
     else{                     // differentiation
	for(i=1; i<N; i+=2){
	   x = wint(F.data[i]*s); 
	   pS[i]   = x - wint(s*pF[i-1]) - mean;
	   pS[i+1] = wint(s*pF[i+1]) - x - mean;
	} 
	if(!(n16&1)) pS[N] = wint(s*pF[N])-wint(s*pF[N-1])-mean;
     }
     
     Bias = wint(pF[0]*s);
     pS[0] = (optz & 0x100) ? -mean : mean;
  }

  else
     for(i=0; i<n16; i++){ 
	pS[i] = short(wint(s*pF[i]) - Bias);
     }
}     

// convert sign statistics
 void WaveRDC::getSign(const waveDouble &F, waveShort &S)
{
   if(!F.size()) return;

   if(F.size() != S.size()) S.resize(F.size());

   size_t N = F.size();
   size_t i;
   double *pF = F.data;
   short  *pS = S.data;

   optz = 0x20;

   for(i=0; i<N; i++)
      pS[i] = (pF[i]>=0.) ? 1 : -1; 

}     

// Compress packs data into layer which is series of blocks each
// consisting of bricks of two different length 'kLong' and 'kShort'. Each
// brick represents one element of original unpacked data
// (16-bit integer) which is shortened to exclude redundant bits.
// Original data are taken from object wd. Packed data are
// placed into array 'data'. Returns number of bytes
// occupied by compressed data.
int WaveRDC::Compress(const waveDouble &F, double loss)
{
  if(F.size()<1) return 0;
  wavearray<short>  S(F.size());
  optz = 0;

  Scale = getScale(F, loss);  // calculate the scale factor
          getShort(F, S);     // convert to short array

  if(Bias  |= 0)  optz |= 0x8;
  if(Scale != 1.) optz |= 0x10;

  return Compress(S);
}


int WaveRDC::Compress(const waveShort &S)
{
  int cdLSW = 5;           // length of LSW buffer
  int n16 = S.size();      // number of uncompressed samples
  short *dt = S.data;      // pointer to input data
  int i;
  int x;
  int y;

  freebits = 8*sizeof(unsigned int);

// calculate histogram
// "x" falls in histogram bin "k" if it needs k bits for encoding;
// "k" is the minimum number of bits to encode x

  int h[17], g[17];
  int k,m;

  for (i = 0; i < 17; i++) {h[i]=0; g[i]=0;}

  for (i = 0; i < n16; i++) {
     x = dt[i];
     y = (x>0) ? x : -x;
     for(k=0; (1<<k) < y; k++);
     m = 1<<k;                          // number of "short zeroes" 
     if(y > 0) k++;                     // add bit for sign
     h[k] += 1;                         // histogram for number of bits
     if(m == x) g[k] += 1;              // histogram for "short zeroes"
  }

//printf(" h[0..8] =");
//for (i = 0; i < 9; i++) printf("%7d",h[i]);
//printf("\n h[9..16]=");
//for (i = 9; i < 17; i++) printf("%7d",h[i]);
//printf("\n");

// ****************************************************************
// * estimate compressed size using histogram h[]
// * and find optimal kShort to get minimal compressed length
// ****************************************************************

// kShort/kLong - number of bits to pack short/long data samples
  kBSW = kShort = kLong = 16;  

// "compressed" length in bytes if kShort = kLong = 16
  int Lmin = 2*n16+2;     

  int m0,ksw;
  int L = 0;

  m = 0;
  for (i = 16; i > 0; i--) {
     m += h[i];                         // # of long bricks
     if (m == 0) kLong = i-1;           // # of bits for long brick
     m0 = g[i-1]<h[0] ? g[i-1] : h[0];  // # of zeroes

     ksw = 16;                            
     if(m+m0 > 0) for(ksw=0; (1<<ksw) < (n16/(m+m0)+2); ksw++);
     ksw = ksw<15 ? ksw+2 : 16;         // # of bits in BSW

     L = ((m+m0)*ksw + (n16-m-m0)*(i-1) + m*kLong)/8 + 1;
     if (L <= Lmin) {
	Lmin = L; 
	kShort = i - 1;
	kBSW = ksw;
        Zero = (kShort==0 || g[kShort]>h[0]) ? 0 : 1<<(kShort-1); // zero in int domain
     }

//     printf(" h[%2d] = %7d g[%2d] = %7d L=%7d Zero=%7d m0=%7d h[0]=%7d\n",
//             i, h[i], i, g[i], L, Zero, m0, h[0]);

  }

//  cout <<" kBSW="<<kBSW<<" kShort="<<kShort<<" kLong="<<kLong<<" Lmin="<<Lmin<<"\n";
  
// ****************************************************************
// *  pack data using kShort bits for small numbers and kLong for large: *
// ****************************************************************
//           encoding  example if kShort = 3
//   -4    -3    -2    -1     4    1    2    3    0
//   110   100   10     0          1   11   101  111
//                            |___ represented by 0 in long word
// 
//  0 - coded by 111, 
//  4 - max number coded with kShort bits; will be coded by 0 in long word 

  
  int ncd = cdLSW;                  // ncd counts elements of compressed array "cd"
  int j, jj = 0;
  unsigned int bsw = 0;             // Block Service Word
  int ns, maxns;                    // ns - # of short words in block
  maxns = (1<<(kBSW-1)) - 1;        // maxns - max # of short bricks in block

  int lcd = 6 + Lmin + n16/maxns/2 + 2;  // length of the cd array
  unsigned int *cd = new unsigned int[lcd];

  for(i=0; i<lcd; i++) cd[i]=0;     // Sazonov 01/29/2001

//  cout<<"lcd="<<lcd<<" ncd="<<ncd<<" shift="<<Bias<<endl;


  if(kShort > 0) m = 1<<(kShort-1); // m - max # encoded with kShort bits
  int jmax = 0;			    // current limit on array index
  while (jj < n16) { 
     j = jj; 
     jmax = ((j+maxns) < n16) ? j+maxns : n16;
     
     if(kShort == 0)                 // count # of real 0
	while (j<jmax && dt[j]==0) j++;
     else                            // count ns for |td|<=m, excluding zero
	while (j<jmax && wabs(dt[j])<=m && dt[j] != Zero) j++;

     ns = j - jj;
     if(ns > maxns) ns = maxns;      // maxns - max number of short words
     j = jj + ns;

//  fill BSW
     bsw = ns << 1;
     if(j<n16){
	if(kShort == 0 && dt[j] != 0)    bsw += 1;
	if(kShort  > 0 && dt[j] != Zero) bsw += 1;
     }

// ***************************************************************
//  generate compressed data skiping zero word that is always long
// ***************************************************************

     Push(bsw, cd, ncd, kBSW);                           // push in BSW  

     if(kShort != 0) Push(dt, jj, cd, ncd, kShort, ns);  // push in a short words
     jj += ns;

     if(j<n16 && dt[j] != Zero) Push(dt,j,cd,ncd,kLong,1); // push in a long word
     jj++;
  }

//  cout<<"**0** ncd="<<ncd<<"  bsw="<<bsw<<"  j="<<jj<<"  fb="<<freebits<<"\n";
//  printf(" data[%6d] = %o data[%6d] = %o\n", ncd, cd[ncd], ncd-1, cd[ncd-1]);

  ncd++;

//  cout<<"lcd="<<lcd<<" ncd="<<ncd<<" bias="<<Bias<<" scale="<<Scale<<endl;
//  cout<<"zero="<<Zero<<" bias="<<Bias<<" scale="<<Scale<<endl;

// compression options defined by first byte.
// bit     if set                               # of bytes
//  1      number of compressed words < 64k         2
//  2      number of compressed words > 64k         4
//         if(!optz&3)                              1
//  3      Zero  != 0                              0-2   
//  4      Bias  != 0                              0-2
//  5      Scale != 0                              0-4
//  6      sign bit                                    
//  7      ----
//  8      dif-                                   
//  9      dif+                                    

  if(ncd >= (1<<sizeof(short)*4)) optz |= 1;
  if(n16 >= (1<<sizeof(short)*4)) optz |= 1;
  if(ncd >= (1<<sizeof(short)*8)) optz |= 2;
  if(n16 >= (1<<sizeof(short)*8)) optz |= 2;
  if(Zero)                        optz |= 4;

  size_t n32 = ncd-cdLSW+getLSW(optz);   // actual length
  size_t nLSW = 2;

  cd[0] |= ((kShort & 0x1F)   << 26); // length of short word
  cd[0] |= ((kLong  & 0x1F)   << 21); // length of long word
  cd[0] |= ((kBSW   & 0x1F)   << 16); // length of the block service word
  cd[0] |= ((optz   & 0xFFFF)      ); // compression option

  if(optz & 0x2 ){      // use 4 bytes to store number of words
     cd[1] = n32;       // # of 32b words occupied by compressed data
     cd[2] = n16;       // # of 16b words occupied by original data
     nLSW++;
  }
  else if(optz & 0x1){  // use 2 bytes to store number of words
     cd[1] |= (n16 & 0xFFFF) << 16;
     cd[1] |= (n32 & 0xFFFF);
  }
  else{                 // use 1 byte to store number of words
     cd[1] |= (*((unsigned short *)&Bias) & 0xFFFF) << 16;
     cd[1] |= (n16 & 0xFF) << 8;
     cd[1] |= (n32 & 0xFF);
  }

  if((optz & 0x8) && (optz & 0x3))      // use 4 bytes to store Bias
     cd[nLSW++] |= (*((unsigned short *)&Bias) & 0xFFFF);

  if(optz & 0x10 )      // use 4 bytes to store Scale
     cd[nLSW++] |= (*((unsigned int *)&Scale) & 0xFFFF);

// concatenate *cd with *this object

  if (n32 > nLSW) {
     int N = size();
     resize(N+n32);                                  // resize this to a new size
     unsigned int *p = data + N + nLSW - cdLSW;

     for(i=0; i<(int)nLSW; i++) data[i+N] = cd[i];   // copy LSW
     for(i=cdLSW; i<ncd; i++)   p[i] = cd[i];        // copy compressed data

     nSample += n16;
     nLayer  += 1;
  }

  delete [] cd;
  return size()*sizeof(*data);
}

int WaveRDC::Push(short *dt, int j, unsigned int *cd, int &ncd,
                         int k, int n)
{
// Push takes "n" elements of data from "dt" and packs it as a series
// of "n" words of k-bit length to the compressed data array "cd".
// lcd counts the # of unsigned int words filled with compressed data.
// This 'Push' makes data conversion:
// x<0: x -> 2*(|x|-1)
// x>0: x -> 2*(|x|-1) + 1
// x=0: x -> 2*(Zero-1) + 1

   int x, jj;
   unsigned int u;

   if(n == 0 || k == 0) return 0;
   jj = j;

   for (int i = 0; i < n; i++) {
      x = (int)dt[jj++];
      u = x == 0 ? Zero-1 : wabs(x)-1;     // unsigned number
      u = x >= 0 ? (u<<1)+1 : u<<1;        // add sign (first bit); 0/1 = -/+
      Push(u, cd, ncd, k);
   }

   return jj-j;
}

void WaveRDC::Push(unsigned int &u, unsigned int *cd, int &ncd, int k)
{
// Push takes unsigned int u  and packs it as a word k-bit leng
// into the compressed data array "cd".
// ncd counts the # of unsigned int words filled with compressed data.

   if(k == 0) return;

   if (k <= freebits) {
     freebits -= k;
     cd[ncd] |= u << freebits;          // add whole word to cd[k] 
   }
   else {
     cd[ncd++] |= u >> (k - freebits);  // add upper bits to cd[k]
     freebits += 8*sizeof(u) - k;       // expected free space in cd[k+1]
     cd[ncd] = u << freebits;           // save lower bits in cd[k+1] 
   }
   return;
}


// Unpacks data from the layer 'm' which is series of blocks each
// consisting from bricks of two different length 'kLong' and 'kShort'.
// Each brick represents one element of original unpacked data
// (16-bit integer) which is shortened to exclude redundant bits.
// Returns number of bytes occupied by unpacked data.
int WaveRDC::unCompress(waveDouble &w, int m)
{
   wavearray<int> in;
   int n = unCompress(in, m);
   waveAssign(w,in);
   if(Scale != 1.) w *= Scale;
   return n;
}

int WaveRDC::unCompress(waveFloat &w, int m)
{
   wavearray<int> in;
   int n = unCompress(in, m);
   waveAssign(w,in);
   if(Scale != 1.) w *= Scale;
   return n;
}


int WaveRDC::unCompress(wavearray<int> &w, int m)
{
   if (m <=0 || m > nLayer) {
      cout << " UnCompress() error: layer "<< m<<" is unavailable.\n";
      return 0;
   }

   int    ncd = 0;
   int     mm = 0;
   size_t n32 = 0;
   size_t n16 = 0;

// find layer number 'm'

   while (mm < m) {
      ncd += n32;    // find beginning of the next layer

      if ((ncd + 2) > (int)size()) {
	 cout <<" unCompress() error: invalid layer number "<< mm <<"\n";
	 return 0;
      }

      optz = data[ncd] & 0xFFFF;
      Bias =0;

      if(optz & 0x2 ){        // 4 bytes was used to store number of words
	 n32 = data[ncd+1];   // # of 32b words occupied by compressed data
	 n16 = data[ncd+2];   // # of 16b words occupied by original data
      }
      else if(optz & 0x1){    // 2 bytes was used to store number of words
	 n32 = data[ncd+1] & 0xFFFF;
	 n16 = (data[ncd+1]>>16) & 0xFFFF;
      }
      else{                 // 1 byte was used to store number of words
	 n32 = data[ncd+1] & 0xFF;
	 n16 = (data[ncd+1]>>8) & 0xFF;
	 Bias = short((data[ncd+1]>>16) & 0xFFFF);
      }

//  printf(" n16[%6d] n32[%6d] %o\n", n16, n32, optz);

      
      if (n32 < getLSW(optz) || !n16) {
	 cout <<" unCompress() error: invalid layer number "<< mm+1 <<"\n";
	 return 0;
      }
      mm++;
   }

   if(!n16) return 0;

   kShort = (data[ncd] >> 26) & 0x1F;
    kLong = (data[ncd] >> 21) & 0x1F;
     kBSW = (data[ncd] >> 16) & 0x1F;

   ncd++;
   if(optz & 0x2) ncd++;

   Zero  = (optz & 0x4 ) ? 1<<(kShort-1) : 0;
   Bias  = ((optz & 0x8) && (optz & 0x3)) ? short(data[++ncd] & 0xFFFF) : Bias;
   Scale = (optz & 0x10) ? *((float *)&data[++ncd]) : 1.;

//  cout<<"kShort="<<kShort<<"  kLong="<<kLong<<" kBSW="<<kBSW<<"\n";
//  cout<<"Bias ="<< Bias<<"  Zero="<<Zero<<" ncd="<<ncd<<"\n";

//  if ((optz & 64) != 0) {kLong = 16; kShort = 16;} // no compression
//  if (kLong == 0) kLong = 16;

   if (w.size() != n16) w.resize(n16);

   int *pw = w.data;
   int ns; 
   unsigned int i;
   unsigned int j=0;
   unsigned int bsw=0;

   freebits = 8*sizeof(unsigned int);

   if (!pw) {
      cout <<"WaveRDC:unCompress - memory allocation error.\n";
      return -1;
   }

  w = 0;        // Sazonov 01/29/2001

  ncd++;        // ncd is counter of elements of compressed array "data"
//  printf(" data[0] = %o\n", data[ncd]);

// ***************************
// start unpack block of data
// ***************************

  while (j < n16) {

    Pop(bsw, ncd, kBSW);          // read block service word
    ns = (bsw >> 1);

    if (ns < 0) {
      cout <<" unCompress() error: invalid number of short words "<< ns <<"\n";
      return -1;
    }

//   cout<<"**0** ncd="<<ncd<<"  bsw="<<bsw<<"  j="<<j<<"  fb="<<freebits<<"\n";
//   printf(" data[%6d] = %o\n", ncd, data[ncd]);

    if (kShort > 0) 
       Pop(pw, j, ncd, kShort, ns);
    else 
       for (int i = 0; i < ns; i++) pw[j+i] = 0;

    j += ns;

//   cout<<"**1** ncd="<< ncd<<"  fb="<<freebits<<" j="<<j<<"\n";
    
    if(bsw & 1) { 
       Pop(pw, j, ncd, kLong, 1);
//   cout<<"**2** ncd="<< ncd<<"  fb="<<freebits<<" j="<<j<<" pw="<<pw[j]<<"\n";
    }
    else
       if(j<n16) pw[j] = Zero;

    j++;
  }

// undifferentiate
  if(optz & 0x180){       
     int mean = pw[0];
     pw[0] = Bias;
     for(i=1; i<n16; i++) pw[i] += pw[i-1]+mean;
     if(optz & 0x100) for(i=1; i<n16; i+=2) pw[i] *= -1; // heterodine back
  }

// if was not differentiated
  else{
     w += int(Bias);
  }

  return w.size();
}

 void WaveRDC::Pop(unsigned int &u, int &ncd, int k)
{
// Pop unpacks one element of data. ncd - current position in the array
// freebits - number of unpacked bits in data[ncd]

   int lul = 8*sizeof(u);
   unsigned int mask = (1<<k) - 1;

   if(k == 0) return;

   if (k <= freebits) {
      freebits -= k;
      u = data[ncd] >> freebits;          // shift & save the word in u 
   }
   else {
      u = data[ncd++] << (k - freebits);  // shift upper bits of the word
      freebits += lul - k;                 // expected unpacked space in cd[k+1]
      u |= data[ncd] >> freebits;         // shift & add lower bits in u 
   }
   u = mask & u;                           // mask u
   return;
}

 int WaveRDC::Pop(int *pw, int j, int &ncd, int k, int n)
{
// Pop unpacks "n" elements of data which are stored in compressed
// array "data" as a series of "n" words of k-bit length. Puts
// unpacked data to integer array "pw" and returns number of words decoded

   int jj, x;
   unsigned int u=0;

   if(n == 0 || k == 0) return 0;
   jj = j;

   for (int i = 0; i < n; i++) {
      Pop(u, ncd, k);
      x = (u>>1) + 1;  
      if(!(u & 1)) x = -x;
      if(x == Zero) x = 0;

      pw[jj++] = x;
   }
   return jj-j;
}











