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


/*---------------------------------------------------------------------
 * Package:     Wavelet Analysis Tool
 * File name:   readfrfile.cc
 * Authors: S.Klimenko, A.Sazonov : University of Florida, Gainesville
 *
 * Virgo Frame Library rev. >= 4.41 is required.
 *---------------------------------------------------------------------
*/

#include "readfrfile.hh"

// Function ReadFrFile transparently reads both WAT-compressed
// and standard-compressed ADC data from single frame file.
// wavearray<double> - output array,
// tlen - length of data in seconds,
// tskip - skip time from the begining of the file, in seconds
// *cname - ADC channel name
// *fname - frame file name
// seek - if true then it is allowed to seek data continuation
//        in next file which name the function will try to guess
bool ReadFrFile(wavearray<double> &out,
	   double tlen, 
	   double tskip, 
	   char *cname, 
	   char *fname, 
	   bool seek,
	   char *channel_type)
{
  FrFile *iFile = FrFileINew(fname);
  if(iFile == NULL) 
  {
    printf(" ReadFrFile(): cannot open the input file %s\n", fname);
    return false;
  }
  if(&out == NULL) 
  {
    printf(" ReadFrFile(): no output array specified \n");
    return false;
  }

  bool failure = false;			// true if channel not found
  int n=0, l=0, k=0;
  int nu, nn=0;
  double rate=0., d;

  short ref = 0x1234;			// for byte-swapping check
  char *refC = (char *)&ref;

  int iskip = 0;			// skip in index space
  if (tskip<0) tskip=0.;

  wavearray<float> wd;

  char nname[512];                       // char array for computable file name
  sprintf(nname,"%s",fname);

//  printf("%s  %s\n",nname,fname);

  char gpstime[12];
  char newgpstime[12];
  char *ptime;                   // pointer to GPS time substring in file name

  FrAdcData  *adc   = NULL;
  FrProcData *proc  = NULL;
  FrSerData  *ser   = NULL;
  FrSimData  *sim   = NULL;
  FrVect     *datap = NULL;

// frame length (should be determined from frame file )
  double frlen = 16.;

// this doesn't work
//  frlen = iFile->length;  // frame length
//  cout<<frlen<<endl;

  double gt,gt0,gts,gte;	 // GPS time, GPS time of file end
  gts = FrFileITStart(iFile);
  gte = FrFileITEnd(iFile);

// nskip is number of full frames (or multiframe files) in tskip
  int nskip = int( tskip/frlen );

// fskip is a fraction of tskip: 0 <= fskip < frlen
  double fskip = tskip - nskip*frlen;

  gt = gts + nskip*frlen; 	// begin GPS time of the first frame to read
  gt0 = gts + tskip;		// GPS time where data begins

//printf(" gts = %16.6f, gte = %16.6f, gt = %16.6f\n", gts, gte, gt);

  iFile->compress = 255;

  do {			// read many files if names are computable

     while(gt<gte) { 	// read frames from a file

//printf(" gts = %16.6f, gte = %16.6f, gt = %16.6f\n", gts, gte, gt);

// gt+1.e-7 in function call to prevent uncertainty in frame access

        failure = false;

        if (!strcasecmp(channel_type,"adc"))
        {
	  adc = FrAdcDataReadT(iFile,cname,gt+1.e-7);
          if (adc == NULL) { failure = true; }
          else { datap = adc->data; }
        }
        else if (!strcasecmp(channel_type,"proc"))
        {
          proc = FrProcDataReadT(iFile,cname,gt+1.e-7);
          if (proc == NULL) { failure = true; } 
          else { datap = proc->data; }
        }
        else
        {
          printf(" ReadFrFile() error: unknown channel type %s\n",
                   channel_type);
        }

	if (failure) {
	   printf(" ReadFrFile() error: channel %s is not found in file %s\n",
		  cname, nname);
	   return false;
	}

	gt += frlen;

	if(datap == NULL ) continue;
	
//        printf("startx=%f%s\n",datap->startX[0],datap->unitX[0]);

	n = datap->nData;
	nu = n;
	if ((int)wd.size() != n) wd.resize(n);
	
	if (!nn) {
//	   rate = adc->sampleRate;		// not available for proc
           if (!datap->dx[0]==0.) rate = 1./datap->dx[0];
	   nn = int(tlen * rate);

           if ( nn == 0 )			// 
	   {
              printf(" ReadFrFile: time interval too short, return no data.\n");
	      return false;
	   }

	   if(out.size() != size_t(nn)) out.resize(nn);

// k counts aready done data samples
// l counts how much to do in current frame
// set initial values
	   k = 0;
	   l = nn;
	}
  
	iskip = int(fskip * rate);
	l = l < (n - iskip)? l : n - iskip;

// do we need to swap the bytes in compressed data?
	bool swap = (refC[0] == 0x34 && datap->compress == 255) ||
	            (refC[0] == 0x12 && datap->compress == 511);

// Swap bytes assuming the vector is used as an int vector
// Note: this code perform very slowly on Alpha CPU unless
// compiler can take advantage of instruction sets of ev56
// or ev6 architecture (option -mcpu=ev56 or ev6 for gcc/g++)
	if(swap) {
	   unsigned char local[2];
	   char *buf = datap->data;
	   
	   for(unsigned int i=0; i<datap->nBytes-3; i=i+4) {
	      local[0] = buf[3];
	      local[1] = buf[2];
	      buf[3]   = buf[0];
	      buf[2]   = buf[1];
	      buf[1]   = local[1];
	      buf[0]   = local[0];
	      buf    = buf + 4;
	   }
	}
  
	if( (datap->compress & 0xff) == 255 ){
	   
	   nu = unCompress(datap->dataI, wd);

	   if (nu!=n) {
	      printf(" ReadFrFile: unCompress returned wrong data length\n");
	      break;   			// break the frame reading loop
	   }
	   
// round data for compatibility with uncompressed data stored in frame files
	   switch(datap->type) {

	      case FR_VECT_2S:
		 for(int i=iskip; i<n; i++) {
		    d=wd.data[i];
		    wd.data[i]=float(short(d>0.? d+0.5 : d-0.5));
		 }
		 break;

	      default:
		 break;
	   }
	   for(int i=iskip; i<l; i++)  
	      out.data[k+i]=double(wd.data[i]);
	}
	
	else {

	   switch(datap->type) {

	      case FR_VECT_2S:
		 for(int i=0; i<l; i++) 
		    out.data[i+k] = double(datap->dataS[i+iskip]);
		 break;

	      case FR_VECT_4R:
		 for(int i=0; i<l; i++) 
		    out.data[i+k] = double(datap->dataF[i+iskip]);
		 break;

	      default:;
	   }
	}
	
	if (adc) FrAdcDataFree(adc);
        if (proc) FrProcDataFree(proc);
        if (sim) FrSimDataFree(sim);
        if (ser) FrSerDataFree(ser);
	
	k += l;
	l = nn - k; // how much to read?
	fskip = 0.; // fskip applyes to first frame only

	if ( l <= 0 ) break;           	// break the frame reading loop

     } // end of frames reading loop
     
     FrFileIEnd(iFile);
     
     if (nn && ((nn-k)<=0)) break; 	// no more data to read
     
// try to calculate next file name
     sprintf(gpstime, "%9d", int(gts));
     sprintf(newgpstime, "%9d", int(gte));
     ptime=strstr(nname, gpstime);
     
     if ( ptime != NULL && atoi(ptime) == int(gts) &&
	  strlen(gpstime) == strlen(newgpstime) ){
	
	strncpy(ptime, newgpstime, strlen( newgpstime ));
//   printf(" guess next file name to be %s\n",nname);

     }
     else  break; 		// break file search loop
     
     iFile = FrFileINew(nname);
     
     if(iFile == NULL) {
	printf(" ReadFrFile(): cannot open next input file %s\n", nname);
	break;			// break file search loop
     }
     
     gts = FrFileITStart(iFile);
     iFile->compress = 255;
     
     if (gts != gte) {
	printf(" ReadFrFile(): next input file");
	printf(" %s doesn't provide continuous data\n", nname);
	FrFileIEnd(iFile);
	break; 			// break file search loop
     }
     
     gte = FrFileITEnd(iFile);
     
  } while (seek);  		// end of file search loop
  
  if (out.size() != size_t(nn)) return false;
  
  if (( nn - k ) > 0)  		// fill rest of data by zeroes
  {
     for (int i=k; i<nn; i++) out.data[i]=0.;
  }

//  delete nname;
//  if (out == 0) return NULL;

// normal return
  out.rate(rate);
  out.start(gt0);

  return true;
}


// Function ReadFrFile transparently reads both WAT-compressed
// and standard-compressed ADC data from single frame file.
// tlen - length of data in seconds,
// tskip - skip time from the begining of the file, in seconds
// *cname - ADC channel name
// *fname - frame file name
// seek - if true then it is allowed to seek data continuation
//        in next file which name the function will try to guess
wavearray<float>* ReadFrFile(double tlen, 
			     double tskip, 
			     char *cname, 
			     char *fname, 
			     bool seek,
			     char *channel_type)
{
  FrFile *iFile = FrFileINew(fname);
  if(iFile == NULL) 
  {
    printf(" ReadFrFile(): cannot open the input file %s\n", fname);
    return NULL;
  }

  bool failure = false;			// true if channel not found
  int n=0, l=0, k=0;
  int nu, nn=0;
  double rate=0., d;

  short ref = 0x1234;			// for byte-swapping check
  char *refC = (char *)&ref;

  int iskip = 0;			// skip in index space
  if (tskip<0) tskip=0.;

  wavearray<float> *out = NULL;
  wavearray<float> wd;

  char* nname = new char[strlen(fname)]; // char array for computable file name
  strcpy(nname,fname);             	 // copy original file name

  char gpstime[12];
  char newgpstime[12];
  char *ptime;                   // pointer to GPS time substring in file name

  FrAdcData *adc = NULL;
  FrProcData *proc = NULL;
  FrSerData *ser = NULL;
  FrSimData *sim = NULL;
  FrVect *datap = NULL;

// frame length (should be determined from frame file )
  double frlen = 16.;

// this doesn't work
//  frlen = iFile->length;  // frame length
//  cout<<frlen<<endl;

  double gt,gt0,gts,gte;	 // GPS time, GPS time of file end
  gts = FrFileITStart(iFile);
  gte = FrFileITEnd(iFile);

// nskip is number of full frames (or multiframe files) in tskip
  int nskip = int( tskip/frlen );

// fskip is a fraction of tskip: 0 <= fskip < frlen
  double fskip = tskip - nskip*frlen;

  gt = gts + nskip*frlen; 	// begin GPS time of the first frame to read
  gt0 = gts + tskip;		// GPS time of data begin

//printf(" gts = %16.6f, gte = %16.6f, gt = %16.6f\n", gts, gte, gt);

  iFile->compress = 255;

  do {			// read many files if names are computable

     while(gt<gte) { 	// read frames from file

//printf(" gts = %16.6f, gte = %16.6f, gt = %16.6f\n", gts, gte, gt);

// gt+1.e-7 in function call to prevent uncertainty in frame access

        failure = false;

        if (!strcasecmp(channel_type,"adc"))
        {
	  adc = FrAdcDataReadT(iFile,cname,gt+1.e-7);
          if (adc == NULL) { failure = true; }
          else { datap = adc->data; }
        }
        else if (!strcasecmp(channel_type,"proc"))
        {
          proc = FrProcDataReadT(iFile,cname,gt+1.e-7);
          if (proc == NULL) { failure = true; } 
          else { datap = proc->data; }
        }
        else
        {
          printf(" ReadFrFile() error: unknown channel type %s\n",
                   channel_type);
        }

	if (failure) {
	   printf(" ReadFrFile() error: channel %s is not found in file %s\n",
		  cname, nname);
	   if (out) delete out;
	   return NULL;
	}

	gt += frlen;

	if(datap == NULL ) continue;
	
//        printf("startx=%f%s\n",datap->startX[0],datap->unitX[0]);
	n = datap->nData;
	nu = n;
	if ((int)wd.size() != n) wd.resize(n);
	
	if (!out) {
//	   rate = adc->sampleRate;		// not available for proc
           if (!datap->dx[0]==0.) rate = 1./datap->dx[0];
	   nn = int(tlen * rate);

           if ( nn == 0 )			// 
	   {
              printf(" ReadFrFile: time interval too short, return no data.\n");
	      return NULL;
	   }

	   out = new wavearray<float>(nn);

// k counts aready done data samples
// l counts how much to do in current frame
// set initial values
	   k = 0;
	   l = nn;
	}
  
	iskip = int(fskip * rate);
	l = l < (n - iskip)? l : n - iskip;

// do we need to swap the bytes in compressed data?
	bool swap = (refC[0] == 0x34 && datap->compress == 255) ||
	            (refC[0] == 0x12 && datap->compress == 511);

// Swap bytes assuming the vector is used as an int vector
// Note: this code perform very slowly on Alpha CPU unless
// compiler can take advantage of instruction sets of ev56
// or ev6 architecture (option -mcpu=ev56 or ev6 for gcc/g++)
	if(swap) {
	   unsigned char local[2];
	   char *buf = datap->data;
	   
	   for(unsigned int i=0; i<datap->nBytes-3; i=i+4) {
	      local[0] = buf[3];
	      local[1] = buf[2];
	      buf[3]   = buf[0];
	      buf[2]   = buf[1];
	      buf[1]   = local[1];
	      buf[0]   = local[0];
	      buf    = buf + 4;
	   }
	}
  
	if( (datap->compress & 0xff) == 255 ){
	   
	   nu = unCompress(datap->dataI, wd);

	   if (nu!=n) {
	      printf(" ReadFrFile: unCompress returned wrong data length\n");
	      break;   			// break the frame reading loop
	   }
	   
// round data for compatibility with uncompressed data stored in frame files
	   switch(datap->type) {

	      case FR_VECT_2S:
		 for(int i=iskip; i<n; i++) {
		    d=wd.data[i];
		    wd.data[i]=float(short(d>0.? d+0.5 : d-0.5));
		 }
		 break;

	      default:
		 break;
	   }
	   out->cpf(wd,l,iskip,k);
	}
	
	else {

	   switch(datap->type) {

	      case FR_VECT_2S:
		 for(int i=0; i<l; i++) 
		    out->data[i+k] = datap->dataS[i+iskip];
		 break;

	      case FR_VECT_4R:
		 for(int i=0; i<l; i++) 
		    out->data[i+k] = datap->dataF[i+iskip];
		 break;

	      default:;
	   }
	}
	
	if (adc) FrAdcDataFree(adc);
        if (proc) FrProcDataFree(proc);
        if (sim) FrSimDataFree(sim);
        if (ser) FrSerDataFree(ser);
	
	k += l;
	l = nn - k; // how much to read?
	fskip = 0.; // fskip applyes to first frame only

	if ( l <= 0 ) break;           	// break the frame reading loop
     } // end of frames reading loop
     
     FrFileIEnd(iFile);
     
     if (out && ((nn-k)<=0)) break; 	// no more data to read
     
// try to calculate next file name
     sprintf(gpstime, "%9d", int(gts));
     sprintf(newgpstime, "%9d", int(gte));
     ptime=strstr(nname, gpstime);
     
     if ( ptime != NULL && atoi(ptime) == int(gts) &&
	  strlen(gpstime) == strlen(newgpstime) ){
	
	strncpy(ptime, newgpstime, strlen( newgpstime ));
//   printf(" guess next file name to be %s\n",nname);

     }
     else  break; 		// break file search loop
     
     iFile = FrFileINew(nname);
     
     if(iFile == NULL) {
	printf(" ReadFrFile(): cannot open next input file %s\n", nname);
	break;			// break file search loop
     }
     
     gts = FrFileITStart(iFile);
     iFile->compress = 255;
     
     if (gts != gte) {
	printf(" ReadFrFile(): next input file");
	printf(" %s doesn't provide continuous data\n", nname);
	FrFileIEnd(iFile);
	break; 			// break file search loop
     }
     
     gte = FrFileITEnd(iFile);
     
  } while (seek);  		// end of file search loop
  
  if (out==NULL) return NULL;
  
  if (( nn - k ) > 0)  		// fill rest of data by zeroes
  {
     for (int i=k; i<nn; i++) out->data[i]=0.;
  }

//  delete nname;
  
  if (out == 0) return NULL;

// normal return
  out->rate(rate);
  out->start(gt0);

  return out;
}

