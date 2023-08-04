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
 * File name:   readframe.cc
 * Authors: S.Klimenko, A.Sazonov : University of Florida, Gainesville
 *
 * Virgo Frame Library rev. >=4.41 is required.
 *---------------------------------------------------------------------
*/


#include "readframe.hh"

// Function ReadFrame
// double t  
// *cname - ADC channel name
// *fname - frame file name
// seek - if true then it is allowed to seek data continuation
//        in next file which name the function will try to guess
WSeries<float>* ReadFrame( double t,
			   char *cname, 
			   char *fname, 
			   bool seek)
{
  FrFile *iFile = FrFileINew(fname);

  if(iFile == NULL) 
  {
    printf(" ReadFrame(): cannot open the input file %s\n", fname);
    return NULL;
  }

  int n=0;
  int nu;
  double rate=0.;

  short ref = 0x1234;			// for byte-swapping check
  char *refC = (char *)&ref;

  WSeries<float> *out = NULL;
  WSeries<float> wd;

  char* nname = new char[strlen(fname)];  //computable file name
  strcpy(nname,fname);

  char gpstime[12];
  char newgpstime[12];
  char *ptime;                   // pointer to GPS time substring in file name

  FrAdcData *adc = NULL;

  double gt, gt0, gts, gte;	 // GPS time, GPS time of file end
  gts = FrFileITStart( iFile );
  gte = FrFileITEnd( iFile );

  gt = gts + t;
  gt0 = gt;

  iFile->compress = 255;

  do { 		// repeat for the next file with computed name

// look for the frame containing specified time 'gt' in the next file
     if ( gt < gte )
     {

// gt+1.e-7 in function call to prevent uncertainty in frame access
	adc = FrAdcDataReadT(iFile, cname, gt + 1.e-7);

	if (adc == NULL) {
	   printf(" ReadFrame() error: channel %s is not found in file %s\n",
		  cname, nname);
	   if (out) delete out;
	   return NULL;
	}

//	gt += frlen;

	if( adc->data == NULL ) continue;
	
	n = adc->data->nData;
printf(" got %d data samples\n",n);

	nu = n;
	if ((int)wd.size() != n) wd.resize(n);
	
	if (!out) {
	   rate = adc->sampleRate;

	   if ( n == 0 ) break;		// break the frame reading loop
printf(" creating new WSeries\n");

	   out = new WSeries<float>(n);

	}
  
// do we need to swap the bytes in compressed data?
	bool swap = (refC[0] == 0x34 && adc->data->compress < 256) ||
	            (refC[0] == 0x12 && adc->data->compress > 255);

printf("check for swap\n");
// Swap bytes assuming the vector is used as an int vector
// Note: this code perform very slowly on Alpha CPU unless
// compiler can take advantage of instruction sets of ev56
// or ev6 architecture (option -mcpu=ev56 or ev6 for gcc/g++)
	if(swap) {
	   unsigned char local[2];
	   char *buf = adc->data->data;
	   
	   for(unsigned int i=0; i<adc->data->nBytes-3; i=i+4) {
	      local[0] = buf[3];
	      local[1] = buf[2];
	      buf[3]   = buf[0];
	      buf[2]   = buf[1];
	      buf[1]   = local[1];
	      buf[0]   = local[0];
	      buf    = buf + 4;
	   }
	}
  
printf("check for WAT compress\n");
/*
	if( (adc->data->compress & 0xff) == 255 )
	{   
printf("call WAT unCompress\n");
	   nu = unCompress(adc->data->dataI, wd);

	   if (nu!=n)
           {
	      printf(" ReadFrame: unCompress returned wrong data length\n");
	      break;   			// break the frame reading loop
	   }

// round data for compatibility with uncompressed data stored in frame files
printf("now rounding data\n");

	   switch(adc->data->type) {

	      case FR_VECT_2S:
		 for(int i=0; i<n; i++) {
		    d=wd.data[i];
		    wd.data[i]=float(short(d>0.? d+0.5 : d-0.5));
		 }
		 break;

	      default:
		 break;
	   }
	   out->cpf(wd,0,0,0);
	}
	
	else {
*/
	   switch(adc->data->type) {

	      case FR_VECT_2S:
		 for(int i=0; i<n; i++) 
		    out->data[i] = adc->data->dataS[i];
		 break;

	      case FR_VECT_4R:
		 for(int i=0; i<n; i++) 
		    out->data[i] = adc->data->dataF[i];
		 break;

	      default:;
	   }
//	}

	if (adc) FrAdcDataFree(adc);
	
     } // end of reading frame from the next file
     
     FrFileIEnd(iFile);
     
     if (out) break; 	// break if frame found and data read
     
// try to calculate next file name
     sprintf(gpstime, "%9d", int(gts));
     sprintf(newgpstime, "%9d", int(gte));
     ptime=strstr(nname, gpstime);
     
     if ( ptime != NULL && atoi(ptime)==int(gts) &&
	  strlen(gpstime) == strlen(newgpstime) )
     {
	
	strncpy(ptime,newgpstime,strlen(newgpstime));
//   printf(" guess next file name to be %s\n",nname);

     }
     else  break; 	// break file search loop
     
     iFile = FrFileINew(nname);
     
     if(iFile == NULL) {
	printf(" ReadFrame(): cannot open next input file %s\n", nname);
	break;	// break file search loop
     }
     
     gts=FrFileITStart(iFile);
     iFile->compress = 255;
     
     if (gts!=gte) {
	printf(" ReadFrame(): next input file");
	printf(" %s doesn't provide continuous data\n", nname);
	FrFileIEnd(iFile);
	break; 	// break file search loop
     }
     
     gte=FrFileITEnd(iFile);
     
  } while (seek);  // end of file search loop
  
  if (out == NULL || n == 0) return NULL;

//  delete nname;
  
  out->rate(rate);
  out->start(gt0);

  return out;
}
