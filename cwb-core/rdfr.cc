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

 */
// Function Read transparently reads both WAT-compressed
// and standard-compressed ADC data from single frame file.
// tlen - length of data in seconds,
// tskip - skip time from the begining of the file, in seconds
// *cname - ADC channel name
// *fname - frame file name
// seek - if true then it is allowed to seek data continuation
//        in next file which name the function will try to guess
#include <iostream>
#include "rdfr.hh"
template<class X> wavearray<X>* rdfrm(double tlen, char *cname,  char *fname, double tskip);
wavearray<float>* rdfrmF(double ten, char *cnme,  char *name, double skip);
wavearray<double>* rdfrmD(double len, char *came,  char *fame, double tkip);


template<class X> wavearray<X>* rdfrm(double tlen, char *cname,  char *fname, double tskip)
{
   FrFile *iFile = FrFileINew(fname);
   if(iFile == NULL) 
   {
      printf(" rdfrm(): cannot open the input file %s\n", fname);
      return NULL;
   }
   
   unsigned long int i;
   if (tskip<0) tskip=0.0;
   double ra=0.0,gt0,gts,gte,ntlen=tskip+tlen;
   gts = FrFileITStart(iFile);
   gte = FrFileITEnd(iFile);
   gt0 = gts;		// GPS time of data begin
   if ((gt0 + ntlen) > gte)
   {
      gte=gt0 + ntlen - gte;
      printf("readfrm error:You are trying to access %f seconds more than are available in this file\n",gte);
      return NULL;
   }
   FrVect *adc=NULL;
   adc = FrFileIGetV(iFile, cname, 0.0,ntlen);
   if (adc == NULL || adc->data ==NULL) 
   {
      printf(" ReadFrFile() error: channel %s is not found in file %s\n",  cname, fname);
      return NULL;
   } 
  
   wavearray<X> *out=new wavearray<X>(adc->nData);
   switch(adc->type)
   {
      case FR_VECT_C :    //vector of char 
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->data[i];
	 break;
      case FR_VECT_1U :   //vector of unsigned char
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataU[i];
	 break;
      case FR_VECT_2U :   //vector of unsigned short
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataUS[i];
	 break;
      case FR_VECT_2S :   //vector of short
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataS[i];
	 break;   
      case FR_VECT_4U :   //vector of unsigned int
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataUI[i];
	 break;
      case FR_VECT_4R :   //vector of float
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataF[i];
	 break;
      case FR_VECT_4S :   //vector of int
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataI[i];
	 break;
      case FR_VECT_8U :   //vector of unsigned long
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataUL[i];
	 break;
      case FR_VECT_8R :   //vector of double
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataD[i];
	 break;
      case FR_VECT_8S :   //vector of long
	 for(i=0;i<adc->nData;i++)out->data[i]=adc->dataL[i];
	 break;
      default:
	 cout<<"Can't find the type of FrVect data"<<endl;
	 out=NULL;
   }
   
   ra=double(adc->nData/ntlen);
   FrVectFree(adc);
   FrFileIEnd(iFile);

    
   wavearray<X> *lout=new wavearray<X>(size_t(tlen*ra));
   for(i=0;i<lout->size();i++)lout->data[i]=out->data[i+int(tskip*ra)];
   lout->rate(ra);
   lout->start(gt0);
   return lout;
}


wavearray<float>* rdfrmF(double ten, char *cnme,  char *name, double skip)
{
  wavearray<float> *bib=rdfrm<float>(ten,cnme,name,skip);
  return bib;
}
wavearray<double>* rdfrmD(double len, char *came,  char *fame, double tkip)
{
  wavearray<double> *bb=rdfrm<double>(len,came,fame,tkip);
  return bb;
}


