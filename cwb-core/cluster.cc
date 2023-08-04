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


//---------------------------------------------------
// WAT cluster class
// S. Klimenko, University of Florida
//---------------------------------------------------

#define WAVECLUSTER_CC
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cluster.hh"

using namespace std;

// sort wavepixel objects on time
int compare_pix(const void *x, const void *y){   
   wavepixel* p = *((wavepixel**)x);
   wavepixel* q = *((wavepixel**)y);
   double a = (p->time+0.5)/p->rate - (q->time+0.5)/q->rate;
   if(a > 0) return 1;
   if(a < 0) return -1;
   return 0;
}

// constructors

wavecluster::wavecluster()
{
  pList.clear();
  sCuts.clear();
  cList.clear();
  cRate.clear();
  start = 0.;
  stop  = 0.;
  shift = 0.;
  bpp   = 0.;
  low   = 0.;
  high  = 0.;
  ifo   = 0;
  run   = 0;
}

wavecluster::wavecluster(const wavecluster& value)
{
   *this = value;
}


wavecluster::wavecluster(WSeries<double>& w, bool halo)
{
   init(w,halo);
}

// destructor

wavecluster::~wavecluster(){}

//: operator =

wavecluster& wavecluster::operator=(const wavecluster& value)
{
   pList.clear();
   sCuts.clear();
   cList.clear();
   cRate.clear();
   pList = value.pList;
   sCuts = value.sCuts;
   cList = value.cList;
   cRate = value.cRate;
   nRMS  = value.nRMS;
   start = value.start;
   stop  = value.stop;
   shift = value.shift;
   bpp   = value.bpp;
   low   = value.low;
   high  = value.high;
   ifo   = value.ifo;
   run   = value.run;
   return *this;
}



//**************************************************************************
// initialize wavecluster from WSeries (binary tree)
//**************************************************************************
size_t wavecluster::init(WSeries<double>& w, bool halo)
{
   pList.clear();
   sCuts.clear();
   cList.clear();
   cRate.clear();
   bpp=0.; start=0.; stop=0., ifo=0; shift=0.;
   
   if(!w.pWavelet->BinaryTree()) return 0;

   start = w.start();                 // set start time
   stop  = start+w.size()/w.wavearray<double>::rate();   // set stop time (end of time interval)
   bpp   = w.getbpp();
   low   = w.getlow();
   high  = w.gethigh();

   size_t i,j,k;

   size_t ni = w.maxLayer()+1;
   size_t nj = w.size()/ni;
   size_t n  = ni-1;
   size_t m  = nj-1;
   size_t nl = size_t(2.*low*ni/w.wavearray<double>::rate());        // low frequency boundary index
   size_t nh = size_t(2.*high*ni/w.wavearray<double>::rate());       // high frequency boundary index

   if(nh>=ni) nh = ni-1;

   int L;
   int* TF = new int[ni*nj];
   int* p = NULL; 
   int* q = NULL; 

   wavearray<double> a;

   k = 0;
   for(i=0; i<ni; i++){
      p  = TF+i*nj; 
      w.getLayer(a,i);	

      if(nj!=a.size()) 
	 cout<<"wavecluster::constructor() size error: "<<nj<<endl;

      for(j=0; j<nj; j++){ 
	 if(a.data[j]==0. || i<nl || i>nh) p[j]=0;  
	 else {p[j]=-1; k++;}
      }
   }
   if(!k) { delete [] TF; return 0; }

/**************************************/
/* fill in the pixel list             */
/**************************************/

   wavepixel pix;
   pix.amplitude.clear();                 // clear link pixels amplitudes 
   pix.neighbors.clear();                 // clear link vector for neighbors 
   pix.clusterID = 0;                     // initialize cluster ID
   L = 1<<w.getLevel();                   // number of layers
   pix.rate = float(w.wavearray<double>::rate()/L);          // pixel rate

   size_t &f = pix.frequency;
   size_t &t = pix.time;
   size_t F,T;

   L = 0;

   for(i=0; i<ni; i++) 
   {
      p  = TF + i*nj;

      for(j=0; j<nj; j++) 
      {
         if(p[j] != -1) continue;

	 t = j; f = i;                // time index; frequency index;
	 pix.core = true;             // core pixel
	 pList.push_back(pix);        // save pixel
	 p[j] = ++L;                  // save pixel index (Fortran indexing)
	 
	 if(!halo) continue;

// include halo

	 pix.core = false;            // halo pixel
	 
	 F = i<n ? i+1 : i; 
	 T = j<m ? j+1 : j; 
	 
	 for(f = i>0 ? i-1 : i; f<=F; f++) {
	    q = TF + f*nj;
	    for(t = j>0 ? j-1 : j; t<=T; t++) {
	       if(!q[t]) {pList.push_back(pix); q[t] = ++L;}
	    }
	 }
      }             
   }

// set amplitude and neighbours

   std::vector<int>* pN;
   size_t nL =  pList.size();
   slice S;

   for(k=0; k<nL; k++)
   {
      i = pList[k].frequency;
      j = pList[k].time;
      if(int(i)>w.maxLayer()) cout<<"wavecluster::constructor() maxLayer error: "<<i<<endl;
      S = w.getSlice(i);                

// set pixel amplitude

      pList[k].amplitude.push_back(w.data[S.start()+S.stride()*j]);

// set neighbors

      pN = &(pList[k].neighbors);
	 
      F = i<n ? i+1 : i; 
      T = j<m ? j+1 : j; 
	 
      for(f = i>0 ? i-1 : i; f<=F; f++) {
	 q = TF + f*nj;
	 for(t = j>0 ? j-1 : j; t<=T; t++) {
	    L =  (f==i && t==j) ? 0 : q[t];
	    if(L) pN->push_back(L-1);
	 }
      }
   }

   delete [] TF;
   return pList.size();
}


//**************************************************************************
// cluster analysis
//**************************************************************************
size_t wavecluster::cluster()
{
   size_t volume;
   size_t i,m;
   vector<int> refMask;
   size_t nCluster = 0;
   size_t n = pList.size();

   if(!pList.size()) return 0;

   cList.clear();
   sCuts.clear();
   cRate.clear();

   for(i=0; i<n; i++){
      if(pList[i].clusterID) continue;
      pList[i].clusterID = ++nCluster;
      volume = cluster(&pList[i]);
      refMask.clear();
      cRate.push_back(refMask);
      refMask.resize(volume);
      cList.push_back(refMask);
      sCuts.push_back(false);
   }

   list<vector_int>::iterator it;
   nCluster = 0;
   if(!cList.size()) return 0;

   for(it=cList.begin(); it != cList.end(); it++) {
      nCluster++;

      m = 0;
      for(i=0; i<n; i++) { 
	if(pList[i].clusterID == nCluster) (*it)[m++] = i;
      }

      if(it->size() != m) { 
	 cout<<"cluster::cluster() size mismatch error: ";
	 cout<<m<<" size="<<it->size()<<" "<<nCluster<<endl;
      }
      if(m==1 && !pList[(*it)[0]].core) { 
	 cout<<"cluster::cluster() : empty cluster. \n";
	 cout<<pList[(*it)[0]].time<<" "<<pList[(*it)[0]].frequency<<endl;
      }
   }
   return nCluster;
}

size_t wavecluster::cluster(wavepixel* p)
{
   size_t volume = 1;
   int i = p->neighbors.size();
   wavepixel* q;

   while(--i >= 0){
      q = &pList[p->neighbors[i]];
      if(!q->clusterID){
	  q->clusterID = p->clusterID;
	  volume += cluster(q);
      }
   }
   return volume;
}


//**************************************************************************
// remove halo pixels from pixel list
//**************************************************************************
size_t wavecluster::cleanhalo(bool keepid)
{
   if(!pList.size() || !cList.size()) return 0;

   size_t i;
   size_t cid = 0;              // new cluster ID
   size_t pid = 0;              // new pixel ID

   wavepixel* pix = NULL;
   list<vector_int>::iterator it;
   std::vector<int> id;

   wavecluster x(*this);

   pList.clear();
   sCuts.clear();
   cList.clear();

   for(it=x.cList.begin(); it != x.cList.end(); it++) {  // loop over clusters

      pix = &(x.pList[((*it)[0])]);
      if(x.sCuts[pix->clusterID-1]) continue;   // apply selection cuts

      cid++;
      id.clear();
      for(i=0; i<it->size(); i++) {         // loop over pixels in the cluster
	 pix = &(x.pList[((*it)[i])]);
	 if(pix->core) {
	    pix->clusterID = keepid ? cid : 0;
	    pix->neighbors.clear();
	    id.push_back(pid++);
	    pList.push_back(*pix);          // fill pixel list
	 }
      }

      i = id.size();
      if(!i) cout<<"wavecluster::cleanhalo() error: empty cluster.";
      
      if(keepid) { 
	 cList.push_back(id);               // fill cluster list
	 sCuts.push_back(false);            // fill selection cuts
      }

      if(i<2) continue;
      
      while(--i > 0) {                     // set neighbors
	 pList[id[i-1]].neighbors.push_back(id[i]);
	 pList[id[i]].neighbors.push_back(id[i-1]);
      }

      
   }
   return pList.size();
}

//**************************************************************************
// save wavelet amplitudes in cluster structure
//**************************************************************************
size_t wavecluster::apush(WSeries<double>& w, double offset)
{
   size_t j,k,m;
   slice S;
   size_t N = w.size()-1;
   size_t M = pList.size();
   size_t max_layer = w.maxLayer();
   wavepixel* p = NULL;                 // pointer to pixel structure
   double a;
   float rate;
   size_t ofFSet;  

   if(!M) return 0;

   offset = fabs(offset);
   if(fabs(w.start()+offset-start)>1.e-12) {
     printf("wavecluster::apush: start time mismatch: dT=%16.13f",start-w.start());
     return 0;
   }

   for(k=0; k<M; k++){
      p = &(pList[k]);

      if(p->frequency > max_layer) {
	 p->amplitude.push_back(0.); 
	 continue;
      }

      S = w.getSlice(p->frequency);
      m = S.stride();

      rate = w.wavearray<double>::rate()/m;                          // rate at this layer
      ofFSet = size_t(offset*w.wavearray<double>::rate()+0.5);       // number of offset samples

      if(int(p->rate+0.1) != int(rate+0.1)) {
	 p->amplitude.push_back(0.); 
	 continue;
      }

      if((ofFSet/m)*m != ofFSet) 
	 cout<<"wavecluster::apush(): illegal offset "<<ofFSet<<" m="<<m<<"\n";

      j = S.start()+ofFSet+S.stride()*p->time;     // pixel index in cluster
      a = j>N ? 0. : w.data[j]; 
      p->amplitude.push_back(a);
   }

   return pList[0].amplitude.size();
}

//**************************************************************************
// append input cluster list
//**************************************************************************
size_t wavecluster::append(wavecluster& w)
{
   size_t i,k,n;
   size_t in = w.pList.size();
   size_t on = pList.size();
   size_t im = w.cList.size();
   size_t om = cList.size();

   if(!in) { return on; }
   if(!on) { *this = w; return in; }

   wavepixel* p = NULL;                  // pointer to pixel structure
   std::vector<int>* v;

   if((w.start!=start) || (w.ifo!=ifo) || (w.shift!=shift)) {
     printf("\n wavecluster::append(): cluster type mismatch");
     printf("%f / %f, %f / %f, %d / %d\n",w.start,start,w.shift,shift,w.ifo,ifo);

     return on;
   }

   if(im && !om) {                      // clear in clusters
      w.sCuts.clear(); w.cList.clear(); im=0; 
      for(i=0; i<in; i++) w.pList[i].clusterID=0;
   }
   if(!im && om) {                      // clear out clusters
      sCuts.clear(); cList.clear(); om=0; 
      for(i=0; i<on; i++) pList[i].clusterID=0;
   }


   for(i=0; i<in; i++){
      p = &(w.pList[i]);
      v = &(p->neighbors);
      n = v->size();
      for(k=0; k<n; k++) (*v)[k] += on;  // new neighbors pointers
      p->clusterID += om;                // update cluster ID
      pList.push_back(*p);
   }

   if(!im) return pList.size();

   list<vector_int>::iterator it;
   n = 0;

   for(it=w.cList.begin(); it != w.cList.end(); it++) {

      for(i=0; i<it->size(); i++) {      // loop over pixels in the cluster
	(*it)[i] += on;                  // update pixel index
      }

      cList.push_back(*it);
      sCuts.push_back(w.sCuts[n++]);
   }

   return pList.size();
}

//**************************************************************************
// merge clusters in the lists
//**************************************************************************
size_t wavecluster::merge(double S)
{
   size_t i,j,k,m;
   size_t n = pList.size();
   int l;
   
   if(!n) return 0;

   wavepixel* p = NULL;                 // pointer to pixel structure
   wavepixel* q = NULL;                 // pointer to pixel structure
   std::vector<int>* v;
   float eps;
   double E;
   bool insert;
   double ptime, pfreq;
   double qtime, qfreq;

   cRate.clear();
   wavepixel** pp = (wavepixel**)malloc(n*sizeof(wavepixel*));

// sort pixels

   for(i=0; i<n; i++) { pp[i] = &(pList[i]); pp[i]->index = i; } 
   qsort(pp, n, sizeof(wavepixel*), &compare_pix);         // sorted time
   

// update neighbors

   for(i=0; i<n; i++) {
      p = pp[i];
      if(!p->core) continue;
      ptime = (p->time+0.5)/p->rate;  
      pfreq = (p->frequency+0.5)*p->rate;
      
      for(j=i+1; j<n; j++){
	 q = pp[j];

	 eps = 0.55/p->rate + 0.55/q->rate;
	 qtime = (q->time+0.5)/q->rate;

	 if(qtime<ptime) cout<<"wavecluster::merge() error"<<endl;

	 if(qtime-ptime > 1.) break;
	 if(qtime-ptime > eps) continue;
	 if(!q->core || p->rate==q->rate) continue;
	 if(!(p->rate==2*q->rate || q->rate==2*p->rate)) continue;

	 eps = 0.55*(p->rate+q->rate);
	 qfreq = (q->frequency+0.5)*q->rate;
	 if(fabs(pfreq-qfreq) > eps) continue;

// insert in p

	 l = q->index;
	 insert = true;
	 v = &(p->neighbors);
	 m = v->size();
	 for(k=0; k<m; k++) {
	    if((*v)[k] == l) {insert=false; break;} 
	 }
	 if(insert) v->push_back(l); 

// insert in q

	 l = p->index;
	 insert = true;
	 v = &(q->neighbors);
	 m = v->size();
	 for(k=0; k<m; k++) {
	    if((*v)[k] == l) {insert=false; break;} 
	 }
	 if(insert) v->push_back(l); 

      }
   }
   free(pp);

//***************
   cluster();
//***************

   std::list<vector_int>::iterator it;
   wavepixel* pix = NULL;
   std::vector<int> rate;
   std::vector<int> temp;
   std::vector<int> sIZe;
   std::vector<bool> cuts;
   std::vector<double> ampl;
   std::vector<double> amax;
   std::vector<double> sigf;
    std::vector<double>* pa;
   double a;
   bool  cut;
   size_t ID;
   size_t max = 0;
   size_t min = 0;
   size_t count=0;

   for(it=cList.begin(); it != cList.end(); it++) {
      k = it->size();
      if(!k) cout<<"wavecluster::merge() error: empty cluster.\n";

// fill cluster statistics

      m = 0; E = 0;
      rate.clear();
      ampl.clear();
      amax.clear();
      sigf.clear();
      cuts.clear();
      sIZe.clear();
      temp.clear();

      ID = pList[((*it)[0])].clusterID;

      for(i=0; i<k; i++) {            
	 pix = &(pList[((*it)[i])]);
	 if(!pix->core) continue;
	 pa = &(pix->amplitude);
	 a = pa->size()>1 ? pow((*pa)[1],2) : (*pa)[0];

	 insert = true;
	 for(j=0; j<rate.size(); j++) {
	    if(rate[j] == int(pix->rate+0.1)) {
	       insert=false;
	       ampl[j] += a;
	       sIZe[j] += 1;
	       sigf[j] += (*pa)[0];
	       if(a>amax[j]) amax[j] = a;
	    }	       
	 }

	 if(insert) {
	    rate.push_back(int(pix->rate+0.1));
	    ampl.push_back(a);
	    amax.push_back(a);
	    sIZe.push_back(1);
	    cuts.push_back(true);
	    sigf.push_back((*pa)[0]);
	 }

	 m++; E += (*pa)[0];

	 if(ID != pix->clusterID) 
	    cout<<"wavecluster::merge() error: cluster ID mismatch.\n";
      }

// cut off single level clusters

      if(!rate.size()) { cout<<"k="<<k<<" id="<<ID<<endl; continue; }
      if(rate.size()<2 || m<=2){ sCuts[ID-1] = true; continue; }
//      sCuts[ID-1] = (gammaCL(E,m) > S-log(m-1.)) ? false : true;
      sCuts[ID-1] = (gammaCL(E,m) > S) ? false : true;

// coincidence between levels
      cut = true;
      for(i=0; i<rate.size(); i++) {  
	 for(j=0; j<rate.size(); j++) {
	    if(rate[i]/2==rate[j] || rate[j]/2==rate[i]) {
	       cuts[i] = cuts[j] = cut = false;	       
	    }
	 }
      }
      if(cut || sCuts[ID-1]) { sCuts[ID-1] = true; continue; }

// select optimal resolution

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {  // select max excess power
	 if(ampl[j]-sIZe[j]>a && !cuts[j]) {max=j; a=ampl[j]-sIZe[j];}
      }

      a = -1.e99;
      for(j=0; j<rate.size(); j++) {
	 if(max==j) continue;
	 if(ampl[j]-sIZe[j]>a && !cuts[j]) {min=j; a=ampl[j]-sIZe[j];}
      }

      temp.push_back(rate[max]);
      temp.push_back(rate[min]);
      cRate[ID-1] = temp;
      count++;
   
   }

   return count;
}


//**************************************************************************
// coincidence between two clusters lists
//**************************************************************************
size_t wavecluster::coincidence(wavecluster& w, double T)
{
   size_t i,j,k;
   size_t ik = w.asize();
   size_t ok = asize();

   if(!ik || !ok) return 0;
   
   k = ok>1 ? ik : 1;
   k =  k>1 ?  2 : 1;
 
   wavearray<float>  tin  = w.get((char*)"time",k);  // get in time
   wavearray<float>  tou  =   get((char*)"time",k);  // get out time
   wavearray<float>  rin  = w.get((char*)"rate",0);  // get in rate
   wavearray<float>  rou  =   get((char*)"rate",0);  // get out rate
   wavearray<float>  cid =    get((char*)"ID",0);    // get out cluster ID

   size_t in = tin.size();
   size_t on = tou.size();
   double window;
   bool cut;

   k = 0;

   for(i=0; i<on; i++){
      cut = true;
      for(j=0; j<in; j++){
	 window = 0.5/rou[i]+0.5/rin[j];
	 if(window<T) window = T; 
	 if(fabs(tou.data[i]-tin.data[j])<window) { cut=false; break; }
      }
      if(cut) sCuts[int(cid[i]-0.5)] = true;
      else k++;
   }
   return k;
}

//**************************************************************************
// save noise rms in  amplitude array in cluster structure
// input fl is used for low pass filter correction
//**************************************************************************
void wavecluster::setrms(WSeries<double>& w, double fl, double fh)
{
   size_t i,j,n,m;
   slice S;
   size_t M = pList.size();
   size_t max_layer = w.maxLayer()+1;
   wavepixel* p = NULL;                 // pointer to pixel structure

   int k;
   int    wsize  = w.size()/max_layer;
   double wstart = w.start();
   double wrate  = w.wavearray<double>::rate();
   double deltaF = w.gethigh()/max_layer;
   double x,f,t,r;
   bool   C;       // low pass filter correction flag

   if(fl<0.) fl = low;
   if(fh<0.) fh = w.gethigh();

   if(!M || !w.size()) return;

   for(i=0; i<M; i++){
      p = &(pList[i]);

      if(p->frequency >= max_layer) continue;

      f = p->frequency*p->rate/2.;
      C = f<fl ? true : false; 
      f = C ? fl : f;
      n = size_t(f/deltaF); // first layer in w
      f = (p->frequency+1)*p->rate/2.;
      m = size_t(f/deltaF); // last layer in w
      t = start+(p->time+0.5)/p->rate;
      k = int((t-wstart)*wrate);     // time index in noise array

      if(k>=wsize) k -= k ? 1 : 0; 
      if(k<0 || n>=m || k>=wsize) {
	 cout<<"wavecluster::setrms() - invalid input\n";
	 continue;
      }

      r = 0.;
      for(j=n; j<m; j++) {         // get noise rms for specified pixel
	 S = w.getSlice(j);
	 x = w.data[S.start()+k*S.stride()];
	 r += 1./x/x;
	 //	 r += C && (j<2*n) ? 2./x/x : 1./x/x;
      }
      r /= double(m)-double(n);
      p->noiserms = sqrt(1./r); 

   }
   return;
}

//**************************************************************************
// save noise variance in  amplitude array in cluster structure
//**************************************************************************
void wavecluster::setvar(wavearray<float>& w, double fl, double fh)
{
   size_t i;
   size_t M = pList.size();
   wavepixel* p = NULL;                 // pointer to pixel structure

   int k;
   int    wsize  = w.size();
   double wstart = w.start();
   double f,t;

   if(!M || !w.size()) return;
   if(fl<0.) fl = low;
   if(fh<0.) fh = high;

   for(i=0; i<M; i++){
      p = &(pList[i]);

      f  = p->frequency*p->rate/2.;
      if(f>=fh && f+p->rate/2. >fh) continue;
      if(f <fl && f+p->rate/2.<=fl) continue;

      t  = start+(p->time+0.5)/p->rate;
      k  = int((t-wstart)*w.rate());     // time index in variability array

      if(k>=wsize) k -= k ? 1 : 0; 
      if(k<0 || k>=wsize) {
	 cout<<"wavecluster::setvar() - invalid input\n";
	 continue;
      }

      p->variability = w.data[k]; 
   }
   return;
}


double wavecluster::getNoiseRMS(double t, double fl, double fh)
{
   if(!nRMS.size()) return 1.;

   size_t i;
   size_t M = nRMS.maxLayer()+1;    // number of layers in nRMS
   size_t n = size_t(fl/(nRMS.gethigh()/M)); // first layer to get in nRMS
   size_t m = size_t(fh/(nRMS.gethigh()/M)); // last layer to get in nRMS

   double rms = 0.;
   double x;
   slice S;

   int inRMS = int((t-nRMS.start())*nRMS.rate());
   int inVAR = nVAR.size() ? int((t-nVAR.start())*nVAR.rate()) : 0;

   if(inRMS>=int(nRMS.size()/M)) inRMS -= inRMS ? 1 : 0; 
   if(inVAR>=int(nVAR.size()))   inVAR -= inVAR ? 1 : 0; 

   if(inRMS<0 || inVAR<0 || n>=m  ||
      inRMS >= int(nRMS.size()/M) ||
      inVAR >= int(nVAR.size())) 
   {
      cout<<"wavecluster::getNoiseRMS() - invalid pixel time\n";
      return 0.;
   }

   for(i=n; i<m; i++) {         // get noise vector for specified fl-fh
      S = nRMS.getSlice(i);
      x = nRMS.data[S.start()+inRMS*S.stride()];
      rms += 1./x/x;
   }
   rms /= double(m)-double(n);
   rms  = sqrt(1./rms);

   if(!nVAR.size() || fh<low || fl>high) return rms;

   return rms*double(nVAR.data[inVAR]);
}

//**************************************************************************
// return cluster parameters.
//**************************************************************************
wavearray<float> wavecluster::get(char* name, int index, size_t type)
{
   wavearray<float> out;
   if(!cList.size()) return out;

   size_t k;
   size_t mp,mm;
   size_t it_size;
   size_t it_core;
   size_t out_size = 0;
   double x,y;
   double a,b;
   double t,r;
   double sum = 0.;
   int ID, rate;

   wavearray<int> skip;
   list<vector_int>::iterator it;
   vector_int* pv = NULL;
   size_t M = pList.size();
   size_t m = abs(index);
   if(m==0) m++;

   out.resize(cList.size());
   out.start(start);
   out.rate(1.);
   out = 0.;

   char c = '0';         

   if(strstr(name,"ID"))     c = 'i';   // clusterID
   if(strstr(name,"size"))   c = 'k';   // size
   if(strstr(name,"volume")) c = 'v';   // volume
   if(strstr(name,"start"))  c = 's';   // start time
   if(strstr(name,"stop"))   c = 'd';   // stop time
   if(strstr(name,"low"))    c = 'l';   // low frequency
   if(strstr(name,"high"))   c = 'h';   // high frequency
   if(strstr(name,"time"))   c = 't';   // time averaged over SNR, index>=0 - Gauss
   if(strstr(name,"TIME"))   c = 'T';   // time averaged over hrss, index>=0 - Gauss
   if(strstr(name,"freq"))   c = 'f';   // frequency averaged over SNR, index>=0 - Gauss
   if(strstr(name,"FREQ"))   c = 'F';   // frequency averaged over hrss, index>=0 - Gauss
   if(strstr(name,"energy")) c = 'e';   // energy;       index<0 - rank, index>0 - Gauss
   if(strstr(name,"like"))   c = 'Y';   // likelihood; index<0 - rank, index>0 - Gauss
   if(strstr(name,"sign"))   c = 'z';   // significance;   index<0 - rank, index>0 - Gauss
   if(strstr(name,"corr"))   c = 'x';   // xcorrelation
   if(strstr(name,"asym"))   c = 'a';   // asymmetry 
   if(strstr(name,"grand"))  c = 'g';   // grandAmplitude 
   if(strstr(name,"rate"))   c = 'r';   // cluster rate
   if(strstr(name,"SNR"))    c = 'S';   // cluster SNR: index<0 - rank, index>0 - Gauss
   if(strstr(name,"hrss"))   c = 'H';   // log10(calibrated cluster hrss) 
   if(strstr(name,"noise"))  c = 'n';   // log10(average calibrated noise) 

   if(c=='0') return out;

   k = 0;

   for(it=cList.begin(); it!=cList.end(); it++){

     ID = pList[((*it)[0])].clusterID;
     if(sCuts[ID-1]) continue;   // apply selection cuts

     it_size = it->size();
     it_core = 0;
     skip.resize(it_size);

     pv =&(cRate[ID-1]);
     rate = type && type<=pv->size()? (*pv)[type-1] : 0;

//     cout<<(*pv)[0]<<"  "<<(*pv)[1]<<"  "<<pv->size()<<endl;

     for(k=0; k<it_size; k++) {          // fill skip array
	M = (*it)[k];
	skip.data[k] = 1;
	if(!pList[M].core) continue;
	if(rate && int(pList[M].rate+0.1)!=rate) continue;
	skip.data[k] = 0;
	it_core++;
     }

     if(!it_core) continue;     // skip cluster
     
     switch (c) {

     case 'i':          // get cluster ID 
       M = (*it)[0];
       out.data[out_size++] = pList[M].clusterID; 
       break;

     case 'k':          // get cluster core size 
       out.data[out_size++] = float(it_core);
       break;

     case 'r':          // get cluster rate 
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(!skip.data[k]) break;
       }
       out.data[out_size++] = pList[M].rate;
       break;

     case 'a':          // get cluster asymmetry 
     case 'x':          // get cluster x-correlation parameter 
       mp = 0; mm = 0;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(skip.data[k]) continue;
	 if(pList[M].amplitude.size()<m) continue;

	 x = pList[M].amplitude[m-1];
	 if(x>0.) mp++;
	 else     mm++;
       }
       if(c == 'a') out.data[out_size++] = (float(mp)-float(mm))/(mp+mm);
       else         out.data[out_size++] = signPDF(mp,mm); 
       break;
       
     case 'e':          // get cluster energy
     case 'S':          // get cluster SNR
     case 'Y':          // get cluster likelihood
     case 'z':          // get cluster significance
       y = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(skip.data[k]) continue;
	 if(pList[M].amplitude.size()<m) continue;

	 x = pList[M].amplitude[m-1];
	 if(index>0){            // get Gaussian statistics
	   if(c=='Y' || c=='z')  y += pow(fabs(x)+1.11/2,2)/2./1.07 + log(bpp);
	   else if(c=='S')       y += x*x-1.;
	   else if(c=='e')       y += x*x;
	 }
	 else {                  // get rank statistics
	   if(c=='Y' || c=='z')  y += fabs(x);
	   else if(c=='S')       y += pow(sqrt(2*1.07*(fabs(x)-log(bpp)))-1.11/2.,2)-1.;
	   else if(c=='e')       y += pow(sqrt(2*1.07*(fabs(x)-log(bpp)))-1.11/2.,2);
	 }
       }     
       out.data[out_size++] = c=='z' ? gammaCL(y,it_core) : y; 
       break;
       
     case 'g':          // get cluster grand amplitude
       y = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];

	 if(skip.data[k]) continue;
	 if(pList[M].amplitude.size()<m) continue;

	 x = fabs(pList[M].amplitude[m-1]);
	 if(x>y) y = x;
       }
       out.data[out_size++] = y;
       break;
       
     case 's':          // get cluster start time
     case 'd':          // get cluster stop time 
       a = 1.e99;
       b = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(skip.data[k]) continue;
	 y = 1./pList[M].rate;               // time resolution
	 x = y * pList[M].time;              // pixel time relative to start
	 if(x  <a) a = x;                    
	 if(x+y>b) b = x+y;
       }
       out.data[out_size++] = (c=='s') ? a : b;
       break;
       
     case 'l':             // get low frequency
     case 'h':             // get high frequency
       a = 1.e99;
       b = 0.;
       for(k=0; k<it_size; k++){
	 M = (*it)[k];
	 if(skip.data[k]) continue;
	 y = pList[M].rate/2.;           // frequency resolution
	 x = y * pList[M].frequency;     // pixel frequency
	 if(x  <a) a = x;
	 if(x+y>b) b = x+y;
       }
       out.data[out_size++] = (c=='l') ? a : b;
       break;

     case 't':			// get central time (SNR)
     case 'f':			// get central frequency (SNR)
     case 'T':			// get central time (hrss)
     case 'F':			// get central frequency (hrss)
       a = 0.;
       b = 0.;
       for(k=0; k<it_size; k++) {
         M = (*it)[k];

	 if(skip.data[k]) continue;
         if(pList[M].amplitude.size()<m) continue;

	 r = pList[M].variability;
	 if(c =='T' || c =='F') {
//	    f = pList[M].frequency*pList[M].rate/2.;      // low frequency
//	    t = start+(pList[M].time+0.5)/pList[M].rate;  // pixel time
//	    r = getNoiseRMS(t,f,f+pList[M].rate/2.);    // noise rms for this pixel
	    r *= pList[M].noiserms;
	 }
	 
         x = pList[M].amplitude[m-1];
	 t = 1./pList[M].rate;
         if(index>=0) {      // get Gaussian statistics
	    x = (x*x-1.)*r*r;
	 }
	 else {              // get rank statistics
	    x = pow(sqrt(2*1.07*(fabs(x)-log(bpp)))-1.11/2.,2)-1.;   
	 }
	 if(x<0.) x = 0.;

	 if(c =='t' || c =='T'){
//	   y = 1./pList[M].rate;                      // time resolution
	   a += (pList[M].time+0.5)*x;              // pixel time sum
	   b += x*pList[M].rate;
	 }
	 else {
//	   y = pList[M].rate/2.;                      // frequency resolution
	   a += (pList[M].frequency+0.5)*x;         // pixel frequency sum
	   b += x*2./pList[M].rate;
	 }
//	 b += x;
       }
       out.data[out_size++] = b>0. ? a/b : -1.;
       break;

     case 'H':			// get calibrated hrss
     case 'n':			// get calibrated noise

       mp  = 0;
       sum = 0.;
       out.data[out_size] = 0.;
//       if(!nRMS.size()) { out_size++; break; }    // no calibration constants

       for(k=0; k<it_size; k++) {
         M = (*it)[k];

	 if(skip.data[k]) continue;
         if(pList[M].amplitude.size()<m) continue;

// calculate noise RMS for pixel

//	 f = pList[M].frequency*pList[M].rate/2.;      // low frequency
//	 t = start+(pList[M].time+0.5)/pList[M].rate;  // pixel time
//	 r = getNoiseRMS(t,f,f+pList[M].rate/2.);    // noise rms for this pixel

	 r = pList[M].variability*pList[M].noiserms;
	 mp++;

	 if(c == 'H'){
	    a = pow(pList[M].amplitude[m-1],2)-1.;
	    sum += a<0. ? 0. : a*r*r;
	 }
	 else {
	    sum += 1./r/r;
	 }

       }

       if(c == 'n') { sum = double(mp)/sum; }     // noise hrss
       out.data[out_size++] = float(log(sum)/2./log(10.));
       break;

     case 'v':
     default:
       out.data[out_size++] = it_size;
       break;
     }
   }
   out.resize(out_size);
   return out;
}

// initialize from binary WSeries

double wavecluster::setMask(WSeries<double>& w, int nc, bool halo)
{
#if !defined (__SUNPRO_CC)
   int i;
   int j;
   int x, y, L, k;
   int* q = NULL;
   int* p = NULL; 
   int* pp; 
   int* pm;
   wavepixel pix;
   
   wavearray<double> a;

   if(!w.pWavelet->BinaryTree()) return 1.;

   start = w.start();            // set start time
   bpp   = w.getbpp();

   int ni = w.maxLayer()+1;
   int nj = w.size()/ni;
   int n = ni-1;
   int m = nj-1;
   int nPixel = 0;

   int FT[ni][nj];
   int XY[ni][nj];

   for(i=0; i<ni; i++){
      p  = FT[i]; 
      q  = XY[i];
 
      w.getLayer(a,i);	

      for(j=0; j<nj; j++){
	 p[j] = (a.data[j]!=0) ? 1 : 0;
	 if(p[j]) nPixel++;
	 q[j] = 0;
      }
   }

   pList.clear();
   sCuts.clear();
   cList.clear();

   if(!nc || ni<3 || nPixel<2) return double(nPixel)/double(size());

// calculate number of neighbors and ignore 1 pixel clusters

// corners
   if(FT[0][0]) XY[0][0] = FT[0][1]   + FT[1][0]   + FT[1][1];
   if(FT[0][m]) XY[0][m] = FT[1][m]   + FT[1][m-1] + FT[0][m-1];
   if(FT[n][0]) XY[n][0] = FT[n-1][0] + FT[n-1][1] + FT[n][1];
   if(FT[n][m]) XY[n][m] = FT[n][m-1] + FT[n-1][m] + FT[n-1][m-1];
  
// up/down raws   
   for(j=1; j<m; j++){
      p  = FT[0]; 
      q  = FT[n];

      if(p[j]){
	 pp = FT[1];
	 XY[0][j] = p[j-1]+p[j+1] + pp[j-1]+pp[j]+pp[j+1];
      }
      if(q[j]){
	 pm = FT[n-1];
 	 XY[n][j] = q[j-1]+q[j+1] + pm[j-1]+pm[j]+pm[j+1];
      }
   }

   for(i=1; i<n; i++){
      pm = FT[i-1]; p  = FT[i]; pp = FT[i+1]; q = XY[i];

      if(p[0])       // left side
	 q[0] = p[1] + pm[0]+pm[1] + pp[0]+pp[1];

      if(p[m])     // right side
	 q[m] = p[m-1] + pm[m]+pm[m-1] + pp[m]+pp[m-1];

      for(j=1; j<m; j++){
	 if(p[j])
	    q[j] = pm[j-1]+pm[j]+pm[j+1] + pp[j-1]+pp[j]+pp[j+1] + p[j-1]+p[j+1];
      }
   }

/**************************************/
/* remove clusters with 2,3 pixels  */
/**************************************/

   if(nc>1){

// corners
      if(XY[0][0]){ 
	 x = XY[0][1]   + XY[1][0]   + XY[1][1];
	 if(x==1 || x==4) XY[0][0]=XY[0][1]=XY[1][0]=XY[1][1]=0;
      }
      if(XY[0][m]){ 
	 x = XY[1][m]   + XY[1][m-1] + XY[0][m-1];
	 if(x==1 || x==4) XY[0][m]=XY[1][m]=XY[1][m-1]=XY[0][m-1]=0;
      }
      if(XY[n][0]){ 
	 x = XY[n-1][0] + XY[n-1][1] + XY[n][1];
	 if(x==1 || x==4) XY[n][0]=XY[n-1][0]=XY[n-1][1]=XY[n][1]=0;
      }
      if(XY[n][m]){
	 x = XY[n-1][m]   + XY[n][m-1] + XY[n-1][m-1];
	 if(x==1 || x==4) XY[n][m]=XY[n-1][m]=XY[n][m-1]=XY[n-1][m-1]=0;
      }

// up/down raws   
      for(j=1; j<m; j++){
	 p  = XY[0]; 
	 q  = XY[n];

	 if(p[j]==1 || p[j]==2){
	    if(p[j-1]+p[j+1] < 4){
	       pp = XY[1];
	       L = p[j-1]+p[j+1] + pp[j];
	       x = pp[j-1] + pp[j+1] + L;
	       
	       if(x==1 || (p[j]==2 && nc>2 && (x==2 || (x==4 && L==4))))
		  p[j]=p[j-1]=p[j+1]=pp[j-1]=pp[j]=pp[j+1]=0;
	    }
	 }

	 if(q[j]==1 || q[j]==2){
	    if(q[j-1]+q[j+1] < 4){
	       pm = XY[n-1];
	       L = q[j-1]+q[j+1] + pm[j];
	       x = pm[j-1] + pm[j+1] + L;

	       if(x==1 || (q[j]==2 && nc>2 && (x==2 || (x==4 && L==4))))
		  q[j]=q[j-1]=q[j+1]=pm[j-1]=pm[j]=pm[j+1]=0;
	    }
	 }
      }

// regular case
      for(i=1; i<n; i++){
	 pm = XY[i-1];
	 p  = XY[i];
	 pp = XY[i+1];
	 
	 
	 if(p[0]==1 || p[0]==2){       // left side
	    if(pm[0]+pp[0] < 4){
	       L = p[1] + pm[0] + pp[0];
	       x = pm[1] + pp[1] + L;
	       
	       if(x==1 || (p[0]==2 && nc>2 && (x==2 || (x==4 && L==4))))
		  p[0]=pm[0]=pp[0]=pm[1]=pp[1]=p[1]=0;;
	    }
	 }

	 if(p[m]==1 || p[m]==2){     // right side
	    if(pm[m]+pp[m] < 4){
	       L = p[m-1] + pm[m] + pp[m];
	       x = pm[m-1] + pp[m-1] + L;
	       
	       if(x==1 || (p[m]==2 && nc>2 && (x==2 || (x==4 && L==4))))
		  p[m]=pm[m]=pp[m]=pm[m-1]=pp[m-1]=p[m-1]=0;
	    }
	 }
	 
	 for(j=1; j<m; j++){
	    y = p[j];
	    if(y == 1 || y == 2){
	       if(pm[j]+pp[j] >3) continue;
	       if(p[j-1]+p[j+1] >3) continue;

	       L  = pm[j]+pp[j] + p[j-1]+p[j+1];
	       x  = pm[j-1]+pm[j+1] + pp[j-1]+pp[j+1] + L;
	       
	       if(x==1 || (y==2 && nc>2 && (x==2 || (x==4 && L==4))))
		  p[j]=p[j-1]=p[j+1]=pm[j-1]=pm[j]=pm[j+1]=pp[j-1]=pp[j]=pp[j+1]=0;
	    }
	 }
      }
   }

/**************************************/
/* fill in the pixel mask             */
/**************************************/

   pix.amplitude.clear();                 // clear link pixels amplitudes 
   pix.neighbors.clear();                 // clear link vector for neighbors 
   pix.clusterID = 0;                     // initialize cluster ID
   L = 1<<w.getLevel();                   // number of layers
   pix.rate = float(w.wavearray<double>::rate()/L);          // pixel bandwidth

   size_t &f = pix.frequency;
   size_t &t = pix.time;

   L = 0;

   for(i=0; i<ni; i++){
      p  = FT[i]; q  = XY[i];
      for(j=0; j<nj; j++)
	 p[j] = q[j];
   }

   for(i=0; i<ni; i++){
      q  = XY[i];
      p  = FT[i];
      for(j=0; j<nj; j++){
         if(q[j]) {
            t = j; f = i;                // time index; frequency index;
	    pix.core = true;             // core pixel
            pList.push_back(pix);        // save pixel
            p[j] = ++L;                  // save pixel mask index (Fortran indexing)

            if(halo){                    // include halo
	       pix.core = false;         // halo pixel

               if(i>0 && j>0) { 
                  t=j-1; f=i-1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(i>0) {
                  t=j;   f=i-1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(i>0 && j<m) { 
                  t=j+1; f=i-1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(j>0)        { 
                  t=j-1; f=i;   if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(j<m)        { 
                  t=j+1; f=i;   if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(i<n && j>0) { 
                  t=j-1; f=i+1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(i<n)        { 
                  t=j;   f=i+1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }
               if(i<n && j<m) { 
                  t=j+1; f=i+1; if(!FT[f][t]) {pList.push_back(pix); FT[f][t] = ++L;}
               }

            }             
	 }
      }
   }

   std::vector<int>* pN;
   int nM =  pList.size();
   slice S;

   for(k=0; k<nM; k++){

// set pixel amplitude
      i = pList[k].frequency;
      j = pList[k].time;
      if(int(i)>w.maxLayer()) cout<<"cluster::setMask() maxLayer error: "<<i<<endl;
      S = w.getSlice(i);                
      pList[k].amplitude.clear();
      pList[k].amplitude.push_back(w.data[S.start()+S.stride()*j]);

// set neighbors
      pN = &(pList[k].neighbors);
      L = 0;

      if(i==0 || i==n){                          // first or last layer
	 if(i==0){ p = FT[0]; q = FT[1];}
	 if(i==n){ p = FT[n]; q = FT[n-1];}

	 if(j==0){                               // first sample
	    if(p[1]) pN->push_back(p[1]-1);
	    if(q[1]) pN->push_back(q[1]-1);
	    if(q[0]) pN->push_back(q[0]-1);
	 }
	 else if(j==m){                          // last sample
	    if(p[m-1]) pN->push_back(p[m-1]-1);
	    if(q[m-1]) pN->push_back(q[m-1]-1);
	    if(q[m]<0) pN->push_back(q[m]-1);
	 }
	 else{                                   // samples in the middle
	    if(p[j-1]) pN->push_back(p[j-1]-1);
	    if(p[j+1]) pN->push_back(p[j+1]-1);
	    if(q[j-1]) pN->push_back(q[j-1]-1);
	    if(q[j])   pN->push_back(q[j]-1);
	    if(q[j+1]) pN->push_back(q[j+1]-1);
	 }
      }

      else{          
	 pp = FT[i+1];
	 p  = FT[i];
	 pm = FT[i-1];

	 if(j==0){                                // first sample
	    if(pm[0]) pN->push_back(pm[0]-1);
	    if(pp[0]) pN->push_back(pp[0]-1);
	    if( p[1]) pN->push_back(p[1]-1);
	    if(pm[1]) pN->push_back(pm[1]-1);
	    if(pp[1]) pN->push_back(pp[1]-1);
	 }
	 else if(j==m){
	    if(pm[m])   pN->push_back(pm[m]-1);          // last sample
	    if(pp[m])   pN->push_back(pp[m]-1);
	    if( p[m-1]) pN->push_back(p[m-1]-1);
	    if(pm[m-1]) pN->push_back(pm[m-1]-1);
	    if(pp[m-1]) pN->push_back(pp[m-1]-1);
	 }
	 else{
	    if(pm[j-1]) pN->push_back(pm[j-1]-1);
	    if(pm[j])   pN->push_back(pm[j]-1);
	    if(pm[j+1]) pN->push_back(pm[j+1]-1);
	    if( p[j-1]) pN->push_back(p[j-1]-1);
	    if( p[j+1]) pN->push_back(p[j+1]-1);
	    if(pp[j-1]) pN->push_back(pp[j-1]-1);
	    if(pp[j])   pN->push_back(pp[j]-1);
	    if(pp[j+1]) pN->push_back(pp[j+1]-1);
	 }
      }

      L = pList[k].neighbors.size();
      x = pList[k].core ? XY[i][j] : L;
      if((x != L && !halo) || L>8){ 
	cout<<"cluster::getMask() vector size error: "<<L<<"  reserved: "<<x<<endl;
	cout<<"k="<<k<<"  i="<<i<<" j="<<j<<endl;
      }
   }
#endif
   return double(pList.size())/double(size());
}













