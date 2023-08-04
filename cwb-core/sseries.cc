/*
# Copyright (C) 2019 Sergey Klimenko, Gabriele Vedovato, Valentin Necula
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


#include "sseries.hh"

ClassImp(SSeries<float>)


//______________________________________________________________________________
/* Begin_Html                                                                   
<center><h2>SSeries class</h2></center>                                           
SSeries is used to store the <font color="red">core+halo</font> pixels contained in the WSeries array<br>
The <font color="red">core</font> pixels are pixels selected according to some user criteria, see <a class="funcname" href="#SSeries_float_:AddCore">AddCore</a><br>
The <font color="red">halo</font> pixels are auxiliary pixels associated to each core pixel, see <a class="funcname" href="#SSeries_float_:SetHalo">SetHalo</a><br>
Note: WAT algorithms use halo pixels to compute the delayed amplitude.<br>
<p>

<h3><a name="usage">Usage</a></h3>
create sseries 
<pre>
   <a class="funcname" href="#SSeries_float_:SSeries_float_">SSeries&lt;float&gt;</a> ss;
</pre>
associate TF map to sparse map
<pre>
   ss.<a class="funcname" href="#SSeries_float_:SetMap">SetMap</a>(&tfmap);           
</pre>
set halo parameters
<pre>
   ss.<a class="funcname" href="#SSeries_float_:SetHalo">SetHalo</a>(network_time_delay);            
</pre>
add core pixels index to sparse map table
<pre>
   ss.<a class="funcname" href="#SSeries_float_:AddCore">AddCore</a>(ifoID,&netcluster);       
</pre>
update sparse map : add TF core+halo pixels to sparse map tables
<pre>
   ss.<a class="funcname" href="#SSeries_float_:UpdateSparseTable">UpdateSparseTable</a>();      
</pre>
resize to 0 the TF map : leave only the sparse map tables
<pre>
   ss.<a class="funcname" href="#SSeries_float_:Shrink">Shrink</a>();                  
</pre>
write sparse map to root file
<pre>
   TFile ofile("file.root","RECREATE");
   ss.Write("sparseMap");
   ofile.Close();
</pre>

<p>
<h3><a name="example">Example</a></h3>
<p>
The macro <a href="./tutorials/wat/SSeriesExample.C.html">SSeriesExample.C</a> is an example which shown how to use the SSeries class.<br>
The picture below gives the macro output plots.<br>
<p>
The <font color="red">TF Map : Signal</font> is the WDM transform of a SG100Q9 signal<br>
The <font color="red">TF Map : Signal + Noise</font> is the WDM transform of a SG100Q9 signal plus a random white noise<br>
The <font color="red">Sparse Map : Core Pixels</font> are the core pixels above the threshold energy 6<br>
The <font color="red">Sparse Map : Core+Halo Pixels</font> are the core pixels above the threshold + their associate halo pixels<br>
End_Html
Begin_Macro
SSeriesExample.C
End_Macro */


using namespace std;

//______________________________________________________________________________
// destructor
template<class DataType_t>
SSeries<DataType_t>::~SSeries() 
{
}

//______________________________________________________________________________
template<class DataType_t>
SSeries<DataType_t>::SSeries() :
WSeries<DataType_t>() {this->Init();}

//______________________________________________________________________________
template<class DataType_t>
SSeries<DataType_t>::SSeries(const WSeries<DataType_t> &w) :
WSeries<DataType_t>(w) {this->Init();}

//______________________________________________________________________________
template<class DataType_t>
SSeries<DataType_t>::SSeries(const Wavelet &w) :
WSeries<DataType_t>(w) {this->Init();}

//______________________________________________________________________________
template<class DataType_t>
SSeries<DataType_t>::SSeries(const wavearray<DataType_t>& value, const Wavelet &w) :
WSeries<DataType_t>(value, w) {this->Init();}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::Init(WSeries<DataType_t>* pws, bool reset) {

  WDM<double>* wdm;
  if(reset) ResetSparseTable();

  if(pws && reset) {                             // initialize from external wavelet
     if(this->pWavelet) {                        // delete default wavelet
        this->pWavelet->release();
        delete this->pWavelet;
     }

     this->pWavelet = pws->pWavelet->Init();     // light-weight wavelet without TD filters
     this->pWavelet->allocate(pws->size(), pws->data); // attach pws data
     
     wdm = (WDM<double>*)pws->pWavelet;          // setup WDM pointer
     this->rate(pws->rate());                    // set time-series rate
     this->wrate(pws->wrate());                  // set wavelet rate
     this->start(pws->start());                  // set data start time
     this->stop(pws->stop());                    // set data stop time
     this->edge(pws->edge());                    // set data edge length
     this->time_Halo = wdm->getTDFsize();        // store half size of TD filter
  }
  else if(pws) {                                 // cross-check consistency if input pws
     if(abs(pws->rate()-this->rate())) {
        cout << "SSeries::Init : Inconsistent index rate  " << pws->rate() 
             << " previously setted index rate : " << this->rate() << endl; 
        exit(1);
     }  
     if(pws->maxLayer()!=this->maxLayer()) {
        cout << "SSeries::Init : Inconsistent maxLayer  " << pws->maxLayer() 
             << " previously setted maxLayer: " << this->maxLayer() << endl; 
        exit(1);
     }  
     if(pws->size()!=this->pWavelet->nWWS) {
        cout << "SSeries::Init : Inconsistent nWWS  " << pws->size() 
             << " previously setted nWWS : " << this->pWavelet->nWWS << endl; 
        exit(1);
     }  
     if(pws->start()!=this->start()) {
        cout << "SSeries::Init : Inconsistent start  " << pws->start() 
             << " previously setted start : " << this->start() << endl; 
        exit(1);
     }  
     if(pws->pWavelet->getTDFsize() != this->time_Halo) {
        cout << "SSeries::Init : Inconsistent time_Halo  " << pws->pWavelet->getTDFsize() 
             << " previously setted time_Halo : " << this->time_Halo << endl; 
        exit(1);
     }  
  }

  wdm = (WDM<double>*) this->pWavelet;

  if(this->pWavelet->m_WaveType==WDMT) {
    wdm_BetaOrder = wdm->BetaOrder;
    wdm_m_Layer = wdm->m_Layer;
    wdm_KWDM = wdm->KWDM;
    wdm_precision = wdm->precision;
    wdm_rate = this->rate();
    wdm_start = this->start();
    wdm_nSTS = wdm->nSTS;
  } 
  else {
    wdm_BetaOrder = 0;
    wdm_m_Layer = 0;
    wdm_KWDM = 0;
    wdm_precision = 0;
    wdm_rate = 0;
    wdm_start = 0;
    wdm_nSTS = 0;
  }
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::ResetSparseTable() {
//
// Reset the sparse tables
//

  sparseLookup.resize(0);
  sparseType.ResetAllBits();
  sparseType.Compact();
  sparseIndex.resize(0);
  sparseMap00.resize(0);
  sparseMap90.resize(0);
  core.ResetAllBits();
  core.Compact();
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::AddCore(size_t ifoID, netcluster* pwc, int ID) {
//
// Add core pixels of the detector ID=ifoID contained in the pwc netcluster with index ID
//
// ifoID : detector index 
// pwc   : pointer to netcluster
// ID    : cluster index - if ID=0 all cluster are selected (default ID=0)
//

   CheckWaveletType("AddCore");

   int index;
   int R = int(this->wrate()+0.1);
   
   wavearray<double> cid;                               // buffers for cluster ID
   cid = pwc->get((char*)"ID",0,'S',0);
   int K = cid.size();

   for(int ik=0; ik<K; ik++) {                          // loop over clusters
      
      int id = size_t(cid.data[ik]+0.1);

      if(ID && id!=ID) continue;			// if ID>0 skip id!=ID
      if(pwc->sCuts[id-1] == 1) continue;         	// skip rejected clusters
      
      vector<int>* vint = &(pwc->cList[id-1]);
      int V = vint->size();
      //cout << "CID " << id << " SIZE " << V << endl;
      
      for(int l=0; l<V; l++) {                          // loop over pixels
         netpixel* pix = pwc->getPixel(id,l);
         if(int(pix->rate+0.01)!=R) continue;		// skip pixel with bad rate
         index = (int)pix->data[ifoID].index;
         core.SetBitNumber(index); 
      }
   }
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::SetHalo(double maxTau, int lHalo, int tHalo) {
//
//  maxTau : delay time (sec) 
//           is stored in net_Delay
//           used to compute extraHalo = net_Delay*sparseRate
//  lHalo  : number of layers above and below each core pixel
//           is stored in layerHalo
//           the total number of layers in the halo is 2*layerHalo+1
//  tHalo  : number of slice on the right and on the left each core pixel 
//           is stored in timeHalo
//           the total number of pixels on the time axis is 2*(timeHalo+extraHalo)+1
//           The default value is -1 : the value is automatically selected from the
//           associated TF map with WDM::getTDFsize() method.  
//           
//  For each core pixels (if layerHalo = 1) the following pixels are saved
//           
//  core      = '.' 
//  extraHalo = '++++++++'
//  timeHalo  = 'xxxx'
//
//  ++++++++xxxx xxxx++++++++
//  ++++++++xxxx.xxxx++++++++
//  ++++++++xxxx xxxx++++++++
//

  ResetSparseTable();
  this->layerHalo = lHalo;
  this->net_Delay = maxTau;
  if(tHalo>=0) this->time_Halo = tHalo;  // override extraHalo 
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::UpdateSparseTable() {
//
// Use the core pixels and halo parameters to update the sparse maps with core+halo pixels
//

   CheckWaveletType("UpdateSparseTable");

   if(time_Halo==0 && layerHalo==0) return;

   extraHalo = int(net_Delay*this->wrate())+8;  // init extra halo : WARNING value 8 ad hoc - to be fixed
   int hSlice = time_Halo+extraHalo;	     // halo slices
   
   TBits cluster;			     // core+halo
   int nLayer = this->maxLayer()+1;          // number of WDM layers
   int nSlice = this->sizeZero();            // number of samples in wavelet layer

   for(int i=0;i<nSlice;i++) {
      for(int j=0;j<nLayer;j++) {
         if(!Core(i,j)) continue;
         int ib = i-hSlice<0 ? 0 : i-hSlice;
         int ie = i+hSlice>nSlice-1 ? nSlice-1 : i+hSlice;
         int jb = j-layerHalo<0 ? 0 : j-layerHalo;
         int je = j+layerHalo>nLayer-1 ? nLayer-1 : j+layerHalo;
         for(int ii=ib;ii<=ie;ii++) 
            for(int jj=jb;jj<=je;jj++) {
               cluster.SetBitNumber(ii*nLayer+jj);
            }
      }
   }
   
   // fill sparse tables
   int csize=cluster.CountBits();	// is the number of non zero pixels
   sparseLookup.resize(nLayer+1);
   sparseType.ResetAllBits();sparseType.Compact();
   sparseIndex.resize(csize);
   sparseMap00.resize(csize);
   sparseMap90.resize(csize);
   
   // data are sorted respect to index
   int n=0;
   for(short j=0;j<nLayer;j++) {
      sparseLookup[j]=n;
      for(int i=0;i<nSlice;i++) {
         int index=i*nLayer+j; 
 
         if(cluster.TestBitNumber(index)) {
            if(Core(index)) sparseType.SetBitNumber(n);
            sparseIndex.data[n] = index;
            sparseMap00.data[n] = GetMap00(index);
            sparseMap90.data[n] = GetMap90(index);
            n++;
         }
      }
   }
   sparseLookup[nLayer]=n;
}

//______________________________________________________________________________
template<class DataType_t>
int SSeries<DataType_t>::GetSparseSize(bool bcore) {

//  CheckWaveletType("GetSparseSize");

  if(bcore) return sparseType.CountBits();	// return number of core pixels 
  else      return sparseIndex.size();	        // retun number of total pixels core+halo
}

//______________________________________________________________________________
template<class DataType_t>
wavearray<int> SSeries<DataType_t>::GetSparseIndex(bool bcore) {

  CheckWaveletType("GetSparseIndex");

  if(!bcore) return sparseIndex;

  wavearray<int> si(GetSparseSize()); 
  int isize=0;
  for(int i=0;i<(int)sparseIndex.size();i++) 
    if(sparseType.TestBitNumber(i)) si[isize++]=sparseIndex[i];

  return si;
}


//______________________________________________________________________________
template<class DataType_t>
bool SSeries<DataType_t>::GetSTFdata(int index, SymmArraySSE<float>* pS) {

  CheckWaveletType("GetSTFdata");

  // if(!Core(index)) return false;

  // check consistency of input vector dimensions
  int nL = 2*layerHalo+1;	          // number of layers in halo
  int nS = 2*time_Halo+1;                 // number of time samples in halo
  int nLayer = this->maxLayer()+1;        // number of WDM layers

  if(pS[0].Last()!=time_Halo) {
    cout << "SSeries<DataType_t>::GetSTFdata : Input Vector Error - wrong slice dimension" << endl;
    cout << "Input dim : " << 2*pS[0].Last()+1 << " Sparse TF slices : " << nS;
    exit(1);
  }

  int layer = GetLayer(index);

  int jb = layer-layerHalo<0 ? 0 : layer-layerHalo;
  int je = layer+layerHalo>nLayer-1 ? nLayer-1 : layer+layerHalo;
  // set to zero layer outside of TF map
  for(int j=0;j<nL;j++) {
    int jp = j+layer-layerHalo;
    if(jp>=jb && jp<=je) continue;
    bzero(&pS[j][-time_Halo],nS*sizeof(float));
    bzero(&pS[nL+j][-time_Halo],nS*sizeof(float));
  }
  // fill array with Map00/90
  for(int j=jb;j<=je;j++) {
    int start = sparseLookup[j];		// sparse table layer offset 
    int end = sparseLookup[j+1]-1;		// sparse table layer+1 offset 
    int key = index+(j-layer); 
    int sindex = binarySearch(sparseIndex.data, start, end, key);
    if(sindex<0) 
      {cout << "SSeries<DataType_t>::GetSTFdata : index not present in sparse table" << endl;exit(1);}
    int ib = sindex-time_Halo;
    int ie = sindex+time_Halo;
    if((ib<start)||(ie>end)) {
      cout << "SSeries<DataType_t>::GetSTFdata : Exceed TF map boundaries" << endl;
      cout << "Check buffer scratch length : " << 
              "probably it is non sufficient to get the correct number of samples" << endl; 
      exit(1);
    }
    int jp = j-layer+layerHalo;
    memcpy(&pS[jp][-time_Halo],   &(sparseMap00.data[ib]),nS*sizeof(float));  // copy Map00
    memcpy(&pS[nL+jp][-time_Halo],&(sparseMap90.data[ib]),nS*sizeof(float));  // copy Map90
  }
  return true;
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::Expand(bool bcore) {
//
// rebuild wseries TF map from sparse table
//
// input - bcore : true -> only core pixels are used
//

  CheckWaveletType("Expand");
  int nLayer = this->maxLayer()+1;        // number of WDM layers
  int nSlice = this->sizeZero();          // number of samples in wavelet layer

  wavearray<DataType_t> x(wdm_nSTS);	
  x.rate(wdm_rate);
  x.start(wdm_start);
  WDM<DataType_t> wdm = WDM<DataType_t>(wdm_m_Layer, wdm_KWDM, wdm_BetaOrder, wdm_precision);
  WSeries<DataType_t>::Forward(x,wdm);

  // rebuild TF map
  for(int i=0;i<(int)sparseIndex.size();i++) {
    if(bcore && !sparseType.TestBitNumber(i)) continue;
    int index = sparseIndex[i];
    SetMap00(index,sparseMap00[i]);
    SetMap90(index,sparseMap90[i]);
  }
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::Clean() {
//
// set to 0 all TF map pixels which do not belong to core+halo
//

  CheckWaveletType("Clean");

  TBits cluster;	// core+halo

  for(int i=0;i<(int)sparseIndex.size();i++) 
    cluster.SetBitNumber(sparseIndex[i]);

  for(int i=0;i<=this->maxIndex();i++) 
    if(!cluster.TestBitNumber(i)) {SetMap00(i,0);SetMap90(i,0);}
}

//______________________________________________________________________________
template<class DataType_t>
int SSeries<DataType_t>::binarySearch(int array[], int start, int end, int key) {
   // Determine the search point.
   // int searchPos = end + ((start - end) >> 1);
   // int searchPos = (start + end) / 2;	
   int searchPos = (start + end) >> 1;	
   // If we crossed over our bounds or met in the middle, then it is not here.
   if (start > end)
      return -1;
   // Search the bottom half of the array if the query is smaller.
   if (array[searchPos] > key)
      return binarySearch (array, start, searchPos - 1, key);
   // Search the top half of the array if the query is larger.
   if (array[searchPos] < key)
      return binarySearch (array, searchPos + 1, end, key);
   // If we found it then we are done.
   if (array[searchPos] == key)
      return searchPos;
}

//______________________________________________________________________________
template<class DataType_t>
int SSeries<DataType_t>::binarySearch(int array[], int size, int key) {
   return binarySearch(array, 0, size - 1, key);
}

//______________________________________________________________________________
template<class DataType_t>
void SSeries<DataType_t>::Streamer(TBuffer &R__b)
{
   // Stream an object of class SSeries<DataType_t>.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      WSeries<DataType_t>::Streamer(R__b);
      sparseLookup.Streamer(R__b);
      sparseIndex.Streamer(R__b);
      sparseMap00.Streamer(R__b);
      sparseMap90.Streamer(R__b);
      //R__b >> nLayer;
      //R__b >> nSlice;
      R__b >> layerHalo;
      R__b >> time_Halo;
      R__b >> wdm_BetaOrder;
      R__b >> wdm_m_Layer;
      R__b >> wdm_KWDM;
      R__b >> wdm_precision;
      R__b >> wdm_rate;
      R__b >> wdm_start;
      if(R__v > 1) R__b >> wdm_nSTS;
      // restore sparseIndex & sparseType  & core
      core.ResetAllBits();core.Compact(); 
      sparseType.ResetAllBits();sparseType.Compact(); 
      for(int i=0;i<(int)sparseIndex.size();i++) {
	if(sparseIndex[i]<0) {
          // restore index 
          sparseIndex[i]+=1;
          sparseIndex[i]*=-1;
          // restore indexType 
          sparseType.SetBitNumber(i);
        }
        // restore core 
        core.SetBitNumber(sparseIndex[i]);
      }
      R__b.CheckByteCount(R__s, R__c, SSeries<DataType_t>::IsA());
   } else {
      R__c = R__b.WriteVersion(SSeries<DataType_t>::IsA(), kTRUE);
      // merge sparseIndex & sparseType 
      for(int i=0;i<(int)sparseIndex.size();i++) {
	if(sparseType.TestBitNumber(i)) {
          // add 1 to avoid 0 index  
          sparseIndex[i]+=1;
          // set as negative index the core pixels
          sparseIndex[i]*=-1;
        }
      }
      WSeries<DataType_t>::Streamer(R__b);
      sparseLookup.Streamer(R__b);
      sparseIndex.Streamer(R__b);
      sparseMap00.Streamer(R__b);
      sparseMap90.Streamer(R__b);
      //R__b << nLayer;
      //R__b << nSlice;
      R__b << layerHalo;
      R__b << time_Halo;
      R__b << wdm_BetaOrder;
      R__b << wdm_m_Layer;
      R__b << wdm_KWDM;
      R__b << wdm_precision;
      R__b << wdm_rate;
      R__b << wdm_start;
      R__b << wdm_nSTS;
      R__b.SetByteCount(R__c, kTRUE);
   }
}


// instantiations

#define CLASS_INSTANTIATION(class_) template class SSeries< class_ >;

//CLASS_INSTANTIATION(short)
//CLASS_INSTANTIATION(int)
//CLASS_INSTANTIATION(unsigned int)
//CLASS_INSTANTIATION(long)
//CLASS_INSTANTIATION(long long)
CLASS_INSTANTIATION(float)
CLASS_INSTANTIATION(double)
 
