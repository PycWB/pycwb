//
// Created by Yumeng Xu on 10.03.23.
//
#define CWB_CORE_COHERENCE_CC
#include "coherence.hh"

using namespace std;

netcluster* getNetworkPixels(int nIFO, std::vector<WSeries<double>> tf_maps, wavearray<short> veto,
                      double Edge, int LAG, double Eo, double norm, std::vector<double> lagShift)
{
    // create a wc_List object and return it
    netcluster* wc_List = new netcluster;


    if(tf_maps[0].w_mode != 1) {
        printf("network::getNetworkPixels(): invalid whitening mode.");
//        return std::make_tuple(0, 0.0, &wc_List);
        return NULL;
    }

    WSeries<double>* pTF = &tf_maps[0]; // pointer to first TF map
    WSeries<double> MAP; MAP = *pTF; MAP=0.;            // initialize TF map
    wavearray<double>* hTS = &tf_maps[0]; // pointer to first TS data

    int i,j,k,m,n,NN,jj,nM,jE,jb,je,J,K;

    double Em = 2*Eo;                                   // maximum (sole pixel) threshold
    double Eh = Em*Em;                                  // halo energy^2
    double R  = pTF->wrate();                           // pixel layer rate
    double r  = hTS->rate();                            // TS rate
    int N  = pTF->size();                               // size of TF array
    int M  = hTS->size();                               // size of TS array
    int I  = pTF->maxLayer()+1;                         // number of layers
    int II = pTF->maxLayer()-1;                         // number of layers - 2
    int jB = int(Edge*R+0.001);                   // number of samples in the edges
//    printf("I = %d \n", I);

    if(jB&1) {
        printf("getNetworkPixels(1): WDM parity violation");
        return NULL;
//        return std::make_tuple(0, 0.0, &wc_List);
    }

    if(jB < 3) {
        printf("network::getNetworkPixels(): insufficient data edge length.");
        return NULL;
//        return std::make_tuple(0, 0.0, &wc_List);
    }

    netpixel pix(nIFO);
    pix.core = true;
    pix.rate = R;
    pix.layers = I;

    int     in[NIFO];                                    // pixel time index
    int     IN[NIFO];                                    // pixel time index
    double* PDATA;
    double* pmap;
    double* pdata[NIFO];                                 // pointers to data
    double* pp[5];                                       // pointers to sorted F-arrays
    for(n=0; n<nIFO; n++) {                              // pointers to data
        pdata[n] = tf_maps[n].data;
    }

    long nPix = 0;
    size_t count = 0;                              // live pixel counter
    double a,b,E,Ct,Cb,Ht,Hb;

//    if(hist) {pixeLHood = *pTF; pixeLHood=-1.;}
    if(veto.size() != M) {                   // set veto array if it is not set
        veto.resize(M); veto = 1;
    }

    short* pveto = veto.data;                // pointer to veto


    double livTime = 0.;

    wc_List->clear();                    // clear netcluster structure
    wc_List->setlow(pTF->getlow());
    wc_List->sethigh(pTF->gethigh());

    a  = 1.e10; nM = 0;                            // master detector
    for(n=0; n<nIFO; n++) {
        b = lagShift[n];    // shift in seconds
        if(a>b) { a = b; nM = n; }
    }

    for(n=0; n<nIFO; n++) {
        b = lagShift[n];    // shift in seconds
        K = int((b-a)*R+0.001);                     // time shift wrt reference
        if(K&1) {cout<<"getNetworkPixels(2): WDM parity violation\n"; exit(1);}
        in[n] = IN[n] = K+jB;                       // time index of first pixel in the layer
    }

    int ib=1;
    int ie=I;
    for(i=0; i<I; i++) {                           // select bandwidth
        if(pTF->frequency(i) <= pTF->gethigh()) ie=i;
        if(pTF->frequency(i) <= pTF->getlow())  ib=i+1;
    }
    if(ie>I-1) ie = I-1;                           // required by catalog
    if(ib<1)   ib = 1;                             // required by catalog

    slice S = pTF->getSlice(0);
    jE = S.size()-jB;                              // last good sample in the layer
    NN = jE-jB;                                    // #of good samples in the layer
    if(jE&1) {cout<<"getNetworkPixels(3): WDM parity violation\n"; exit(1);}

    //cout<<r<<" "<<R<<" "<<I<<" "<<jB<<" "<<this->veto.size()<<endl;
    //cout<<ib<<" "<<ie<<" "<<NN<<" "<<jB<<" "<<jE<<endl;
    for(jj=0; jj<NN; jj++) {                       // loop over time stamps

        double VETO = 1.;
        pmap = MAP.data+(jj+jB)*I;                  // pointer to 0 F sample in MAP
        for(n=0; n<nIFO; n++) {
            if(in[n] >= jE) in[n] -= NN;             // go to jB sample
            jb = int(in[n]*r/R+0.01);                // first veto index
            je = int((in[n]+1)*r/R+0.01);            // last veto index
            while(jb<je) if(!pveto[jb++]) VETO=0.;   // set veto value
            PDATA = &(pdata[n][in[n]*I]);            // pointer to 0 F sample
            for(i=0; i<I; i++) pmap[i]+=*PDATA++;    // sum energy
            in[n]++;                                 // increment index pointer
        }

        for(i=0; i<I; i++) {
            pmap[i] *= VETO;
            if(pmap[i]<Eo || i<ib) pmap[i]=0.;       // zero sub-threshold pixels
            if(pmap[i]>Em) pmap[i]=Em+0.1;           // degrade loud pixels
        }
        count += VETO;                              // count live time
    }
//    printf("jE = %d, jB = %d, NN = %d, ib = %d, ie = %d \n", jE, jB, NN, ib, ie);

    for(jj=0; jj<NN; jj++) {                        // loop over time stamps

        pmap = MAP.data+(jj+jB)*I;                   // pointer to 0 F sample in MAP
        for(n=0; n<nIFO; n++) {
            if(IN[n] >= jE) IN[n] -= NN;              // go to jB sample
        }
        for(n=0; n<5; n++) pp[n]=pmap+(n-2)*I;       // initialize set of pointers

        for(i=ib; i<ie; i++) {
            if((E=pp[2][i])<Eo) continue;             // skip subthreshold pixels
            Ct = pp[2][i+1]+pp[3][ i ]+pp[3][i+1];    // top core
            Cb = pp[2][i-1]+pp[1][ i ]+pp[1][i-1];    // bottom core
            Ht = pp[4][i+1];                          // top halo
            Ht+= i<II? pp[4][i+2]+pp[3][i+2]:0.;      // top halo
            Hb = pp[0][i-1];                          // bottom halo
            Hb+= i>1 ? pp[0][i-2]+pp[1][i-2]:0.;      // bottom halo

            if((Ct+Cb)*E<Eh &&
               (Ct+Ht)*E<Eh &&
               (Cb+Hb)*E<Eh &&
               E<Em) continue;
            E = 0;
            for(n=0; n<nIFO; n++) {
                j = IN[n]*I+i;                          // sample index
                pix.data[n].index = j;
                pix.data[n].asnr = sqrt(pdata[n][j]);
                E += pdata[n][j];
            }
            j = IN[nM]*I+i;                            // reference sample index
            pix.time = j;
            pix.frequency = i;
            pix.likelihood = E;
            pix.phi = 1;                               // set pixel mark 1 (will be owerriden in likelihood)
            wc_List->append(pix);	            // save pixels in wc_List
            nPix++;
        }
        for(n=0; n<nIFO; n++) IN[n]++;                // increment IN
    }

// set metadata in wc_List
    wc_List->start = pTF->start();
    wc_List->stop  = pTF->stop();
    wc_List->rate  = pTF->rate();
    livTime = count/R;                    // live time depends on resolution

//    if(nPix) this->setRMS();
// TODO: setRMS afterwards
    printf("%ld, %f, size = %zu", nPix, livTime, wc_List->size());

//    return std::make_tuple(nPix, livTime, &wc_List);
    return wc_List;
}


double threshold(std::vector<WSeries<double>> tf_maps, int nIFO, double Edge, double p, double shape) {
// calculate WaveBurst energy threshold for a given black pixel probability p
// and single detector Gamma distribution shape. TF data should contain pixel energy
    int N = nIFO;
    WSeries<double>* pw = &tf_maps[0];
    size_t M  = pw->maxLayer()+1;
    size_t nL = size_t(Edge*pw->wrate()*M);
    size_t nR = pw->size() - nL - 1;
    wavearray<double> w = *pw;
    for(int i=1; i<N; i++) w += tf_maps[i];
    double amp, avr, bbb, alp;
    avr = bbb = 0.;
    int nn = 0;
    for(int i=nL; i<nR; i++) {                              // get Gamma shape & mean
        amp = (double)w.data[i];
        if(amp>N*100) amp = N*100.;
        if(amp>0.001) {avr+=amp; bbb+=log(amp); nn++;}
    }
    avr = avr/nn;                                           // Gamma mean
    alp = log(avr)-bbb/nn;
    alp = (3-alp+sqrt((alp-3)*(alp-3)+24*alp))/12./alp;     // Gamma shape
    bbb = p*alp/shape;                                      // corrected bpp
    //cout<<bbb<<" "<<avr<<" "<<alp<<" "<<shape<<" "<<iGamma(alp,bbb)<<endl;
    return avr*iGamma(alp,bbb)/alp/2;
}