#define CWB_CORE_LIKELIHOOD_CC

#include "likelihood.hh"

using namespace std;

long likelihoodWP(std::vector <netcluster> wc_List, size_t nIFO, std::vector <detector> ifo,
                  size_t skyMask_size, short *skyMask, std::vector<double> skyMaskCC,
                  skymap nSkyStat, skymap nSensitivity, skymap nAlignment, skymap nDisbalance,
                  skymap nLikelihood, skymap nNullEnergy, skymap nCorrEnergy, skymap nCorrelation,
                  skymap nEllipticity, skymap nPolarisation, skymap nNetIndex, skymap nAntenaPrior,
                  skymap nProbability, skymap tau,
                  monster wdmMRA, double netCC, bool EFEC,
                  double precision, double gamma, bool optim, double netRHO, double delta, double acor,
                  char mode, int lag, int iID, char *Search) {
//  Likelihood analysis with packets
//  skymask_size: this.index.size()
//  mode: analysis mode: 
//  OPTRES analyses, if upper case and optim=true
//  MRA analysis in low case or optim=false
//        r - un-modeled
//        i - iota - wave: no,partial dispersion correction
//        p - Psi - wave (no dispersion correction)
//      l,s - linear, loose linear
//      c,g - circular. loose circular
//      e,b - elliptical (no dispersion correction), b=p for now
//   iID: cluster ID, if negative - sky error regions are calculated 
//   lag: lag index
// hist: chirp histogram: If not needed, TGraphErrors* hist=NULL
// Search: if Search = ""/cbc/bbh/imbhb then mchirp is reconstructed
// shold be used as input
// return number of processed pixels
// Negative gamma regulator turns on the AP prior for sky localization
//
    size_t nRun = 0;
    std::vector < netpixel * > pList;
    wavearray<float> a_00;         //! buffer for cluster sky 00 amplitude
    wavearray<float> a_90;         //! buffer for cluster sky 90 amplitudes
    wavearray<float> rNRG;         //! buffers for cluster residual energy
    wavearray<float> pNRG;         //! buffers for cluster MRA energy
    wavearray<double> p00_POL[2]; //! buffer for projection on network plane 00 ampl
    wavearray<double> p90_POL[2]; //! buffer for projection on network plane 90 ampl
    wavearray<double> r00_POL[2]; //! buffer for standard response 00 ampl
    wavearray<double> r90_POL[2]; //! buffer for standard response 90 ampl
    wavearray<double> skyProb;     // sky probability

    if (!wc_List[lag].size()) return 0;

    // bool wdm = true;
    // this->tYPe = mode;

    bool cirwave = mode == 'g' || mode == 'G' || mode == 'c' || mode == 'C';
    bool linwave = mode == 'l' || mode == 'L' || mode == 's' || mode == 'S';
    bool iotwave = mode == 'i' || mode == 'l' || mode == 'e' || mode == 'c' ||
                   mode == 'I' || mode == 'L' || mode == 'E' || mode == 'C';
    bool psiwave = mode == 'l' || mode == 'e' || mode == 'p' ||
                   mode == 'L' || mode == 'E' || mode == 'P';
    bool mureana = mode == 'i' || mode == 'e' || mode == 'c' ||
                   mode == 'r' || mode == 'p' || mode == 'b' ||
                   mode == 'l' || mode == 's' || mode == 'g';
    bool rndwave = mode == 'r' || mode == 'R';

    bool prior = gamma < 0 ? true : false;     // gamma<0 : antenna pattern prior is used
    bool m_chirp = optim ? false : mureana;

    if (!optim) mureana = true;

    size_t ID = abs(iID);

    if (nIFO > NIFO) {
        cout << "network::likelihoodAVX(): invalid network.\n";
        exit(0);
    }

    float En = 2 * acor * acor * nIFO;                             // network energy threshold in the sky loop
    float gama = gamma * gamma * 2. / 3.;                // gamma regulator for x componet
    float deta = fabs(delta);
    if (deta > 1) deta = 1;       // delta regulator for + component
    float REG[2];
    REG[0] = deta * sqrt(2);
    float netEC = netRHO * netRHO * 2;                 // netEC/netRHO threshold

    static const __m128 _oo = _mm_set1_ps(1.e-16);             // nusance parameter
    static const __m128 _sm = _mm_set1_ps(-0.f);               // sign mask: -0.f = 1 << 31
    static const __m128 _En = _mm_set1_ps(En);                 // network threshold

    float aa, AA, Lo, Eo, Co, No, Ep, Lp, Np, Cp, Ec, Dc, To, Fo, Em, Lm, Rc, Mo, Mw, Eh;
    float STAT, ee, EE, cc, ff, FF, Lw, Ew, Cw, Nw, Gn, rho, norm, ch, CH, Cr, Mp, N;
    float penalty, ecor;    // used in the definition of XGB rho0 (XGB.rho0)
    float xrho;        // original 2G definition (XGB.rho0)

    size_t i, j, k, l, m, Vm, lm, V, V4, V44, id, K, M;
    size_t L = skyMask_size;             // total number of source locations
    wavearray<short> skyMMcc(L);
    short *mm = skyMask;
    short *MM = skyMMcc.data;
    bool skymaskcc = (skyMaskCC.size() == L);
    int f_ = NIFO / 4;

    float vvv[8];
    float *v00[NIFO];
    float *v90[NIFO];
    float *pe[NIFO];
    float *pa[NIFO];
    float *pA[NIFO];
    float *pd[NIFO], *pD[NIFO];
    float *ps[NIFO], *pS[NIFO];
    float *pn[NIFO], *pN[NIFO];
    short *ml[NIFO];
    double *FP[NIFO];
    double *FX[NIFO];
    double xx[NIFO];

    std::vector<float *> _vtd;              // vectors of TD amplitudes
    std::vector<float *> _vTD;              // vectors of TD amplitudes
    std::vector<float *> _eTD;              // vectors of TD energies
    std::vector<float *> _AVX;              // vectors for network pixel statistics
    std::vector<float *> _APN;              // vectors for noise and antenna patterns
    std::vector<float *> _DAT;              // vectors for packet amplitudes
    std::vector<float *> _SIG;              // vectors for packet amplitudes
    std::vector<float *> _NUL;              // vectors for packet amplitudes
    std::vector<float *> _TMP;              // temp array for _avx_norm_ps() function

    // TODO: remove the dependency on variable "ifo"
    for (i = 0; i < NIFO; i++) {
        if (i < nIFO) {
            ml[i] = ifo[i].index.data;
            FP[i] = ifo[i].fp.data;
            FX[i] = ifo[i].fx.data;
        } else {
            ml[i] = ifo[i].index.data;
            FP[i] = ifo[i].fp.data;
            FX[i] = ifo[i].fx.data;
        }
    }

    // allocate buffers
    std::vector<int> pI;                      // buffer for pixel IDs
    std::vector<int> pJ;                      // buffer for pixel index
    wavearray<double> cid;                    // buffers for cluster ID
    wavearray<double> cTo;                    // buffers for cluster time
    wavearray<float> S_snr(NIFO);            // energy SNR of signal
    wavearray<float> D_snr(NIFO);            // energy SNR of data time series
    wavearray<float> N_snr(NIFO);            // energy of null streams
    netpixel *pix;
    std::vector<int> *vint;
    std::vector<int> *vtof;
    netcluster *pwc = &wc_List[lag];

    size_t count = 0;
    size_t tsize = 0;

    std::map<int, float> vLr;             // resolution map

    // initialize parameters to manage big clusters 
    int precision_i = int(fabs(precision));
    int csize = precision_i % 65536;                 // get number of pixels threshold per level
    int healpix = nSkyStat.getOrder(); // get healpix order of likelihood skymap
    int order = (precision_i - csize) / 65536;      // get resampled order
    wavearray<short> BB(L);
    BB = 1;             // index array for setting sky mask
    bool bBB = false;
    if (healpix && csize && order && order < healpix) {
        skymap rsm(order);             // resampled skymap
        for (int l = 0; l < rsm.size(); l++) {
            int m = nSkyStat.getSkyIndex(rsm.getTheta(l), rsm.getPhi(l));
            BB[m] = 0;
        }
        for (int l = 0; l < L; l++) BB[l] = BB[l] ? 0 : 1;
    }

    cid = pwc->get((char *) "ID", 0, 'S', 0);                 // get cluster ID
    cTo = pwc->get((char *) "time", 0, 'L', 0);                 // get cluster time

    K = cid.size();

    //---------------------------------------------------------
    // start processing cluster
    //---------------------------------------------------------

    id = size_t(1);

    if (pwc->sCuts[id - 1] != -2) return 0;                // skip rejected/processed clusters

    // check if cluster is empty
    vint = &(pwc->cList[id - 1]);                         // pixel list
    vtof = &(pwc->nTofF[id - 1]);                         // TofFlight configurations
    V = vint->size();
    if (!V) return 0;

    // get cross-talk coefficients
    pI = wdmMRA.getXTalk(pwc, id);
    V = pI.size();
    if (!V) return 0;

    bBB = (V > wdmMRA.nRes * csize) ? true : false;      // check big cluster size condition

    // TODO: remove the id check
    if (ID == id) {
        nSensitivity = 0.;
        nAlignment = 0.;
        nNetIndex = 0.;
        nDisbalance = 0.;
        nLikelihood = 0.;
        nNullEnergy = 0.;
        nCorrEnergy = 0.;
        nCorrelation = 0.;
        nSkyStat = 0.;
        nEllipticity = 0.;
        nPolarisation = 0.;
        nProbability = 0.;
    }
    nAntenaPrior = 0.;


    pix = pwc->getPixel(id, pI[0]);
    tsize = pix->tdAmp[0].size();
    if (!tsize || tsize & 1) {                       // tsize%1 = 1/0 = power/amplitude
        cout << "network::likelihoodWP() error: wrong pixel TD data\n";
        exit(1);
    }

    tsize /= 2;

    // redundant code
    // if (!(V = pI.size())) return 0;

    V4 = V + (V % 4 ? 4 - V % 4 : 0);
    V44 = V4 + 4;
    pJ.clear();
    for (j = 0; j < V4; j++) pJ.push_back(0);

    //    float *ptmp;                                     // allocate aligned arrays
    //    if (_vtd.size()) _avx_free_ps(_vtd);              // array for 00 amplitudes
    //    if (_vTD.size()) _avx_free_ps(_vTD);              // array for 90 amplitudes
    //    if (_eTD.size()) _avx_free_ps(_eTD);              // array for pixel energy
    //    if (_APN.size()) _avx_free_ps(_APN);              // container for noise rms and antenna patterns
    //    if (_DAT.size()) _avx_free_ps(_DAT);              // container for data packet amplitudes
    //    if (_SIG.size()) _avx_free_ps(_SIG);              // container for signal packet amplitudes
    //    if (_NUL.size()) _avx_free_ps(_NUL);              // container for null packet amplitudes
    //    for (i = 0; i < NIFO; i++) {
    //        ptmp = (float *) _mm_malloc(tsize * V4 * sizeof(float), 32);
    //        for (j = 0; j < tsize * V4; j++) ptmp[j] = 0;
    //        _vtd.push_back(ptmp);   // array of aligned vectors
    //        ptmp = (float *) _mm_malloc(tsize * V4 * sizeof(float), 32);
    //        for (j = 0; j < tsize * V4; j++) ptmp[j] = 0;
    //        _vTD.push_back(ptmp);   // array of aligned vectors
    //        ptmp = (float *) _mm_malloc(tsize * V4 * sizeof(float), 32);
    //        for (j = 0; j < tsize * V4; j++) ptmp[j] = 0;
    //        _eTD.push_back(ptmp);   // array of aligned vectors
    //        ptmp = (float *) _mm_malloc((V4 * 3 + 16) * sizeof(float), 32);
    //        for (j = 0; j < (V4 * 3 + 16); j++) ptmp[j] = 0;
    //        _APN.push_back(ptmp);  // concatenated arrays {f+}{fx}{rms}{a+,A+,ax,AX}
    //        ptmp = (float *) _mm_malloc((V4 * 3 + 8) * sizeof(float), 32);
    //        for (j = 0; j < (V4 * 3 + 8); j++) ptmp[j] = 0;
    //        _DAT.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
    //        ptmp = (float *) _mm_malloc((V4 * 3 + 8) * sizeof(float), 32);
    //        for (j = 0; j < (V4 * 3 + 8); j++) ptmp[j] = 0;
    //        _SIG.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
    //        ptmp = (float *) _mm_malloc((V4 * 3 + 8) * sizeof(float), 32);
    //        for (j = 0; j < (V4 * 3 + 8); j++) ptmp[j] = 0;
    //        _NUL.push_back(ptmp);   // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
    //    }

    // Free previous memory allocations
    free_if_not_empty(_vtd);
    free_if_not_empty(_vTD);
    free_if_not_empty(_eTD);
    free_if_not_empty(_APN);
    free_if_not_empty(_DAT);
    free_if_not_empty(_SIG);
    free_if_not_empty(_NUL);

    // Allocate new memory and push to the containers
    for (int i = 0; i < NIFO; i++) {
        _vtd.push_back(alloc_and_init(tsize * V4));  // array of aligned vectors
        _vTD.push_back(alloc_and_init(tsize * V4));  // array of aligned vectors
        _eTD.push_back(alloc_and_init(tsize * V4));  // array of aligned vectors
        _APN.push_back(alloc_and_init(V4 * 3 + 16)); // concatenated arrays {f+}{fx}{rms}{a+,A+,ax,AX}
        _DAT.push_back(alloc_and_init(V4 * 3 + 8));  // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
        _SIG.push_back(alloc_and_init(V4 * 3 + 8));  // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
        _NUL.push_back(alloc_and_init(V4 * 3 + 8));  // concatenated arrays {amp}{AMP}{norm}{n,N,c,s}
    }

    // data arrays for polar coordinates storage : [0,1] = [radius,angle]
    for (i = 0; i < 2; i++) {
        p00_POL[i].resize(V4);
        p00_POL[i] = 0.;
        p90_POL[i].resize(V4);
        p90_POL[i] = 0.;
        r00_POL[i].resize(V4);
        r00_POL[i] = 0.;
        r90_POL[i].resize(V4);
        r90_POL[i] = 0.;
    }

    // set up zero delay and packet pointers
    for (i = 0; i < NIFO; i++) {
        pa[i] = _vtd[i] + (tsize / 2) * V4;
        pA[i] = _vTD[i] + (tsize / 2) * V4;
        pe[i] = _eTD[i] + (tsize / 2) * V4;
        pd[i] = _DAT[i];
        pD[i] = _DAT[i] + V4;
        ps[i] = _SIG[i];
        pS[i] = _SIG[i] + V4;
        pn[i] = _NUL[i];
        pN[i] = _NUL[i] + V4;
    }

    a_00.resize(NIFO * V44);
    a_00 = 0.;      // array for pixel amplitudes in sky loop
    a_90.resize(NIFO * V44);
    a_90 = 0.;      // array for pixel amplitudes in sky loop
    rNRG.resize(V4);
    rNRG = 0.;
    pNRG.resize(V4);
    pNRG = 0.;

//    __m128 *_aa = (__m128 *) a_00.data;         // set pointer to 00 array
//    __m128 *_AA = (__m128 *) a_90.data;         // set pointer to 90 array
//
//    if (_AVX.size()) _avx_free_ps(_AVX);
//    float *p_et = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 0
//    for (j = 0; j < V4; j++) p_et[j] = 0;
//    _AVX.push_back(p_et);
//    float *pMSK = (float *) _mm_malloc(V44 * sizeof(float), 32);     // 1  - pixel mask
//    for (j = 0; j < V44; j++) pMSK[j] = 0;
//    _AVX.push_back(pMSK);
//    pMSK[V4] = nIFO;
//    float *p_fp = (float *) _mm_malloc(V44 * sizeof(float), 32);     // 2- |f+|^2 (0:V4), +norm (V4:V4+4)
//    for (j = 0; j < V44; j++) p_fp[j] = 0;
//    _AVX.push_back(p_fp);
//    float *p_fx = (float *) _mm_malloc(V44 * sizeof(float), 32);     // 3- |fx|^2 (0:V4), xnorm (V4:V4+4)
//    for (j = 0; j < V44; j++) p_fx[j] = 0;
//    _AVX.push_back(p_fx);
//    float *p_si = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 4
//    for (j = 0; j < V4; j++) p_si[j] = 0;
//    _AVX.push_back(p_si);
//    float *p_co = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 5
//    for (j = 0; j < V4; j++) p_co[j] = 0;
//    _AVX.push_back(p_co);
//    float *p_uu = (float *) _mm_malloc((V4 + 16) * sizeof(float),
//                                       32); // 6 - 00+ unit vector(0:V4), norm(V4), cos(V4+4)
//    for (j = 0; j < V4 + 16; j++) p_uu[j] = 0;
//    _AVX.push_back(p_uu);
//    float *p_UU = (float *) _mm_malloc((V4 + 16) * sizeof(float),
//                                       32); // 7 - 90+ unit vector(0:V4), norm(V4), sin(V4+4)
//    for (j = 0; j < V4 + 16; j++) p_UU[j] = 0;
//    _AVX.push_back(p_UU);
//    float *p_vv = (float *) _mm_malloc((V4 + 16) * sizeof(float),
//                                       32); // 8- 00x unit vector(0:V4), norm(V4), cos(V4+4)
//    for (j = 0; j < V4 + 16; j++) p_vv[j] = 0;
//    _AVX.push_back(p_vv);
//    float *p_VV = (float *) _mm_malloc((V4 + 16) * sizeof(float),
//                                       32); // 9- 90x unit vector(0:V4), norm(V4), sin(V4+4)
//    for (j = 0; j < V4 + 16; j++) p_VV[j] = 0;
//    _AVX.push_back(p_VV);
//    float *p_au = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 10
//    for (j = 0; j < V4; j++) p_au[j] = 0;
//    _AVX.push_back(p_au);
//    float *p_AU = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 11
//    for (j = 0; j < V4; j++) p_AU[j] = 0;
//    _AVX.push_back(p_AU);
//    float *p_av = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 12
//    for (j = 0; j < V4; j++) p_av[j] = 0;
//    _AVX.push_back(p_av);
//    float *p_AV = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 13
//    for (j = 0; j < V4; j++) p_AV[j] = 0;
//    _AVX.push_back(p_AV);
//    float *p_uv = (float *) _mm_malloc(V4 * 4 * sizeof(float), 32);    // 14 special array for GW norm calculation
//    for (j = 0; j < V4 * 4; j++) p_uv[j] = 0;
//    _AVX.push_back(p_uv);
//    float *p_ee = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 15 + energy array
//    for (j = 0; j < V4; j++) p_ee[j] = 0;
//    _AVX.push_back(p_ee);
//    float *p_EE = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 16 x energy array
//    for (j = 0; j < V4; j++) p_EE[j] = 0;
//    _AVX.push_back(p_EE);
//    float *pTMP = (float *) _mm_malloc(V4 * 4 * NIFO * sizeof(float), 32); // 17 temporary array for _avx_norm_ps()
//    for (j = 0; j < V4 * 4 * NIFO; j++) pTMP[j] = 0;
//    _AVX.push_back(pTMP);
//    float *p_ni = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 18 + network index
//    for (j = 0; j < V4; j++) p_ni[j] = 0;
//    _AVX.push_back(p_ni);
//    float *p_ec = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 19 + coherent energy
//    for (j = 0; j < V4; j++) p_ec[j] = 0;
//    _AVX.push_back(p_ec);
//    float *p_gn = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 20 + Gaussian noise correction
//    for (j = 0; j < V4; j++) p_gn[j] = 0;
//    _AVX.push_back(p_gn);
//    float *p_ed = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 21 + energy disbalance
//    for (j = 0; j < V4; j++) p_ed[j] = 0;
//    _AVX.push_back(p_ed);
//    float *p_rn = (float *) _mm_malloc(V4 * sizeof(float), 32);      // 22 + sattelite noise in TF domain
//    for (j = 0; j < V4; j++) p_rn[j] = 0;
//    _AVX.push_back(p_rn);

    // Set pointer to arrays
    __m128 *_aa = (__m128 *) a_00.data;
    __m128 *_AA = (__m128 *) a_90.data;

    // Free previous memory allocations
    free_if_not_empty(_AVX);

    // Allocate new memory, initialize, and push to the containers
    float *p_et = alloc_and_init(V4);
    _AVX.push_back(p_et); // 0
    float *pMSK = alloc_and_init(V44);
    _AVX.push_back(pMSK); // 1  - pixel mask
    pMSK[V4] = nIFO;
    float *p_fp = alloc_and_init(V44); // 2- |f+|^2 (0:V4), +norm (V4:V4+4)
    _AVX.push_back(p_fp);
    float *p_fx = alloc_and_init(V44); // 3- |fx|^2 (0:V4), xnorm (V4:V4+4)
    _AVX.push_back(p_fx);
    float *p_si = alloc_and_init(V4);
    _AVX.push_back(p_si); // 4
    float *p_co = alloc_and_init(V4);
    _AVX.push_back(p_co); // 5
    float *p_uu = alloc_and_init(V4 + 16);
    _AVX.push_back(p_uu); // 6 - 00+ unit vector(0:V4), norm(V4), cos(V4+4)
    float *p_UU = alloc_and_init(V4 + 16);
    _AVX.push_back(p_UU); // 7 - 90+ unit vector(0:V4), norm(V4), sin(V4+4)
    float *p_vv = alloc_and_init(V4 + 16);
    _AVX.push_back(p_vv); // 8- 00x unit vector(0:V4), norm(V4), cos(V4+4)
    float *p_VV = alloc_and_init(V4 + 16);
    _AVX.push_back(p_VV); // 9- 90x unit vector(0:V4), norm(V4), sin(V4+4)
    float *p_au = alloc_and_init(V4);
    _AVX.push_back(p_au); // 10
    float *p_AU = alloc_and_init(V4);
    _AVX.push_back(p_AU); // 11
    float *p_av = alloc_and_init(V4);
    _AVX.push_back(p_av); // 12
    float *p_AV = alloc_and_init(V4);
    _AVX.push_back(p_AV); // 13
    float *p_uv = alloc_and_init(V4 * 4);
    _AVX.push_back(p_uv); // 14 special array for GW norm calculation
    float *p_ee = alloc_and_init(V4);
    _AVX.push_back(p_ee); // 15 + energy array
    float *p_EE = alloc_and_init(V4);
    _AVX.push_back(p_EE); // 16 x energy array
    float *pTMP = alloc_and_init(V4 * 4 * NIFO);
    _AVX.push_back(pTMP); // 17 temporary array for _avx_norm_ps()
    float *p_ni = alloc_and_init(V4);
    _AVX.push_back(p_ni); // 18 + network index
    float *p_ec = alloc_and_init(V4);
    _AVX.push_back(p_ec); // 19 + coherent energy
    float *p_gn = alloc_and_init(V4);
    _AVX.push_back(p_gn); // 20 + Gaussian noise correction
    float *p_ed = alloc_and_init(V4);
    _AVX.push_back(p_ed); // 21 + energy disbalance
    float *p_rn = alloc_and_init(V4);
    _AVX.push_back(p_rn); // 22 + sattelite noise in TF domain

    //---------------------------------------------------------
    // loop over selected pixels
    //---------------------------------------------------------
    pList.clear();
    for (j = 0; j < V; j++) {
        pix = pwc->getPixel(id, pI[j]);
        pList.push_back(pix);                   // store pixel pointers for MRA

        double rms = 0.;
        for (i = 0; i < nIFO; i++) {
            xx[i] = 1. / pix->data[i].noiserms;
            rms += xx[i] * xx[i];                        // total inverse variance
        }

        rms = sqrt(rms);
        for (i = 0; i < nIFO; i++) {
            _APN[i][V4 * 2 + j] = (float) xx[i] / rms;        // noise array for AVX processing
            for (l = 0; l < tsize; l++) {
                aa = pix->tdAmp[i].data[l];             // copy TD 00 data
                AA = pix->tdAmp[i].data[l + tsize];       // copy TD 90 data
                _vtd[i][l * V4 + j] = aa;                   // copy 00 data
                _vTD[i][l * V4 + j] = AA;                   // copy 90 data
                _eTD[i][l * V4 + j] = aa * aa + AA * AA;          // copy power
            }
        }
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // sky loop
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    __m256 _CC;
    size_t lb = 0;
    size_t le = L - 1;
    double sky = 0.;
    STAT = -1.e12;
    lm = 0;
    Em = Lm = ff = FF = 0;

    // Notes: MM, FF, aa are all temporary variables
    // TODO: This section is to set REG? Here can be one function
    skyMMcc = 0;
    for (l = lb; l <= le; l++) {                            // loop over sky locations
        if (!mm[l]) continue;                           // skip delay configurations
        if (bBB && !BB[l]) continue;                    // skip delay configurations : big clusters

        if (skymaskcc) {                                // transform l into celestial coordinates lc
            skymap *sm = &(nSkyStat);
            double gT = cTo.data[k] + pwc->start;          // trigger gps time
            double RA = sm->phi2RA(sm->getPhi(l), gT);    // phi -> RA
            int lc = tau.getSkyIndex(sm->getTheta(l), RA);   // get sky index in celestial coordinates
            if (!skyMaskCC[lc]) continue;
        }
        MM[l] = 1;
        FF += 1;                            // set final skymap
        aa = _avx_dpf_ps(FP, FX, l, _APN, _AVX, V4);        // calculate DPF f+,fx and their norms
        if (aa > gama) ff += 1;
    }
    REG[1] = (FF * FF / (ff * ff + 1.e-9) - 1) * En;              // setup x regulator

    // TODO: what's the use of the do-while loop?
    do {
        AA = 0.;                                          // initialize sky statistic
        for (l = lb; l <= le; l++) {                            // loop over sky locations
            skyProb.data[l] = -1.e12;
            if (!MM[l]) continue;                           // apply sky mask
            // TODO: here pa(_vtd, v00), pA(_vTD, v90) these are (not) the same variables, a_00, a_90 are copied from these two variables
            // TODO: v00 and v90 are the time delay data at position l, pa and pA are the full data
            // TODO: ml is not zero for ifo >= 1, network.setDelayIndex

            pnt_(v00, pa, ml, (int) l, (int) V4);            // pointers to first pixel 00 data
            pnt_(v90, pA, ml, (int) l, (int) V4);            // pointers to first pixel 90 data
            Eo = _avx_loadata_ps(v00, v90, pd, pD, En, _AVX, V4);  // calculate data stats and store in _AVX

            _avx_dpf_ps(FP, FX, l, _APN, _AVX, V4);             // calculate DPF f+,fx and their norms
            _avx_cpf_ps(v00, v90, ps, pS, V4);                 // copy data for GW reconstruction
            Mo = _avx_GW_ps(ps, pS, _APN, REG, _AVX, V4);       // gw strain packet, return number of selected pixels

            // if (lb == le) _avx_saveGW_ps(ps, pS, V);            // save gw strain packet into a_00,a_90
            if (lb == le) {
                for (int i = 0; i < V; i++) {
                    for (int n = 0; n < NIFO; n++) {
                        a_00[i * NIFO + n] = ps[n][i];
                        a_90[i * NIFO + n] = pS[n][i];
                    }
                }
            }

            Lo = _avx_ort_ps(ps, pS, _AVX, V4);              // othogonalize signal amplitudes
            _CC = _avx_stat_ps(pd, pD, ps, pS, _AVX, V4);       // coherent statistics
            _mm256_storeu_ps(vvv, _CC);                     // extract coherent statistics
            Cr = vvv[0];                                   // cc statistics
            Ec = vvv[1];                                   // signal coherent energy in TF domain
            Mp = vvv[2];                                   // signal energy disbalance in TF domain
            No = vvv[3];                                   // total noise in TF domain
            CH = No / (nIFO * Mo + sqrt(Mo));                    // chi2 in TF domain
            cc = CH > 1 ? CH : 1;                            // noise correction factor in TF domain
            Co = Ec / (Ec + No * cc - Mo * (nIFO - 1));                // network correlation coefficient in TF

            if (Cr < netCC) continue;

            aa = Eo > 0. ? Eo - No : 0.;                        // likelihood skystat
            AA = aa * Co;                                     // x-correlation skystat
            skyProb.data[l] = delta < 0 ? aa : AA;

            ff = FF = ee = 0.;
            for (j = 0; j < V; j++) {
                if (pMSK[j] <= 0) continue;
                ee += p_et[j];                             // total energy
                ff += p_fp[j] * p_et[j];                     // |f+|^2
                FF += p_fx[j] * p_et[j];                     // |fx|^2
            }
            ff = ee > 0. ? ff / ee : 0.;
            FF = ee > 0. ? FF / ee : 0.;
            nAntenaPrior.set(l, sqrt(ff + FF));

            if (ID == id) {
                nSensitivity.set(l, sqrt(ff + FF));
                nAlignment.set(l, ff > 0 ? sqrt(FF / ff) : 0);
                nLikelihood.set(l, Eo - No);
                nNullEnergy.set(l, No);
                nCorrEnergy.set(l, Ec);
                nCorrelation.set(l, Co);
                nSkyStat.set(l, AA);
                nProbability.set(l, skyProb.data[l]);
                nDisbalance.set(l, CH);
                nNetIndex.set(l, cc);
                nEllipticity.set(l, Cr);
                nPolarisation.set(l, Mp);
            }

            // TODO: find the optimal sky location
            if (AA >= STAT) {
                STAT = AA;
                lm = l;
                Em = Eo - Eh;
            }
            if (skyProb.data[l] > sky) sky = skyProb.data[l];            // find max of skyloc stat

            if (lb != le) continue;

            Eo = _avx_packet_ps(pd, pD, _AVX, V4);            // get data packet
            Lo = _avx_packet_ps(ps, pS, _AVX, V4);            // get signal packet
            D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, V4);           // data packet energy snr
            S_snr = _avx_norm_ps(pS, pD, p_ec, V4);           // set signal norms, return signal SNR
            Ep = D_snr[0];
            Lp = S_snr[0];

            _CC = _avx_noise_ps(pS, pD, _AVX, V4);            // get G-noise correction
            _mm256_storeu_ps(vvv, _CC);                     // extract coherent statistics
            Gn = vvv[0];                                   // gaussian noise correction
            Ec = vvv[1];                                   // core coherent energy in TF domain
            Dc = vvv[2];                                   // signal-core coherent energy in TF domain
            Rc = vvv[3];                                   // EC normalization
            Eh = vvv[4];                                   // satellite energy in TF domain

            N = _avx_setAMP_ps(pd, pD, _AVX, V4) - 1;           // set data packet amplitudes

            _avx_setAMP_ps(ps, pS, _AVX, V4);                 // set signal packet amplitudes
            _avx_loadNULL_ps(pn, pN, pd, pD, ps, pS, V4);        // load noise TF domain amplitudes
            D_snr = _avx_norm_ps(wdmMRA, pd, pD, _AVX, -V4);          // data packet energy snr
            N_snr = _avx_norm_ps(wdmMRA, pn, pN, _AVX, -V4);          // noise packet energy snr
            Np = N_snr.data[0];                            // time-domain NULL
            Em = D_snr.data[0];                            // time domain energy
            Lm = Em - Np - Gn;                                 // time domain signal energy
            norm = Em > 0 ? (Eo - Eh) / Em : 1.e9;               // norm
            if (norm < 1) norm = 1;                           // corrected norm
            Ec /= norm;                                   // core coherent energy in time domain
            Dc /= norm;                                   // signal-core coherent energy in time domain
            ch = (Np + Gn) / (N * nIFO);                         // chi2
            if (netRHO >= 0) {    // original 2G
                cc = ch > 1 ? ch : 1;                          // rho correction factor
                rho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;           // cWB detection stat
            } else {        // (XGB.rho0)
                penalty = ch;
                ecor = Ec;
                rho = sqrt(ecor / (1 + penalty * (max((float) 1., penalty) - 1)));
                // original 2G rho statistic: only for test
                cc = ch > 1 ? ch : 1;                          // rho correction factor
                xrho = Ec > 0 ? sqrt(Ec * Rc / 2.) : 0.;          // cWB detection stat
            }
            // save projection on network plane in polar coordinates
            // The Dual Stream Transform (DSP) is applied to v00,v90
            _avx_pol_ps(v00, v90, p00_POL, p90_POL, _APN, _AVX, V4);
            // save DSP components in polar coordinates
            _avx_pol_ps(v00, v90, r00_POL, r90_POL, _APN, _AVX, V4);
        }
        lb = le = lm;
    } while (le - lb); // process all pixels at opt sky location

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // reject cluster if detection statistics are below threshold
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (netRHO >= 0) {    // original 2G
        if (Lm <= 0. || (Eo - Eh) <= 0. || Ec * Rc / cc < netEC || N < 1) {
            pwc->sCuts[id - 1] = 1;
            count = 0;                   // reject cluster
            pwc->clean(id);
            return 0;
        }
    } else {            // (XGB.rho0)
        if (Lm <= 0. || (Eo - Eh) <= 0. || rho < fabs(netRHO) || N < 1) {
            pwc->sCuts[id - 1] = 1;
            count = 0;                   // reject cluster
            pwc->clean(id);
            return 0;
        }
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // detection statistics at selected sky location
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    vint = &(pwc->cList[id - 1]);                       // pixel list
    for (j = 0; j < vint->size(); j++) {                   // initialization for all pixels
        pix = pwc->getPixel(id, j);
        pix->core = false;
        pix->likelihood = 0.;
        pix->null = 0;
    }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // detection statistics at selected sky location
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    M = Mw = 0;                                        // add denoised pixels
    for (j = 0; j < V; j++) {                               // loop over pixels
        pix = pwc->getPixel(id, pI[j]);
        if (pMSK[j] > 0) {                                 // Mo - EP pixels: stored in size[0]
            pix->core = true;
            // TODO: p_ee and p_EE are not initialized anywhere? Ans: set in _avx_ort_ps
            pix->likelihood = -(p_ee[j] + p_EE[j]) / 2;      // negative total pixel energy
        }

        for (i = 0; i < nIFO; i++) {
            pix->setdata(double(pd[i][j]), 'W', i);        // 00 whitened
            pix->setdata(double(pD[i][j]), 'U', i);        // 90 whitened
            pix->setdata(double(ps[i][j]), 'S', i);        // 00 reconstructed whitened response
            pix->setdata(double(pS[i][j]), 'P', i);        // 90 reconstructed whitened response
        }
    }

    for (j = 0; j < V; j++) {                               // loop over pixels
        pix = pwc->getPixel(id, pI[j]);
        if (!pix->core) continue;
        if (p_gn[j] <= 0) continue;                        // skip satellites
        Mw += 1.;                                       // event size stored in size[1]
        for (k = 0; k < V; k++) {                            // loop over xtalk components
            netpixel *xpix = pwc->getPixel(id, pI[k]);
            struct xtalk xt = wdmMRA.getXTalk(pix->layers, pix->time, xpix->layers, xpix->time);
            if (!xpix->core || p_gn[k] <= 0 || xt.CC[0] > 2) continue;
            for (i = 0; i < nIFO; i++) {
                pix->null += xt.CC[0] * pn[i][j] * pn[i][k];
                pix->null += xt.CC[1] * pn[i][j] * pN[i][k];
                pix->null += xt.CC[2] * pN[i][j] * pn[i][k];
                pix->null += xt.CC[3] * pN[i][j] * pN[i][k];
            }
        }

        if (p_ec[j] <= 0) continue;                         // skip incoherent pixels
        M += 1;                                          // M - signal size: stored in volume[1]
        pix->likelihood = 0;                             // total pixel energy
        for (k = 0; k < V; k++) {                             // loop over xtalk components
            netpixel *xpix = pwc->getPixel(id, pI[k]);
            struct xtalk xt = wdmMRA.getXTalk(pix->layers, pix->time, xpix->layers, xpix->time);
            if (!xpix->core || p_ec[k] <= 0 || xt.CC[0] > 2) continue;
            for (i = 0; i < nIFO; i++) {
                pix->likelihood += xt.CC[0] * ps[i][j] * ps[i][k];
                pix->likelihood += xt.CC[1] * ps[i][j] * pS[i][k];
                pix->likelihood += xt.CC[2] * pS[i][j] * ps[i][k];
                pix->likelihood += xt.CC[3] * pS[i][j] * pS[i][k];
            }
        }
    }
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // subnetwork statistic
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    double Emax = 0;
    double Nmax = 0;
    for (j = 1; j <= nIFO; j++) {                            // loop over detectors
        if (S_snr[j] > Emax) Emax = S_snr[j];                 // detector with max energy
    }
    double Esub = S_snr.data[0] - Emax;
    Esub = Esub * (1 + 2 * Rc * Esub / Emax);
    Nmax = Gn + Np - N * (nIFO - 1);

    //if(hist) hist->Fill(pwc->cData[id-1].skycc,pwc->cData[id-1].netcc);

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // fill in detection statistics, prepare output data
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    // fill in backward delay configuration

    vtof->clear();
    NETX(vtof->push_back(ml[0][lm]); ,
    vtof->push_back(ml[1][lm]); ,
    vtof->push_back(ml[2][lm]); ,
    vtof->push_back(ml[3][lm]); ,
    vtof->push_back(ml[4][lm]); ,
    vtof->push_back(ml[5][lm]); ,
    vtof->push_back(ml[6][lm]); ,
    vtof->push_back(ml[7][lm]); )

    // need to fix a problem below
//        if((wfsave)||(mdcListSize() && !lag)) {     // if wfsave=false only simulated wf are saved
//            if(this->getMRAwave(id,lag,'S',0,true)) { // reconstruct whitened shifted pd->waveForm
//                detector* pd;
//                for(i=0; i<nIFO; i++) {                 // loop over detectors
//                    pd = ifo[i];
//                    pd->RWFID.push_back(id);              // save cluster ID
//                    WSeries<double>* wf = new WSeries<double>;
//                    *wf = pd->waveForm;
//                    wf->start(pwc->start+pd->waveForm.start());
//                    pd->RWFP.push_back(wf);
//                }
//            }
//            if(this->getMRAwave(id,lag,'s',0,true)) { // reconstruct strain shifted pd->waveForm
//                detector* pd;
//                for(i=0; i<nIFO; i++) {                 // loop over detectors
//                    pd = ifo[i];
//                    pd->RWFID.push_back(-id);             // save cluster -ID
//                    WSeries<double>* wf = new WSeries<double>;
//                    *wf = pd->waveForm;
//                    wf->start(pwc->start+pd->waveForm.start());
//                    pd->RWFP.push_back(wf);
//                }
//            }
//        }

    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // TODO: calculate with python to avoid getMRAwave
    //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//    Lw = Ew = To = Fo = Nw = ee = norm = 0.;
//    for (i = 0; i < nIFO; i++) {
//        detector *d = &ifo[i];
//        d->sSNR = d->xSNR = d->null = d->enrg = 0.;
//    }
//
//    this->getMRAwave(id, lag, 'W', 0);
//    this->getMRAwave(id, lag, 'S', 0);
//    for (i = 0; i < nIFO; i++) {
//        detector *d = &ifo[i];
//        d->waveNull = d->waveBand;
//        d->waveNull -= d->waveForm;
//        float sSNR = d->get_SS();
//        float xSNR = d->get_XS();
//        float null = d->get_NN();
//        float enrg = d->get_XX();
//        d->sSNR += sSNR;
//        d->xSNR += xSNR;
//        d->null += null;
//        d->enrg += enrg;
//        To += sSNR * d->getWFtime();
//        Fo += sSNR * d->getWFfreq();
//        Lw += sSNR;
//        Ew += enrg;
//        Nw += null;
//    }
//    To /= Lw;
//    Fo /= Lw;
//    ch = (Nw + Gn) / (N * nIFO);                               // chi2
//    cc = ch > 1 ? 1 + (ch - 1) * 2 * (1 - Rc) : 1;                   // Cr correction factor
//    Cr = Ec * Rc /
//         (Ec * Rc + (Dc + Nw + Gn) * cc - N * (nIFO - 1));         // reduced network correlation coefficient
//    cc = ch > 1 ? ch : 1;                                  // rho correction factor
//    Cp = Ec * Rc / (Ec * Rc + (Dc + Nw + Gn) - N * (nIFO - 1));            // network correlation coefficient
//    norm = (Eo - Eh) / Ew;
//
//    pwc->cData[id - 1].norm = norm * 2;                     // packet norm  (saved in norm)
//    pwc->cData[id - 1].skyStat = 0;                          //
//    pwc->cData[id - 1].skySize = Mw;                         // event size in the skyloop    (size[1])
//    pwc->cData[id - 1].netcc = Cp;                         // network cc                   (netcc[0])
//    pwc->cData[id - 1].skycc = Cr;                         // reduced network cc           (netcc[1])
//    pwc->cData[id - 1].subnet = Esub / (Esub + Nmax);           // sub-network statistic        (netcc[2])
//    pwc->cData[id - 1].SUBNET = Co;                         // sky cc                       (netcc[3])
//    pwc->cData[id - 1].likenet = Lw;                         // waveform likelihood
//    pwc->cData[id - 1].netED = Nw + Gn + Dc - N * nIFO;            // residual NULL energy         (neted[0])
//    pwc->cData[id - 1].netnull = Nw + Gn;                      // packet NULL                  (neted[1])
//    pwc->cData[id - 1].energy = Ew;                         // energy in time domain        (neted[2])
//    pwc->cData[id - 1].likesky = Em;                         // energy in the loop           (neted[3])
//    pwc->cData[id - 1].enrgsky = Eo;                         // TF-domain all-res energy     (neted[4])
//    pwc->cData[id -
//               1].netecor = Ec;                         // packet (signal) coherent energy
//    pwc->cData[id - 1].normcor = Ec * Rc;                      // normalized coherent energy
//    if (netRHO >= 0) {    // original 2G
//        pwc->cData[id - 1].netRHO = rho / sqrt(cc);             // reduced rho - stored in rho[0]
//        pwc->cData[id - 1].netrho = rho;                      // chirp rho   - stored in rho[1]
//    } else {            // (XGB.rho0)
//        pwc->cData[id -
//                   1].netRHO = -rho;                     // reduced rho - stored in rho[0] with negative value in order to inform netevent.cc that it is XGB.rho0
//        pwc->cData[id - 1].netrho =
//                xrho / sqrt(cc);            // original 2G rho - stored in rho[1], only for test
//    }
//    pwc->cData[id - 1].cTime = To;
//    pwc->cData[id - 1].cFreq = Fo;
//    pwc->cData[id - 1].theta = nLikelihood.getTheta(lm);
//    pwc->cData[id - 1].phi = nLikelihood.getPhi(lm);
//    pwc->cData[id - 1].gNET = sqrt(ff + FF);
//    pwc->cData[id - 1].aNET = sqrt(FF / ff);
//    pwc->cData[id - 1].iNET = 0;                          // degrees of freedom
//    pwc->cData[id - 1].nDoF = N;                          // degrees of freedom
//    pwc->cData[id - 1].skyChi2 = CH;
//    pwc->cData[id - 1].Gnoise = Gn;
//    pwc->cData[id - 1].iota = 0;
//    pwc->cData[id - 1].psi = 0;
//    pwc->cData[id - 1].ellipticity = 0.;
//
//    cc = pwc->cData[id - 1].netcc;

//        if(hist) {
//            printf("rho=%4.2f|%4.2f cc: %5.3f|%5.3f|%5.3f subnet=%4.3f|%4.3f \n",
//                   rho,rho*sqrt(Cp),Co,Cp,Cr,pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
//            printf(" L: %5.1f|%5.1f|%5.1f E: %5.1f|%5.1f|%5.1f|%5.1f N: %4.1f|%4.1f|%4.1f|%4.1f|%4.1f \n",
//                   Lw,Lp,Lo,Ew,Ep,Eo,Em,Nw,Np,Rc,Eh,No);
//            printf("id|lm %3d|%6d  Vm|m=%3d|%3d|%3d|%3d T|F: %6.3f|%4.1f (t,p)=(%4.1f|%4.1f) \n",
//                   int(id),int(lm),int(V),int(Mo),int(Mw),int(M),To,Fo,nLikelihood.getTheta(lm),nLikelihood.getPhi(lm));
//            cout<<" L: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",S_snr[i]);}
//            cout<<" E: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",D_snr[i]);}
//            cout<<" N: |"; for(i=1; i<nIFO+1; i++) {printf("%5.1f|",N_snr[i]);}
//            cout<<endl<<" dof|G|G+R "; printf("%5.1f|%5.1f|%5.1f r[1]=%4.1f",N,Gn,Nw+Gn,REG[1]);
//            printf(" norm=%3.1f chi2 %3.2f|%3.2f Rc=%3.2f, Dc=%4.1f\n",norm,ch,CH,Rc,Dc);
//            //     cout<<" r1="<<REG[1]<<" norm="<<norm<<" chi2="<<ch<<"|"<<CH<<" Rc="<<Rc<<" Dc="<<Dc<<endl;
//            //hist->Fill(pwc->cData[id-1].subnet,pwc->cData[id-1].SUBNET);
//        }
    count++;

// calculation of error regions

    pwc->p_Ind[id - 1].push_back(Mo);
    double T = To + pwc->start;                          // trigger time
    std::vector<float> sArea;
    pwc->sArea.push_back(sArea);
    pwc->p_Map.push_back(sArea);

    double var = norm * Rc * sqrt(Mo) * (1 + fabs(1 - CH));

    // TODO: fix this
//    if (iID <= 0 || ID == id) {
//        network::getSkyArea(id, lag, T, var);       // calculate error regions
//    }

    // calculation of chirp mass

    pwc->cData[id - 1].mchirp = 0;
    pwc->cData[id - 1].mchirperr = 0;
    pwc->cData[id - 1].tmrgr = 0;
    pwc->cData[id - 1].tmrgrerr = 0;
    pwc->cData[id - 1].chi2chirp = 0;

    // It works only for MRA.
    if (m_chirp) {
        if (netRHO >= 0) {
            ee = pwc->mchirp(id);        // original mchirp 2G
            cc = Ec / (fabs(Ec) + ee);            // chirp cc
            printf("mchirp_2g : %d %g %.2e %.3f %.3f %.3f %.3f \n\n",
                   int(id), cc, pwc->cData[id - 1].mchirp,
                   pwc->cData[id - 1].mchirperr, pwc->cData[id - 1].tmrgr,
                   pwc->cData[id - 1].tmrgrerr, pwc->cData[id - 1].chi2chirp);
        } else {                // Enabled only for Search=CBC/BBH/IMBHB
            if (m_chirp && (TString(Search) == "CBC" || TString(Search) == "BBH" || TString(Search) == "IMBHB")) {
                ee = pwc->mchirp_upix(id, nRun);        // mchirp micropixel version
            }
        }
    }

    if (ID == id && !EFEC) {
        nSensitivity.gps = T;
        nAlignment.gps = T;
        nDisbalance.gps = T;
        nLikelihood.gps = T;
        nNullEnergy.gps = T;
        nCorrEnergy.gps = T;
        nCorrelation.gps = T;
        nSkyStat.gps = T;
        nEllipticity.gps = T;
        nPolarisation.gps = T;
        nNetIndex.gps = T;
    }

    pwc->sCuts[id - 1] = -1;
    pwc->clean(id);

    if (_vtd.size()) _avx_free_ps(_vtd);
    if (_vTD.size()) _avx_free_ps(_vTD);
    if (_eTD.size()) _avx_free_ps(_eTD);
    if (_AVX.size()) _avx_free_ps(_AVX);
    if (_APN.size()) _avx_free_ps(_APN);              // container for antenna patterns and noise RMS
    if (_DAT.size()) _avx_free_ps(_DAT);              // container for data packet amplitudes
    if (_SIG.size()) _avx_free_ps(_SIG);              // container for signal packet amplitudes
    if (_NUL.size()) _avx_free_ps(_NUL);              // container for null packet amplitudes

    return count;
}
