//
// Created by Yumeng Xu on 10.03.23.
//

#ifndef CWB_CORE_COHERENCE_H
#define CWB_CORE_COHERENCE_H

#include "wseries.hh"
#include "netcluster.hh"

using namespace std;

inline std::vector<WSeries<double>> create_wseries_vector(WSeries<double> *tf_map)
{
    std::vector<WSeries<double>> v;
    v.push_back(*tf_map);
    return v;
}


//inline std::vector<wavearray<double>> create_wavearray_vector(wavearray<double> *tf_map)
//{
//    std::vector<wavearray<double>> v;
//    v.push_back(*tf_map);
//    return v;
//}

bool setrms(netcluster* wc, WSeries<double> * nRMS, size_t I);

netcluster *getNetworkPixels(int nIFO, std::vector <WSeries<double>> tf_maps, std::vector <WSeries<double>*> nRMS, wavearray<short> veto,
                             double Edge, int LAG, double Eo, double norm, std::vector<double> lagShift);


double threshold(std::vector<WSeries<double>> tf_maps, int nIFO, double Edge, double p, double shape);

#endif //CWB_CORE_COHERENCE_H
