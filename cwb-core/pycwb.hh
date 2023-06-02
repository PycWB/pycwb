#ifndef PYCWB_H
#define PYCWB_H

#include "wavearray.hh"
using namespace std;

void inline pycwb_copy_to_wavearray(double *value, wavearray<double> *wave, int size) {
    for (int i = 0; i < size; i++) {
        wave->data[i] = value[i];
    }
};

std::vector<double> inline pycwb_get_wavearray_data(wavearray<double> *wave) {
    std::vector<double> data;
    for (int i = 0; i < wave->size(); i++) {
        data.push_back(wave->data[i]);
    }

    return data;
};

std::vector<double> inline pycwb_get_wseries_data(WSeries<double> *wave) {
    std::vector<double> data;
    for (int i = 0; i < wave->size(); i++) {
        data.push_back(wave->data[i]);
    }

    return data;
};

#endif //PYCWB_H