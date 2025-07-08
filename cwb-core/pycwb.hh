#ifndef PYCWB_H
#define PYCWB_H

#include "wavearray.hh"
#include "wseries.hh"
#include "WDM.hh"
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

    // wave->resize(0); // Clear the wavearray to free memory

    return data;
};

std::vector<short> inline pycwb_get_short_wavearray_data(wavearray<short> *wave) {
    std::vector<short> data;
    for (int i = 0; i < wave->size(); i++) {
        data.push_back(wave->data[i]);
    }

    // wave->resize(0); // Clear the wavearray to free memory

    return data;
};

std::vector<double> inline pycwb_get_wseries_data(WSeries<double> *wave) {
    std::vector<double> data;
    for (int i = 0; i < wave->size(); i++) {
        data.push_back(wave->data[i]);
    }

    return data;
};

std::pair<int, std::vector<double>> inline pycwb_get_base_wave(WDM<double> *pwdm, int tf_index, bool Quad) {
    wavearray<double> wave;
    int j = pwdm->getBaseWave(tf_index, wave, Quad);

    return {j ,pycwb_get_wavearray_data(&wave)};
};

#endif //PYCWB_H