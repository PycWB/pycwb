"""
Here is the unfinished code for the qveto module.
"""
from numpy import inf
from scipy.signal import hilbert
import numpy as np


def q_veto(detectors, reconstructed_whiten_waveforms, whiten_waveforms, sSNR, likelihood, enable_qveto3=True):
    qveto = inf
    qfactor = inf

    qfactor1 = [0] * len(detectors)
    qfactor2 = [0] * len(detectors)

    for i in range(detectors):
        qveto_rec, qfactor_rec = get_q_veto(reconstructed_whiten_waveforms[i])
        if qveto_rec < qveto:
            qveto = qveto_rec
        if qfactor_rec < qfactor:
            qfactor = qfactor_rec

        qfactor1[i] = qfactor

        fpeak = get_peak_frequency(reconstructed_whiten_waveforms[i])
        qfactor2[i] = get_q_factor(reconstructed_whiten_waveforms[i], fpeak, True)

        qveto_wh, qfactor_wh = get_q_veto(whiten_waveforms[i])
        if qveto_wh < qveto:
            qveto = qveto_wh
        if qfactor_wh < qfactor:
            qfactor = qfactor_wh

    qveto2 = 0
    qveto3 = 0
    for i in range(detectors):
        qveto2 += sSNR[i] * qfactor1[i]
    qveto2 /= likelihood

    if enable_qveto3:
        for i in range(detectors):
            qveto3 += sSNR[i] * qfactor2[i]
        qveto3 /= likelihood

    return {"qveto": qveto, "qveto2": qveto2, "qveto3": qveto3, "qfactor": qfactor}


def get_q_veto(waveform):
    # wavearray<double> x = *wf;;

    # // resample data by a factor 4
    # int xsize=x.size();
    # x.FFTW(1);
    # x.resize(4*x.size());
    # x.rate(4*x.rate());
    # for(int j=xsize;j<x.size();j++) x[j]=0;
    # x.FFTW(-1);
    #
    # // extract max/min values and save the absolute values in the array 'a'
    # wavearray<double> a(x.size());
    # int size=0;
    # double dt = 1./x.rate();
    # double prev=x[0];
    # double xmax=0;
    # for (int i=1;i<x.size();i++) {
    # if(fabs(x[i])>xmax) xmax=fabs(x[i]);
    # if(prev*x[i]<0) {
    # a[size]=xmax;
    # size++;
    # xmax=0;
    # }
    # prev=x[i];
    # }
    #
    # // find max value/index ans save on  amax/imax
    # int imax=-1;
    # double amax=0;
    # for (int i=1;i<size;i++) {
    # if(a[i]>amax) {amax=a[i];imax=i;}
    # }
    #
    # /*
    # cout << endl;
    # cout << "a[imax-2] " << a[imax-2] << endl;
    # cout << "a[imax-1] " << a[imax-1] << endl;
    # cout << "a[imax]   " << a[imax] << endl;
    # cout << "a[imax+1] " << a[imax+1] << endl;
    # cout << "a[imax+2] " << a[imax+2] << endl;
    # cout << endl;
    # */
    #
    # // compute Qveto
    # double ein=0;	// energy of max values inside NTHR
    # double eout=0;	// energy of max values outside NTHR
    # for (int i=0;i<size;i++) {
    # if(abs(imax-i)<=NTHR) {
    # ein+=a[i]*a[i];
    # //cout << i << " ein " << a[i] << " " << amax << endl;
    # } else {
    # if(a[i]>amax/ATHR) eout+=a[i]*a[i];
    # //if(a[i]>amax/ATHR) cout << i << " eout " << a[i] << " " << amax << endl;
    # }
    # }
    # Qveto = ein>0 ? eout/ein : 0.;
    # //cout << "Qveto : " << Qveto << " ein : " << ein << " eout : " << eout << endl;
    #
    # // compute Qfactor
    # float R = (a[imax-1]+a[imax+1])/a[imax]/2.;
    # Qfactor = sqrt(-pow(TMath::Pi(),2)/log(R)/2.);
    # //cout << "Qfactor : " << Qfactor << endl;
    #
    # return;
    qveto = 0
    qfactor = 0

    return qveto, qfactor


def get_peak_frequency(waveform):
    """
    multiply the input waveform wf by its envelope, perform FFT transform, compute max frequency -> peak, return frequency peak
    :param waveform: input waveform
    :type waveform: TimeSeries
    :return:
    """

    # wavearray<double> x  = *wf;
    # wavearray<double> ex = CWB::mdc::GetEnvelope(&x);      // get envelope
    # int N = x.size();
    # for(int i=0;i<N;i++) x[i]*=ex[i];
    #
    # x.FFTW(1);
    #
    # // fill time/amp graph
    # double df=(x.rate()/(double)N)/2.;
    # double amax=0.;
    # double fmax=0.;
    # for(int i=0;i<N;i+=2) {
    #     double freq=i*df;
    # double amp=x[i]*x[i]+x[i+1]*x[i+1];
    # if(amp>amax) {fmax=freq;amax=amp;}
    # }
    #
    # return fmax;

    # ex = get_envelop(waveform)

    # x = waveform * ex
    sample_rate = waveform.sample_rate.value

    analytic_signal = hilbert(waveform)
    envelope = np.abs(analytic_signal)
    # envelope = get_envelop(waveform)
    x = np.array(waveform * envelope)

    # Step 3: Compute FFT
    fft_result = np.fft.fft(x)
    N = len(x)

    # Step 4: Find dominant frequency
    df = (sample_rate / N) / 2

    # find the index of the maximum value in the FFT result
    max_index = np.argmax(np.abs(fft_result[:N // 2]))
    dominant_freq = max_index * df

    return dominant_freq


def get_q_factor(waveform, peak_frequency, fix_amplitude):
    """
    Get Q-factor of a waveform.
    The input waveform is resampled 4x, get mean and sigma from a gaussian fit of the waveform envelope,
    compute the range [xmin,xmax] = [mean-3*sigma,mean-3*sigma]], compute area of the envelope in the rage [xmin,xmax],
    compute qfactor using area, max_amplitude and frequency, return qfactor

    :param waveform: input waveform
    :type waveform: TimeSeries
    :param peak_frequency: input peak frequency
    :type peak_frequency: float
    :param fix_amplitude: true/false : fix/not-fix amplitude of the gaussian fit
    :type fix_amplitude: bool
    """

    # wavearray<double> x = *wf;;
    #
    # // resample data by a factor 4
    # int xsize=x.size();
    #
    # x.FFTW(1);
    # x.resize(4*x.size());
    # x.rate(4*x.rate());
    # for(int j=xsize;j<x.size();j++) x[j]=0;
    # x.FFTW(-1);
    #
    # // compute range [xmin,xmax] used for final fit using the square of x
    # double gmean, gsigma;
    # GetGaussFitPars2(&x, gmean, gsigma, fixAmax);
    # //  cout << "GetQfactor -> gmean " << gmean << " gsigma " << gsigma << endl;
    #
    # double xmin = gmean-3.0*gsigma;
    # double xmax = gmean+3.0*gsigma;
    # //  cout << "GetQfactor -> xmin " << xmin << " xmax " << xmax << endl;
    #
    # x = CWB::mdc::GetEnvelope(&x);      // get envelope
    #
    # // extract max/min values and save the absolute values in the array 'a'
    # double dt = 1./x.rate();
    # // find max value/index and save on  amax/imax
    # int imax=-1;
    # double amax=0;
    # for (int i=1;i<x.size();i++) {
    # if(x[i]>amax) {amax=x[i];imax=i;}
    # }
    #
    # double area=0.;
    # for (int i=1;i<x.size();i++) {
    # double t=i*dt;
    # if(t>xmin && t<xmax) area+=x[i]*dt;
    # }
    #
    # double sigma = area/amax/sqrt(2*TMath::Pi());
    # //  cout << "sigma integral is " << sigma << endl;
    # double qfactor = sigma*(TMath::TwoPi()*frequency);    // compute qfactor
    #
    # return qfactor;
    pass
