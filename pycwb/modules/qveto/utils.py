import numpy as np


def get_envelop(waveform):
    # CWB::mdc::GetEnvelope(wavearray<double>* x) {
    # //
    # // compute the envelope of wavearray x
    # //
    # // Input:
    # //         x      - input wavearray
    # //
    # // Output:
    # //         xq     - return the envelope of wavearray x

    # wavearray<double> xq;	// quadrature

    # if(x==NULL)      return xq;
    # if(x->size()==0) return xq;

    # // get quadrature
    # xq = *x;
    # CWB::mdc::PhaseShift(xq,90);

    # // get envelope
    # for(int i=0;i<x->size();i++) xq[i] = sqrt(pow(x->data[i],2)+pow(xq[i],2));

    # return xq;
    # }
    """
    Compute the envelope of a waveform array.

    Parameters
    ----------
    waveform : array_like
        Input waveform data as a 1D array.

    Returns
    -------
    envelope : ndarray
        The envelope of the waveform data.
    """
    waveform = np.array(waveform, dtype=float)

    if waveform.size == 0:
        return np.array([])
    
    # Compute quadrature (90-degree phase shift)
    # quadrature = phase_shift(waveform, 90)
    quadrature = apply_phase_shift(waveform, 90)

    # Compute envelope
    envelope = np.sqrt(np.square(waveform) + np.square(quadrature))

    return envelope


def phase_shift(x, phase_shift_degrees):
    """
    Apply a phase shift to a waveform array.

    Parameters
    ----------
    x : array_like
        Input waveform data as a 1D array.
    phase_shift_degrees : float
        Phase shift in degrees to be applied to the waveform.

    Returns
    -------
    x : ndarray
        The waveform data after applying the phase shift.
    """
    x = np.array(x, dtype=float)
    non_zero = np.nonzero(x)[0]
    if not non_zero.size:
        return x
    ibeg, iend = non_zero[0], non_zero[-1]
    ilen = iend - ibeg + 1

    # Prepare padded array
    isize = 2 * ilen
    isize += (-isize) % 4  # Ensure multiple of 4
    w = np.zeros(isize)
    start = isize // 4
    w[start:start+ilen] = x[ibeg:iend+1]
    x[ibeg:iend+1] = 0  # Clear original segment

    # FFT and phase shift
    fft_w = np.fft.fft(w)
    fft_w *= np.exp(-1j * np.deg2rad(phase_shift_degrees))
    shifted_w = np.fft.ifft(fft_w).real

    # Overlap-add back to x
    for i in range(len(shifted_w)):
        j = ibeg - start + i
        if 0 <= j < len(x):
            x[j] += shifted_w[i]
    return x


def apply_phase_shift(signal, phase_shift_deg, sample_rate=1024.0):
    """
    Applies a phase shift to a signal using FFT.
    
    Parameters:
        signal (numpy array): Input signal
        phase_shift_deg (float): Phase shift in degrees
        sample_rate (float): Sampling rate in Hz
        
    Returns:
        numpy array: Phase-shifted signal
    """
    n = len(signal)
    phase_shift_rad = np.deg2rad(phase_shift_deg)

    # FFT
    freq_domain = np.fft.fft(signal)
    
    # Frequency bins
    freqs = np.fft.fftfreq(n, d=1/sample_rate)

    # Apply phase shift: multiply by exp(j * phase)
    shift = np.exp(1j * phase_shift_rad * np.sign(freqs))
    shifted_freq_domain = freq_domain * shift

    # Inverse FFT
    shifted_signal = np.fft.ifft(shifted_freq_domain)

    return np.real(shifted_signal)

# void
# CWB::mdc::PhaseShift(wavearray<double>& x, double pShift) {
# //
# // apply phase shift
# //
# //
# // Input: x      - wavearray which contains the waveform data
# //        pShift - phase shift (degrees)
# //

#   if(pShift==0) return;

#   // search begin,end of non zero data
#   int ibeg=0; int iend=0;
#   for(int i=0;i<(int)x.size();i++) {
#     if(x[i]!=0 && ibeg==0) ibeg=i;
#     if(x[i]!=0) iend=i;
#   }
#   int ilen=iend-ibeg+1;
#   // create temporary array for FFTW & add scratch buffer + tShift
#   int isize = 2*ilen;
#   isize = isize + (isize%4 ? 4 - isize%4 : 0); // force to be multiple of 4
#   wavearray<double> w(isize);
#   w.rate(x.rate()); w=0;
#   // copy x data !=0 in the middle of w array & set x=0
#   for(int i=0;i<ilen;i++) {w[i+isize/4]=x[ibeg+i];x[ibeg+i]=0;}

#   // apply phase shift to waveform vector
#   w.FFTW(1);
#   TComplex C;
#   //cout << "pShift : " << pShift << endl;
#   pShift*=TMath::Pi()/180.;   
#   for (int ii=0;ii<(int)w.size()/2;ii++) {
#     TComplex X(w[2*ii],w[2*ii+1]);
#     X=X*C.Exp(TComplex(0.,-pShift));  // Phase Shift
#     w[2*ii]=X.Re();
#     w[2*ii+1]=X.Im();
#   }
#   w.FFTW(-1);

#   // copy shifted data to input x array
#   for(int i=0;i<(int)w.size();i++) {
#     int j=ibeg-isize/4+i;
#     if((j>=0)&&(j<(int)x.size())) x[j]=w[i];
#   }

#   return;
# }