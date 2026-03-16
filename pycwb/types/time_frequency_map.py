import numpy as np
from dataclasses import dataclass

from .wavelet import Wavelet
from .time_series import TimeSeries


@dataclass
class TDBatchInputs:
	"""
	Pre-computed padded quadrature planes and filter tables for batch
	time-delay vector extraction.

	Created by :meth:`TimeFrequencyMap.prepare_td_inputs`.

	Attributes
	----------
	padded00 : np.ndarray
		Zero-padded 00-phase plane, float32, shape ``(n_time + 2*n_coeffs, M+1)``.
	padded90 : np.ndarray
		Zero-padded 90-phase plane, float32, shape ``(n_time + 2*n_coeffs, M+1)``.
	T0 : np.ndarray
		Same-band TD filter table, float64, shape ``(2*J+1, 2*n_coeffs+1)``.
	Tx : np.ndarray
		Cross-band TD filter table, float64, shape ``(2*J+1, 2*n_coeffs+1)``.
	M : int
		Number of frequency layers (Nyquist index).
	n_coeffs : int
		Half-width of symmetric TD filters.
	J : int
		Maximum delay index (``M * L``).
	"""
	padded00: np.ndarray
	padded90: np.ndarray
	T0: np.ndarray
	Tx: np.ndarray
	M: int
	n_coeffs: int
	J: int

	def extract_td_vecs(self, pixel_indices, K):
		"""
		Batch TD vector extraction for the given pixel indices.

		Parameters
		----------
		pixel_indices : array-like, shape (n_pixels,)
			Flat pixel indices into the TF plane (int32).
		K : int
			TD filter half-range (corresponds to config.TDSize).

		Returns
		-------
		np.ndarray, shape (n_pixels, 4*K+2), dtype float32
		"""
		from pycwb.utils.td_vector_batch import batch_get_td_vecs
		return batch_get_td_vecs(
			np.asarray(pixel_indices, dtype=np.int32),
			self.padded00,
			self.padded90,
			self.T0,
			self.Tx,
			self.M,
			self.n_coeffs,
			K,
			self.J,
		)


@dataclass
class TimeFrequencyMap:
	"""
	Data class for storing time-frequency map information.
	"""
	data: np.ndarray
	is_whitened: bool
	dt: float
	df: float
	start: float
	stop: float
	# allow none
	f_low: float | None
	f_high: float | None
	edge: float | None
	wavelet: Wavelet  # Replace 'object' with the actual type of wavelet if available
	len_timeseries: int = None  # Original time series length before transform
	ts_data: np.ndarray = None  # Original time series data (avoids roundtrip w2t→t2w in max_energy)

	@classmethod
	def from_timeseries(cls, ts: TimeSeries, wavelet: Wavelet,
						is_whitened: bool = False,
						f_low: float = None, f_high: float = None, edge: float = None):
		"""
		Create a TimeFrequencyMap from a TimeSeries and wavelet.

		:param ts: Time series
		:type ts: TimeSeries
		:param wavelet: Wavelet object
		:type wavelet: Wavelet
		:param f_low: Low frequency cutoff
		:type f_low: float | None
		:param f_high: High frequency cutoff
		:type f_high: float | None
		:param edge: Edge parameter
		:type edge: float | None
		:return: TimeFrequencyMap object
		:rtype: TimeFrequencyMap
		"""
		# Perform wavelet transform (placeholder, replace with actual implementation)
		# Here we assume a function `wavelet_transform` exists that performs the transform.
		# data = wavelet_transform(ts.data, wavelet)

		# Placeholder for transformed data, replace with actual transformed data
		data = wavelet.t2w(ts.data, ts.sample_rate, ts.t0)

		return cls(
			data=data.data,
			is_whitened=is_whitened,
			dt=data.dt,
			df=data.df,
			start=ts.t0,
			stop=ts.end_time,
			f_low=f_low,
			f_high=f_high,
			edge=edge,
			wavelet=wavelet,
			len_timeseries=len(ts.data),
			ts_data=np.asarray(ts.data, dtype=np.float64),
		)

	@property
	def timeseries(self):
		return self.wavelet.inverse(self.data)

	@property
	def wavelet_rate(self):
		return int(1.0 / self.dt)

	@property
	def size(self):
		return int(np.asarray(self.data).size)

	@property
	def maxLayer(self):
		return self.wavelet.M

	def wavecount(self, threshold, edge_length=None):
		"""
		Count coefficients above a threshold.

		This is a Python-native compatibility helper for
		`ROOT.WaveArray::wavecount`.

		:param threshold: threshold value
		:type threshold: float
		:param edge_length: optional flattened edge width excluded on both sides
		:type edge_length: int | None
		:return: number of coefficients above the threshold
		:rtype: int
		"""
		flat = np.asarray(self.data)
		if np.iscomplexobj(flat):
			flat = flat.real
		flat = flat.ravel()

		if edge_length is not None:
			n = int(edge_length)
			if n <= 0:
				return int(np.sum(flat > threshold))
			if 2 * n >= flat.size:
				return 0
			return int(np.sum(flat[n:-n] > threshold))
		return int(np.sum(flat > threshold))

	def wavesplit(self, start_index, end_index, split_index):
		"""
		Return the order-statistics value in a flattened slice.

		Uses `np.partition` for efficiency and keeps Python compatibility with
		`ROOT.WaveArray::wavesplit` semantics.

		:param start_index: start index of the segment
		:type start_index: int
		:param end_index: end index of the segment
		:type end_index: int
		:param split_index: index to split the sorted segment
		:type split_index: int
		:return: value at the split index
		:rtype: float
		"""
		flat = np.asarray(self.data)
		if np.iscomplexobj(flat):
			flat = flat.real
		flat = flat.ravel()

		section = flat[start_index:end_index]
		if section.size == 0:
			raise ValueError("wavesplit() empty input segment")
		split_index = int(max(0, min(split_index, section.size - 1)))
		parted = np.partition(section, split_index)
		value = parted[split_index]
		return value

	def Gamma2Gauss(self, hist=None):
		"""
		Apply gamma-to-Gaussian style normalization on TF energy values.

		The transform updates `self.data` in place and optionally appends
		intermediate/final values into `hist`.

		:param hist: optional list-like accumulator for diagnostics
		:type hist: list | None
		:return: scaling pivot (`ALP`) used by the transform, or 0.0 on failure
		:rtype: float
		"""
		original = np.asarray(self.data)
		shape = original.shape
		flat_2d = original.real if np.iscomplexobj(original) else original
		flat_2d = np.asarray(flat_2d, dtype=np.float64)
		# C++ stores data in time-major order (j = t*M + m); ravel accordingly.
		if flat_2d.ndim == 2:
			flat = flat_2d.T.ravel()  # (M,T) → (T,M) → 1D time-major: flat[t*M + m]
		else:
			flat = flat_2d.ravel()

		if flat.size < 4:
			return 0.0

		nn_all = int(flat.size)
		M = int(shape[0]) if len(shape) == 2 else 1
		nL = int(float(self.edge or 0.0) * self.wavelet_rate * M)
		nL = max(0, min(nL, nn_all - 2))
		nR = nn_all - nL - 1
		if nR <= nL:
			return 0.0

		region = flat[nL:nR]
		if region.size == 0:
			return 0.0

		# C++ waveSplit(nL, nR, m) operates on [nL, nR] inclusive.
		# Use nR+1 as the exclusive end for partition to match C++ range.
		ws_region = flat[nL:nR + 1]

		wavecount_1 = int(np.sum(flat > 0.001))  # count over full array, matches C++ wavecount(0.001)
		fff = (nR - nL) * wavecount_1 / float(nn_all)

		split_idx_med = nR - int(0.5 * fff)
		split_idx_med = max(nL, min(split_idx_med, nR))
		rel_med = split_idx_med - nL
		med = float(np.partition(ws_region, rel_med)[rel_med])
		if med <= 0.0:
			return 0.0

		mask = (region > 0.01) & (region < 20.0 * med)
		valid_data = region[mask]
		if valid_data.size == 0:
			return 0.0

		aaa = float(np.sum(valid_data))
		bbb = float(np.sum(np.log(valid_data)))
		count = float(valid_data.size)
		alp = np.log(aaa / count) - bbb / count
		if alp <= 0.0:
			return 0.0
		alp = (3.0 - alp + np.sqrt((alp - 3.0) * (alp - 3.0) + 24.0 * alp)) / (12.0 * alp)

		avr = med * (3.0 * alp + 0.2) / (3.0 * alp - 0.8)

		ALP = med * alp / avr

		amp = flat * alp / avr
		transformed = np.zeros_like(flat)
		mask_pos = amp >= ALP
		if np.any(mask_pos):
			transformed[mask_pos] = amp[mask_pos] - ALP * (1.0 + np.log(amp[mask_pos] / ALP))

		if hist is not None:
			hist.extend(transformed[nL:nR].tolist())

		region2 = transformed[nL:nR]
		if region2.size == 0:
			return 0.0
		# C++ waveSplit(nL, nR, m) operates on [nL, nR] inclusive for partition.
		ws_region2 = transformed[nL:nR + 1]
		# C++ wavecount(1e-5, nL) counts [nL, size-nL) symmetrically
		wavecount_2 = int(np.sum(transformed[nL:nn_all - nL] > 1.0e-5))
		split_idx_rms = nR - int(0.3173 * wavecount_2)
		split_idx_rms = max(nL, min(split_idx_rms, nR))
		rel_rms = split_idx_rms - nL
		qv = float(np.partition(ws_region2, rel_rms)[rel_rms])
		if qv <= 0.0:
			return 0.0

		rms = 1.0 / qv
		transformed *= rms

		if hist is not None:
			hist.extend(np.sqrt(np.clip(transformed[nL:nR], 0.0, None)).tolist())

		if len(shape) == 2:
			M2, T2 = shape  # shape = (M, T) → time-major reshape (T, M) then transpose
			self.data = transformed.reshape(T2, M2).T
		else:
			self.data = transformed

		return float(ALP)

	def bandpass(self, f_low=None, f_high=None):
		"""
		Set the frequency band for the time-frequency series.

		:param f_low: low frequency cutoff
		:type f_low: float
		:param f_high: high frequency cutoff
		:type f_high: float
		"""
		if f_low is not None:
			self._f_low = f_low
		if f_high is not None:
			self._f_high = f_high

	def _compute_bounds(self, n_freq=None, n_time=None):
		"""
		Compute flattened/time-frequency bounds used by packet operations.

		:param n_freq: number of frequency bins
		:type n_freq: int | None
		:param n_time: number of time bins
		:type n_time: int | None
		:return: `(jb, je, mL, mH)` flattened and frequency limits
		:rtype: tuple[int, int, int, int]
		"""
		if n_freq is None or n_time is None:
			coeffs = np.asarray(self.data)
			if coeffs.ndim != 2:
				raise ValueError("_compute_bounds expects a 2D time-frequency map")
			n_freq, n_time = coeffs.shape

		M = int(n_freq)
		J = int(n_freq * n_time)
		edge = float(self.edge or 0.0)
		jb = int(edge * self.wavelet_rate / 4.0) * M
		if jb < 4 * M:
			jb = 4 * M
		je = J - jb
		df = self.df
		f_low = 0.0 if self.f_low is None else float(self.f_low)
		f_high = (df * (M - 1)) if self.f_high is None else float(self.f_high)
		mL = int(f_low / df + 0.1)
		mH = int(f_high / df + 0.1)
		mL = max(0, mL)
		mH = min(M - 1, mH)
		return jb, je, mL, mH

	def wdm_packet(self, pattern: int, mode: str = 'e', coeffs=None, return_map: bool = False):
		"""
		Compute WDM packet energy/amplitude map for a given pattern.

		Vectorized NumPy implementation with non-wrapping edge handling.
		Uses `self.data` by default, or external `coeffs` if provided.

		:param pattern: packet pattern ID (compatible with cWB-style presets)
		:type pattern: int
		:param mode: output mode: `'e'` energy, `'l'` likelihood-like, `'a'` amplitude
		:type mode: str
		:param coeffs: optional complex TF coefficient map
		:type coeffs: np.ndarray | None
		:param return_map: if True, return computed map instead of shape scalar
		:type return_map: bool
		:return: packet shape (float) or computed map when `return_map=True`
		:rtype: float | np.ndarray
		"""
		pattern = abs(int(pattern))
		mode = mode.lower()
		if mode not in {'e', 'l', 'a'}:
			raise ValueError("mode must be one of {'e', 'l', 'a'}")

		complex_map = np.asarray(self.data if coeffs is None else coeffs)
		if complex_map.ndim != 2:
			raise ValueError("wdm_packet expects a 2D time-frequency map")

		M, T = complex_map.shape
		jb, je, mL, mH = self._compute_bounds(M, T)

		# Determine shape/mean and offsets exactly as in cWB WSeries::wdmPacket
		if pattern in (1, 3, 4):
			shape = mean = 3.0
			mL += 1
			mH -= 1
		elif pattern == 2:
			shape = mean = 3.0
		elif pattern in (5, 6):
			shape = mean = 5.0
			mL += 2
			mH -= 2
		elif pattern in (7, 8):
			shape = mean = 5.0
			mL += 1
			mH -= 1
		elif pattern == 9:
			shape = mean = 9.0
			mL += 1
			mH -= 1
		else:
			shape = mean = 1.0

		# p offsets in flattened indexing j = m + t*M (Fortran order)
		p = [0] * 9
		if pattern == 1:
			p[1], p[2] = 1, -1
		elif pattern == 2:
			p[1], p[2] = M, -M
		elif pattern == 3:
			p[1], p[2] = M + 1, -M - 1
		elif pattern == 4:
			p[1], p[2] = -M + 1, M - 1
		elif pattern == 5:
			p[1], p[2], p[3], p[4] = M + 1, -M - 1, 2 * M + 2, -2 * M - 2
		elif pattern == 6:
			p[1], p[2], p[3], p[4] = -M + 1, M - 1, -2 * M + 2, 2 * M - 2
		elif pattern == 7:
			p[1], p[2], p[3], p[4] = 1, -1, M, -M
		elif pattern == 8:
			p[1], p[2], p[3], p[4] = M + 1, -M + 1, M - 1, -M - 1
		elif pattern == 9:
			p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8] = 1, -1, M, -M, M + 1, M - 1, -M + 1, -M - 1

		real_f = np.asarray(np.real(complex_map), dtype=np.float64).reshape(-1, order='F')
		imag_f = np.asarray(np.imag(complex_map), dtype=np.float64).reshape(-1, order='F')
		J = real_f.size

		energy_f = np.zeros(J, dtype=np.float64)
		amp_real_f = np.zeros(J, dtype=np.float64)
		amp_imag_f = np.zeros(J, dtype=np.float64)

		if jb < 0:
			jb = 0
		if je > J:
			je = J
		if je <= jb:
			self.last_energy = energy_f.reshape((M, T), order='F')
			self.last_amplitude = (amp_real_f + 1j * amp_imag_f).reshape((M, T), order='F')
			if coeffs is None:
				self.data = self.last_amplitude if mode == 'a' else self.last_energy
			if return_map:
				return self.last_amplitude if mode == 'a' else self.last_energy
			return shape

		j = np.arange(jb, je, dtype=np.int64)
		m = j % M
		band_mask = (m >= mL) & (m <= mH)
		if not np.any(band_mask):
			self.last_energy = energy_f.reshape((M, T), order='F')
			self.last_amplitude = (amp_real_f + 1j * amp_imag_f).reshape((M, T), order='F')
			if coeffs is None:
				self.data = self.last_amplitude if mode == 'a' else self.last_energy
			if return_map:
				return self.last_amplitude if mode == 'a' else self.last_energy
			return shape

		jv = j[band_mask]

		# exact cWB accumulation: p[1..8] always contribute; unspecified p[n] are 0
		ss = np.zeros_like(jv, dtype=np.float64)
		ee = np.zeros_like(jv, dtype=np.float64)
		EE = np.zeros_like(jv, dtype=np.float64)
		for n in range(1, 9):
			idx = jv + p[n]
			qr = real_f[idx]
			qi = imag_f[idx]
			ss += qr * qi
			ee += qr * qr
			EE += qi * qi

		q0r = real_f[jv]
		q0i = imag_f[jv]
		ss += q0r * q0i * (mean - 8.0)
		ee += q0r * q0r * (mean - 8.0)
		EE += q0i * q0i * (mean - 8.0)

		cc = ee - EE
		ss2 = ss * 2.0
		nn = np.sqrt(cc * cc + ss2 * ss2)
		sum_eeEE = ee + EE
		small_mask = sum_eeEE < nn
		if np.any(small_mask):
			nn[small_mask] = sum_eeEE[small_mask]

		a1 = np.sqrt(np.clip((sum_eeEE + nn) / 2.0, 0.0, None))
		a2 = np.sqrt(np.clip((sum_eeEE - nn) / 2.0, 0.0, None))
		aa = a1 + a2

		if (mode == 'e') or (mode == 'l') or (mean == 1.0):
			em = sum_eeEE / 2.0
		else:
			em = (aa * aa) / 4.0

		alp = shape - np.log(shape) / 3.0 if shape > 0 else shape

		if mode == 'l':
			em = em * (shape / mean)
			# where em < alp, set to 0; otherwise apply the correction
			mask2 = em < alp
			em2 = em.copy()
			# avoid log of zero or negative
			pos_mask = ~mask2
			if pos_mask.any():
				em2[pos_mask] = em[pos_mask] - alp * (1.0 + np.log(em[pos_mask] / alp))
			em2[mask2] = 0.0
			em = em2

		# amplitude branch: compute amplitude complex array
		if mode == 'a':
			safe_nn = nn.copy()
			safe_nn[safe_nn == 0.0] = 1.0
			cc_norm = cc / safe_nn
			ss_norm = ss2 / safe_nn
			amp_real = aa * cc_norm
			amp_imag = aa * ss_norm / 2.0
			zero_mask = nn == 0.0
			amp_real[zero_mask] = 0.0
			amp_imag[zero_mask] = 0.0
			amp_real_f[jv] = amp_real
			amp_imag_f[jv] = amp_imag
		else:
			energy_f[jv] = em

		energy_out = energy_f.reshape((M, T), order='F')
		amp_out = (amp_real_f + 1j * amp_imag_f).reshape((M, T), order='F')

		self.last_energy = energy_out
		self.last_amplitude = amp_out

		if coeffs is None:
			self.data = amp_out if mode == 'a' else energy_out

		if return_map:
			return amp_out if mode == 'a' else energy_out

		return shape

	def prepare_td_inputs(self, td_filters):
		"""
		Build padded quadrature planes and filter tables for batch TD extraction.

		This is a pure computation — no caching.  Callers are responsible for
		storing the result when it needs to be reused across lags.

		Parameters
		----------
		td_filters : wdm_wavelet.core.time_delay.TDFilterBank
			Pre-built filter bank (from ``WDM.set_td_filter``).

		Returns
		-------
		TDBatchInputs
		"""
		data = np.asarray(self.data, dtype=np.complex128)  # (M+1, n_time)
		tf00 = np.ascontiguousarray(data.real.T, dtype=np.float64)  # (n_time, M+1)
		tf90 = np.ascontiguousarray(data.imag.T, dtype=np.float64)

		n_coeffs = int(td_filters.n_coeffs)
		M = int(td_filters.M)
		J = int(td_filters.max_delay)

		pad = [(n_coeffs, n_coeffs), (0, 0)]
		padded00 = np.ascontiguousarray(np.pad(tf00, pad), dtype=np.float32)
		padded90 = np.ascontiguousarray(np.pad(tf90, pad), dtype=np.float32)

		T0 = np.ascontiguousarray(td_filters.T0, dtype=np.float64)
		Tx = np.ascontiguousarray(td_filters.Tx, dtype=np.float64)

		return TDBatchInputs(
			padded00=padded00,
			padded90=padded90,
			T0=T0,
			Tx=Tx,
			M=M,
			n_coeffs=n_coeffs,
			J=J,
		)


def whiten_slice(data, rate, t, mode=1, offset=0.0, stride=0.0):
	"""
	Robust time-domain whitening by local variance normalization (NumPy optimized).
	"""
	data = np.ascontiguousarray(data, dtype=np.float64)
	N = len(data)
	segT = N / rate

	if t <= 0:
		t = segT - 2 * offset

	offset_samples = int(offset * rate + 0.5)
	if offset_samples % 2:
		offset_samples -= 1

	if stride > t or stride <= 0:
		stride = t

	K = int((segT - 2 * offset) / stride)
	if K == 0:
		K = 1

	n = N - 2 * offset_samples
	k = n // K
	if k % 2:
		k -= 1

	m = int(t * rate + 0.5)
	mL = int(0.15865 * m + 0.5)
	mR = m - mL - 1

	if m < 3 or mL < 2 or mR > m - 2:
		raise ValueError("whiten_timeseries: input array too short")

	# ---- build starting indices of blocks ----
	jL = (N - k * K) // 2
	jR = N - offset_samples - m
	jj = jL - m // 2
	starts = []
	for _ in range(K + 1):
		if jj < offset_samples:
			starts.append(offset_samples)
		elif jj >= jR:
			starts.append(jR)
		else:
			starts.append(jj)
		jj += k
	starts = np.array(starts)

	# ---- extract all windows at once ----
	# shape: (K+1, m)
	windows = np.stack([data[s:s + m] for s in starts])

	# ---- compute robust stats per window ----
	q16, med, q84 = np.quantile(windows, [0.15865, 0.5, 0.84135], axis=1)
	medians = med if mode else np.sqrt(med * 0.7191)
	norms = (q84 - q16) / 2.0

	if mode == 0:
		return medians  # only noise estimates

	# ---- interpolation of median and norm across samples ----
	out = np.empty_like(data)

	# left boundary
	left_len = jL
	if left_len > 0:
		x = data[:left_len] - medians[0]
		r = norms[0]
		out[:left_len] = x / r if mode == 1 else x / (r * r)

	# main blocks (vectorized over each block of length k)
	p = left_len
	for j in range(K):
		idx = np.arange(k)  # [0, 1, ..., k-1]
		w = idx / k
		med_interp = medians[j] * (1 - w) + medians[j + 1] * w
		norm_interp = norms[j] * (1 - w) + norms[j + 1] * w
		x = data[p:p + k] - med_interp
		out[p:p + k] = x / norm_interp if mode == 1 else x / (norm_interp * norm_interp)
		p += k

	# right boundary
	if p < N:
		x = data[p:] - medians[-1]
		r = norms[-1]
		out[p:] = x / r if mode == 1 else x / (r * r)

	return out


def compute_rms(tf_map, t, mode, offset, stride):
	"""
	Compute noise RMS for each layer in the time-frequency map.

	:param tf_map: Time-frequency map
	:type tf_map: TimeFrequencyMap
	:param t: whitening interval length in seconds (if <=0 use full duration minus offset*2)
	:param mode: whitening mode (0 = return medians, 1 = whitened, 2 = power-normalized)
	:param offset: boundary offset in seconds
	:param stride: step length in seconds
	:return: noise RMS for each layer
	:rtype: np.ndarray
	"""
	duration = tf_map.stop - tf_map.start
	if t <= 0:
		t = duration - 2. * offset
	w_mode = abs(mode)

	K = int((duration - 2 * offset) / stride) + 1  # number of noise measurements

	n_layers = len(tf_map.data)
	nRMS = np.zeros((n_layers, K))

	for i, layer in enumerate(tf_map.data):
		nRMS[i] = whiten_slice(np.abs(layer) ** 2, 1.0 / tf_map.dt, t, w_mode, offset, stride)

	return nRMS
