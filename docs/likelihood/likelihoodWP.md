# likelihoodWP: Mathematical and Technical Notes

## Purpose

This document describes the pure-Python `likelihoodWP` implementation in `pycwb/modules/likelihoodWP` from both a mathematical and implementation perspective. The goal is to explain what the code computes, how the main quantities relate to coherent burst reconstruction, and how the Python port maps those operations onto Numba and NumPy kernels.

The main entry points are:

- `setup_likelihood(...)`
- `likelihood(...)`
- `likelihood_wrapper(...)`

The core supporting kernels are implemented in:

- `dpf.py`: dominant-polarization-frame construction
- `sky_stat.py`: sky-loop statistics and packet orthogonalization
- `utils.py`: packet norms, null energy, waveform normalization, and auxiliary statistics
- `pixel_batch_ops.py`: vectorized extraction of per-pixel TD amplitudes and RMS values

## High-Level Pipeline

For a single cluster, the Python likelihood pipeline is:

1. Build or reuse the sky-delay and antenna-pattern tables.
2. Load pixel amplitudes and per-pixel normalized RMS weights.
3. Scan the sky grid.
4. For each sky direction, apply detector-dependent time delays.
5. Build the dominant polarization frame (DPF).
6. Reconstruct the coherent signal packet and compute coherent statistics.
7. Select the best sky location.
8. Recompute detailed statistics at the maximizing sky point.
9. Apply threshold cuts.
10. Fill cluster-level detection statistics, waveform-derived outputs, and chirp mass.

At the implementation level this is centered in `likelihood(...)`, with the sky scan performed by `find_optimal_sky_localization(...)` and the best-direction detailed evaluation performed by `calculate_sky_statistics(...)`.

## Inputs and Data Layout

### Cluster and Pixel Data

Each cluster is a list of time-frequency pixels. Each pixel carries, per interferometer:

- a noise RMS estimate,
- whitened time-delay amplitudes `td_amp`,
- reconstructed packet amplitudes filled later by the likelihood stage.

The vectorized loader `load_data_from_pixels_vectorized(...)` converts the Python object graph into dense arrays:

- `rms` with shape `(n_ifo, n_pix)`
- `td00` with shape `(n_ifo, n_pix, n_delay)` before transpose
- `td90` with shape `(n_ifo, n_pix, n_delay)` before transpose

Inside `likelihood(...)`, these are rearranged into forms better suited to the Numba kernels:

- `rms.T -> (n_pix, n_ifo)` for DPF construction
- `td00 -> (n_delay, n_ifo, n_pix)`
- `td90 -> (n_delay, n_ifo, n_pix)`

### Sky Grid

`setup_likelihood(...)` produces or reuses:

- `ml[i, l]`: integer time-delay sample index for interferometer `i` and sky pixel `l`
- `FP[i, l]`: plus antenna pattern
- `FX[i, l]`: cross antenna pattern

These are stored in transposed float32 form for the hot path:

- `FP_t[l, i]`
- `FX_t[l, i]`

The function also builds `(ra, dec)` arrays for the same sky grid using `_build_sky_directions(...)`, allowing the maximizing sky index to be converted back to physical coordinates.

## Precomputed Regulators and Thresholds

The setup stage translates configuration parameters into quantities used throughout the likelihood code.

Let:

- `Acore = config.Acore`
- `gamma = config.gamma`
- `delta = config.delta`
- `nIFO = number of detectors`

Then the code defines:

$$
E_{\mathrm{thr}} = 2 A_{\mathrm{core}}^2 n_{\mathrm{IFO}}
$$

This is the per-pixel network-energy threshold used to decide whether a time-frequency pixel is active at a given sky location.

The two regularization terms are:

$$
\gamma_{\mathrm{reg}} = \frac{2}{3} \gamma^2
$$

and

$$
\delta_{\mathrm{reg}} = \min(|\delta|, 1)
$$

with the special handling that `delta = 0` is replaced by `10^{-5}` to avoid degeneracy.

The runtime regulator vector used in the sky kernels is

$$
\mathrm{REG} = \left[\sqrt{2}\,\delta_{\mathrm{reg}},\; \mathrm{DPF\ energy\ regulator},\; 0\right].
$$

The second component is filled later by a sky-averaged DPF scan.

## Network-Weighted RMS Normalization

For a given pixel and detector `i`, let the detector noise RMS be $\sigma_i$. The code first forms inverse RMS weights $1 / \sigma_i$, then defines a per-pixel network RMS

$$
\sigma_{\mathrm{net}} = \left(\sum_i \sigma_i^{-2}\right)^{-1/2}.
$$

The detector weights used in the DPF step are then

$$
r_i = \sigma_{\mathrm{net}} \sigma_i^{-1}.
$$

In the code, `rms[pixel, ifo]` stores these normalized weights, not the raw noise values. This is important because the DPF and subsequent coherent projections work in whitened, noise-weighted detector coordinates.

## Time-Delay Slicing for a Sky Position

For a fixed sky index `l`, the code pulls one delayed TD sample for each interferometer using `ml[i, l]`:

$$
v^{00}_{i,j}(l) = \mathrm{td00}[\,ml(i,l)+o,\; i,\; j\,]
$$

$$
v^{90}_{i,j}(l) = \mathrm{td90}[\,ml(i,l)+o,\; i,\; j\,]
$$

where:

- $i$ indexes interferometers,
- $j$ indexes pixels,
- $o = n_{\mathrm{delay}} / 2$ is the central delay offset.

These arrays form the whitened detector data vectors at the tested sky point.

## Per-Pixel Network Energy and Active-Pixel Mask

For each pixel `j`, the code computes the total network energy:

$$
e_j = \sum_i \left[\left(v^{00}_{i,j}\right)^2 + \left(v^{90}_{i,j}\right)^2\right].
$$

The active-pixel mask is then

$$
m_j = \mathbf{1}\{e_j > E_{\mathrm{thr}}\}.
$$

This is implemented in `load_data_from_td(...)`, which returns:

- `Eo = \frac{1}{2} \sum_j m_j e_j`
- `NN = \sum_j m_j`
- the per-pixel energy array,
- the active mask.

The factor of $1/2$ follows the cWB convention used throughout the packet-energy bookkeeping.

## Dominant Polarization Frame

### Motivation

For each sky point, the detector response vectors $F_+$ and $F_\times$ define a two-dimensional polarization plane in detector space. The DPF rotates these vectors into a basis that is as close as possible to orthogonal and concentrates the signal into a dominant plus-like axis and an orthogonal cross-like axis.

### Weighted Detector Response

For each pixel `j`, the code forms weighted response vectors:

$$
f_{j,i}^{(0)} = r_{j,i} F_{+,i},
$$

$$
F_{j,i}^{(0)} = r_{j,i} F_{\times,i}.
$$

For each pixel, it computes:

$$
ff_j = \sum_i \left(f_{j,i}^{(0)}\right)^2,
$$

$$
FF_j = \sum_i \left(F_{j,i}^{(0)}\right)^2,
$$

$$
fF_j = \sum_i f_{j,i}^{(0)} F_{j,i}^{(0)}.
$$

Then it defines the rotation auxiliaries:

$$
s_j = 2 fF_j,
$$

$$
c_j = ff_j - FF_j,
$$

$$
A_j = ff_j + FF_j,
$$

$$
n_j = \sqrt{c_j^2 + s_j^2}.
$$

The dominant-polarization energy is

$$
f_{p,j} = \frac{A_j + n_j}{2},
$$

and the rotation angle is encoded via

$$
\cos(2\psi_j) = \frac{c_j}{n_j + \epsilon}.
$$

The code then reconstructs $\sin\psi_j$ and $\cos\psi_j$ and rotates the original response vectors into the DPF basis. This is implemented in `dpf_np_loops_vec(...)`.

### Orthogonalization and Network Index

After the rotation, the code projects out any remaining component of the cross vector along the dominant vector and computes:

$$
f_{x,j} = \lVert F_j^{\perp} \rVert^2.
$$

It also computes the network-index-like quantity

$$
n_i^{(j)} = \frac{\sum_i f_{j,i}^4}{f_{p,j}^2 + \epsilon}.
$$

Finally, the DPF routine returns a sky-point statistic of the form

$$
\mathrm{NI} = \sqrt{\frac{\sum_j f_{x,j} / (n_i^{(j)} + \epsilon)}{N_{+} + 0.01}},
$$

where $N_+$ is the number of pixels with positive $f_{p,j}$. In the code this quantity is used as the DPF quality measure per sky point.

## DPF-Based Energy Regulator

Before the main sky scan, `calculate_dpf(...)` evaluates the DPF quality across all sky points. For each sky location it computes the scalar returned by `dpf_np_loops_vec(...)`, then counts how many sky directions satisfy

$$
\mathrm{DPF}(l) > \gamma_{\mathrm{reg}}.
$$

If the total number of sky points is $N_{\mathrm{sky}}$ and the number above threshold is $N_\gamma$, the code defines the energy regulator

$$
\mathrm{REG}[1] = \left(\frac{N_{\mathrm{sky}}^2}{N_\gamma^2 + 10^{-9}} - 1\right) E_{\mathrm{thr}}.
$$

This mirrors the original cWB logic: if only a small fraction of the sky has a good DPF, the regulator increases.

## Coherent Signal Packet Reconstruction

Once the DPF basis is known for a sky point, the code projects the delayed detector data $(v^{00}, v^{90})$ onto the DPF basis in `avx_GW_ps(...)`.

For each pixel it first forms four detector-space inner products:

$$
x_p = \sum_i v^{00}_{i,j} f_{j,i}, \qquad X_p = \sum_i v^{90}_{i,j} f_{j,i},
$$

$$
x_\times = \sum_i v^{00}_{i,j} F_{j,i}, \qquad X_\times = \sum_i v^{90}_{i,j} F_{j,i}.
$$

The plus-polarization regularized inverse norm is then built as

$$
\alpha_j = \frac{m_j}{f_{p,j} + \max\left(\sqrt{\frac{n_i^{(j)} (x_p^2 + X_p^2)}{e_j + \epsilon}}\,\mathrm{REG}[0] - f_{p,j},\; 0\right) + \epsilon}.
$$

Similarly, the cross-polarization inverse norm is regularized using a dynamic cross regulator based on `REG[1]`:

$$
R_j = 0.1 + \frac{\mathrm{REG}[1]}{e_j + \epsilon}.
$$

This yields amplitude coefficients:

$$
a_u = x_p \alpha_j, \qquad A_u = X_p \alpha_j,
$$

$$
a_v = x_\times \beta_j, \qquad A_v = X_\times \beta_j,
$$

and the reconstructed detector packets:

$$
p^{00}_{i,j} = f_{j,i} a_u + F_{j,i} a_v,
$$

$$
p^{90}_{i,j} = f_{j,i} A_u + F_{j,i} A_v.
$$

The routine also updates the pixel mask according to whether the Gaussian-noise correction remains acceptable.

## Orthogonalization of the Reconstructed Packet

The reconstructed packet is passed to `avx_ort_ps(...)`, which orthogonalizes the two quadratures. For each pixel it computes:

$$
a_j = \sum_i \left(p^{00}_{i,j}\right)^2,
$$

$$
A_j = \sum_i \left(p^{90}_{i,j}\right)^2,
$$

$$
aA_j = \sum_i p^{00}_{i,j} p^{90}_{i,j}.
$$

From these it derives a local rotation that diagonalizes the two quadratures into two orthogonal components with energies:

$$
e_j = \frac{(a_j + A_j) + \sqrt{(a_j - A_j)^2 + (2aA_j)^2}}{2},
$$

$$
E_j = \frac{(a_j + A_j) - \sqrt{(a_j - A_j)^2 + (2aA_j)^2}}{2}.
$$

These are later stored in `SkyStatistics.energy_array_plus` and `SkyStatistics.energy_array_cross`.

## Coherent Statistics per Sky Point

The routine `avx_stat_ps(...)` compares the reconstructed packet `(s, S)` with the delayed data `(v00, v90)`. After a further per-pixel rotation into the orthogonal packet frame, it accumulates:

- coherent energy,
- residual noise,
- Gaussian-noise correction,
- correlation coefficients.

At pixel level the code builds:

$$
ec_j = m_j \left[e_j^{\mathrm{coh}} + E_j^{\mathrm{coh}}\right],
$$

$$
gn_j = 2 m_j \cdot \mathrm{mask}_j,
$$

$$
rn_j = m_j \cdot \text{residual}_j.
$$

It then uses these to form the event-level quantities returned by the kernel:

- `Cr`: a polarization/correlation statistic,
- `Ec`: coherent energy,
- `Mp`: effective number of contributing pixels,
- `No`: residual null energy.

The sky-loop code then constructs:

$$
\chi_{\mathrm{TF}} = \frac{N_o}{n_{\mathrm{IFO}} M_o + \sqrt{M_o}},
$$

$$
c_c = \max(\chi_{\mathrm{TF}}, 1),
$$

$$
C_o = \frac{E_c}{E_c + N_o c_c - M_o (n_{\mathrm{IFO}} - 1)}.
$$

This $C_o$ is the network correlation coefficient in the time-frequency domain for that sky point.

## Sky Statistic and Best Sky Selection

For each sky point, the code forms:

$$
a_a = \max(E_o - N_o, 0),
$$

$$
A_A = a_a C_o.
$$

The quantity $A_A$ is stored as `nSkyStat` and is the quantity maximized in the sky scan. The final sky index is chosen with the same tie-breaking rule as C++: if multiple sky points share the same maximum value, the last one in the forward scan wins.

Additional per-sky quantities stored in `SkyMapStatistics` are:

- antenna prior:

$$
g_{\mathrm{net}}(l) = \sqrt{\langle f_p \rangle_E + \langle f_x \rangle_E}
$$

- alignment factor:

$$
a_{\mathrm{net}}(l) = \sqrt{\frac{\langle f_x \rangle_E}{\langle f_p \rangle_E}}
$$

where the averages are energy-weighted over active pixels.

After the sky scan, a normalized sky probability map is produced from `nSkyStat` using a softmax:

$$
P(l) = \frac{\exp(s_l - s_{\max})}{\sum_{l'} \exp(s_{l'} - s_{\max})},
$$

where $s_l = \mathrm{nSkyStat}[l]$.

## Detailed Statistics at the Best Sky Location

After selecting `l_max`, `calculate_sky_statistics(...)` recomputes the best-sky quantities and packages them into the `SkyStatistics` dataclass.

In addition to the sky-loop quantities (DPF, GW reconstruction, orthogonalization, coherent statistics), it computes additional packet-level quantities described in the following sections.

### Packet Rotation and Amplitude Decomposition (`avx_packet_ps`)

This Numba-compiled kernel decomposes both the data packet $(v^{00}, v^{90})$ and the signal packet $(p^{00}, p^{90})$ into per-IFO amplitude and rotation components. For each interferometer $i$, it accumulates the masked inner products:

$$
aa_i = \sum_j m_j \left(v^{00}_{i,j}\right)^2, \quad
AA_i = \sum_j m_j \left(v^{90}_{i,j}\right)^2, \quad
aA_i = \sum_j m_j \, v^{00}_{i,j} v^{90}_{i,j}.
$$

From these it derives a per-IFO rotation angle and amplitudes:

$$
\tilde{s}_i = 2\,aA_i, \quad \tilde{c}_i = aa_i - AA_i, \quad \tilde{x}_i = aa_i + AA_i + \epsilon,
$$

$$
\tilde{n}_i = \sqrt{\tilde{c}_i^2 + \tilde{s}_i^2}.
$$

The two orthogonal component amplitudes are:

$$
a_i = \sqrt{\frac{\tilde{x}_i + \tilde{n}_i}{2}}, \quad
A_i = \sqrt{\left|\frac{\tilde{x}_i - \tilde{n}_i}{2}\right|}.
$$

The rotation sine and cosine are extracted from $\cos(2\psi_i) = \tilde{c}_i / (\tilde{n}_i + \epsilon)$:

$$
\sin\psi_i = \sqrt{\frac{1 - \cos(2\psi_i)}{2}}, \quad
\cos\psi_i = \mathrm{sign}(\tilde{s}_i) \sqrt{\frac{1 + \cos(2\psi_i)}{2}}.
$$

The per-IFO energy is $E_i = (a_i + A_i)^2 / 2$, and the total packet energy is $E_p = \frac{1}{2}\sum_i E_i$.

The packet is then rotated and normalized pixel-wise:

$$
\hat{v}^{00}_{i,j} = m_j \frac{v^{00}_{i,j} \cos\psi_i + v^{90}_{i,j} \sin\psi_i}{a_i + \epsilon},
$$

$$
\hat{v}^{90}_{i,j} = m_j \frac{v^{90}_{i,j} \cos\psi_i - v^{00}_{i,j} \sin\psi_i}{A_i + \epsilon}.
$$

This produces unit-normalized, rotation-aligned packets suitable for xtalk-corrected norm computation.

### Packet Amplitude Setting (`avx_setAMP_ps`)

After signal reconstruction, `avx_setAMP_ps(...)` sets waveform-reconstruction amplitudes by rotating the signal packets $(p, q)$ using the per-IFO rotation parameters $(\sin\psi_i, \cos\psi_i, a_i, A_i)$ from `avx_packet_ps`. The transformation is:

$$
n_{i,j} = (a_i + A_i)\, \frac{m_j}{2}\, q^{\mathrm{norm}}_{i,j},
$$

$$
p'_{i,j} = n_{i,j} \left(p_{i,j}\cos\psi_i - q_{i,j}\sin\psi_i\right),
$$

$$
q'_{i,j} = n_{i,j} \left(q_{i,j}\cos\psi_i + p_{i,j}\sin\psi_i\right).
$$

The function returns the effective number of degrees of freedom per detector:

$$
N_p = \frac{4}{n_{\mathrm{IFO}}} \sum_j \frac{m_j}{2} \sum_i q^{\mathrm{norm}}_{i,j}.
$$

### Null Stream Computation (`avx_loadNULL_ps`)

The null stream is the residual after subtracting the reconstructed signal from the data:

$$
n^{00}_{i,j} = d^{00}_{i,j} - h^{00}_{i,j}, \qquad
n^{90}_{i,j} = d^{90}_{i,j} - h^{90}_{i,j},
$$

where $(d, D)$ are the data packets and $(h, H)$ are the signal packets. The null stream captures noise plus any signal component not captured by the coherent reconstruction.

## Cross-Talk-Corrected Packet Norms

The likelihood code uses an `XTalk` catalog to account for overlaps between neighboring wavelet pixels. If $x_{ik}^{(ab)}$ denotes one of the four cross-talk coefficients between pixel $i$ and pixel $k$ ($a,b \in \{0,1\}$ for the two quadrature phases), then packet energies are not simple diagonal sums. Instead they have the general form

$$
T_i = \sum_{k \in \mathcal{N}(i)} \left[x_{ik}^{(00)} \mathbf{p}_k \cdot \mathbf{p}_i + x_{ik}^{(01)} \mathbf{p}_k \cdot \mathbf{q}_i + x_{ik}^{(10)} \mathbf{q}_k \cdot \mathbf{p}_i + x_{ik}^{(11)} \mathbf{q}_k \cdot \mathbf{q}_i\right],
$$

where $\mathbf{p}_i, \mathbf{q}_i$ denote the $n_{\mathrm{IFO}}$-dimensional detector packet vectors for pixel $i$, and $\mathcal{N}(i)$ is the set of neighbors of pixel $i$ in the xtalk catalog.

The three norm routines differ in how they process this bilinear form:

### `xtalk_energy_sum_numpy` (Raw Xtalk Energy)

This computes the total xtalk-convolved energy without clamping:

$$
E_{\mathrm{xt}} = \sum_i m_i \sum_d \max(T_{i,d}, 0),
$$

where $d$ indexes detectors and $T_{i,d}$ is the per-detector contribution. Only positive terms are accumulated (matching the C++ `_avx_norm_ps` with `I<0` branch). This is used for the data energy $E_m$ and null energy $N_p$ in the pixel-domain statistics.

### `packet_norm_numpy` (Detector-Wise Norm)

For each pixel $i$, the xtalk kernel computes:

$$
T_{i,d} = \sum_{k \in \mathcal{N}(i)} \left[(x_{ik}^{(00)} p_{d,k} + x_{ik}^{(10)} q_{d,k}) p_{d,i} + (x_{ik}^{(01)} p_{d,k} + x_{ik}^{(11)} q_{d,k}) q_{d,i}\right].
$$

Each per-detector contribution is clamped: $T^+_{i,d} = \max(T_{i,d}, 0)$. The detector norms are accumulated:

$$
\mathrm{norm}_d = \sum_i m_i\, T^+_{i,d}, \qquad \mathrm{norm}_d \leftarrow \max(\mathrm{norm}_d, 2).
$$

The per-pixel q-norm is:

$$
q^{\mathrm{norm}}_{d,i} = \begin{cases}
\frac{p_{d,i}^2 + q_{d,i}^2}{T^+_{i,d} + \epsilon} & \text{if } \frac{p_{d,i}^2 + q_{d,i}^2}{T^+_{i,d} + \epsilon} < 1, \\
0 & \text{otherwise.}
\end{cases}
$$

The halo residual is:

$$
rn_i = \sum_d (u_{i,d}^2 + v_{i,d}^2), \quad \text{where } u_{i,d} = \sum_k (x_{ik}^{(00)} + x_{ik}^{(10)}) p_{d,k}, \; v_{i,d} = \sum_k (x_{ik}^{(01)} + x_{ik}^{(11)}) q_{d,k}.
$$

The detector SNR is:

$$
\mathrm{SNR}_d = \frac{2\,E_d}{\mathrm{norm}_d},
$$

where $E_d$ is the per-IFO energy from packet decomposition.

### `gw_norm_numpy` (Signal Norm)

Starting from the q-norms from `packet_norm_numpy`, this function computes signal-specific norms:

$$
p^{\mathrm{norm}}_{d,i} = \begin{cases}
q^{\mathrm{norm}}_{d,i} & \text{if } ec_i > 0, \\
0 & \text{otherwise,}
\end{cases}
$$

where $ec_i$ is the coherent energy at pixel $i$. The signal SNR per detector is:

$$
\mathrm{SNR}^{\mathrm{sig}}_d = \frac{2\,p^E_d}{q^E_d},
$$

where $p^E_d$ corresponds to the signal packet energy and $q^E_d$ to the data packet norm. The total signal SNR is $\sum_d \mathrm{SNR}^{\mathrm{sig}}_d$.

## Noise Decomposition

`avx_noise_ps(...)` decomposes the event into core, satellite, and signal-support regions using masks derived from the packet norms and the reconstructed packet.

Define the per-pixel average signal and norm energies:

$$
ns_j = \frac{1}{n_{\mathrm{IFO}}} \sum_d p_{d,j}, \qquad nx_j = \frac{1}{n_{\mathrm{IFO}}} \sum_d q_{d,j},
$$

and the masks:

$$
\mathrm{core}_j = m_j \cdot \mathbf{1}\{nx_j > 0\}, \quad
\mathrm{halo}_j = m_j - \mathrm{core}_j, \quad
\mathrm{sig}_j = m_j \cdot \mathbf{1}\{ns_j > 0\}, \quad
\mathrm{sat}_j = m_j - \mathrm{sig}_j.
$$

The returned quantities are:

- $G_n = \sum_j \mathrm{sig}_j \, gn_j \, nx_j$ — Gaussian-noise correction (with $gn_j$ updated in-place),
- $E_c = \sum_j \mathrm{core}_j \, ec_j$ — core coherent energy,
- $D_c = S_c - E_c$ where $S_c = \sum_j \mathrm{sig}_j \, ec_j$ — signal-core coherent energy deficit,
- $R_c = \frac{\sum_j \mathrm{sig}_j \, rc_j}{\sum_j \mathrm{sig}_j \, ec_j + 0.01}$ — coherent-energy normalization ratio,
- $E_h = \frac{1}{2}\sum_j \mathrm{sat}_j \, et_j$ — halo energy,
- $E_s = \frac{1}{2}\sum_j \mathrm{halo}_j \, rn_j$ — satellite residual energy,
- $N_c = \sum_j \mathrm{core}_j$ — number of core pixels,
- $N_s = \sum_j \mathrm{sig}_j$ — number of signal pixels.

The piecewise rule for $rc_j$ is:

$$
rc_j = \begin{cases}
\frac{ec_j}{1} & \text{if } gn_j < 2, \\
\frac{ec_j}{gn_j \cdot 0.5} & \text{otherwise.}
\end{cases}
$$

## Polar Coordinate Projection (`avx_pol_ps`)

The `avx_pol_ps(...)` routine projects packet vectors into polar coordinates in the DPF basis. Given per-pixel DPF response vectors $\mathbf{f}_j, \mathbf{F}_j$ (antenna patterns) and data packets $\mathbf{p}_j, \mathbf{q}_j$, it computes four dot products per pixel:

$$
x_{p,j} = \sum_i f_{j,i}\, m_j\, p_{i,j}, \quad
X_{p,j} = \sum_i f_{j,i}\, m_j\, q_{i,j},
$$

$$
x_{\times,j} = \sum_i F_{j,i}\, m_j\, p_{i,j}, \quad
X_{\times,j} = \sum_i F_{j,i}\, m_j\, q_{i,j}.
$$

The polar radii and angles for each quadrature are:

$$
r_j^{00} = \sqrt{\frac{x_{p,j}^2}{f_{p,j}+\epsilon} + \frac{x_{\times,j}^2}{f_{x,j}+\epsilon}}, \qquad
\theta_j^{00} = \arctan2\!\left(\frac{x_{\times,j}}{\sqrt{f_{x,j}}+\epsilon},\;\frac{x_{p,j}}{\sqrt{f_{p,j}}+\epsilon}\right),
$$

$$
r_j^{90} = \sqrt{\frac{X_{p,j}^2}{f_{p,j}+\epsilon} + \frac{X_{\times,j}^2}{f_{x,j}+\epsilon}}, \qquad
\theta_j^{90} = \arctan2\!\left(\frac{X_{\times,j}}{\sqrt{f_{x,j}}+\epsilon},\;\frac{X_{p,j}}{\sqrt{f_{p,j}}+\epsilon}\right).
$$

The outputs `(pol00, pol90)` respectively store `(radius, angle)` per pixel.

### Projection to Network Plane (PnP) and Dual Stream Phase (DSP)

After computing polar coordinates, the routine applies two additional transforms:

**PnP (network-plane projection):** The data is projected back onto the DPF plane:

$$
p'_{i,j} = f_{j,i} \frac{x_{p,j}}{f_{p,j}+\epsilon} + F_{j,i} \frac{x_{\times,j}}{f_{x,j}+\epsilon}, \qquad
q'_{i,j} = f_{j,i} \frac{X_{p,j}}{f_{p,j}+\epsilon} + F_{j,i} \frac{X_{\times,j}}{f_{x,j}+\epsilon}.
$$

**DSP (dual-stream phase alignment):** The 00/90 components are rotated into a common phase frame:

$$
N_j = \sqrt{c_{p,j}^2 + C_{p,j}^2}, \quad c_j = \frac{c_{p,j}}{N_j + \epsilon}, \quad C_j = \frac{C_{p,j}}{N_j + \epsilon},
$$

$$
\hat{p}_{i,j} = p'_{i,j} c_j + q'_{i,j} C_j, \qquad
\hat{q}_{i,j} = q'_{i,j} c_j - p'_{i,j} C_j,
$$

where $c_{p,j}$ and $C_{p,j}$ are the PnP-projected plus-polarization coefficients from the 00 and 90 quadratures respectively.

These projections are applied twice in `calculate_sky_statistics`: once to the data packets and once to the signal packets, yielding the polar-coordinate representations used for ellipticity and polarization outputs.

## Rho Computation at Best Sky Point

Within `calculate_sky_statistics(...)`, the detection statistic $\rho$ is computed from the xtalk-corrected energies:

$$
E_m = \texttt{xtalk\_energy\_sum\_numpy}(\hat{p}^{\mathrm{data}}, \hat{q}^{\mathrm{data}}, \ldots), \quad
N_p = \texttt{xtalk\_energy\_sum\_numpy}(\hat{p}^{\mathrm{null}}, \hat{q}^{\mathrm{null}}, \ldots),
$$

where the data packets come from the polar-projected data and the null packets from the polar-projected null stream. Then:

$$
\rho^2 = \frac{E_m - N_p}{2}.
$$

This $\rho$ is further modified by `fill_detection_statistic` depending on the mode (2G vs XGB), as described below.

## Time-Domain Waveform Reconstruction

`fill_detection_statistic(...)` performs an additional waveform-domain reconstruction using `get_MRA_wave(...)` from `pycwb.modules.reconstruction.getMRAwaveform`.

For each interferometer $i$, it reconstructs:

- the whitened signal waveform $z^{\mathrm{sig}}_i(t)$,
- the whitened data waveform $z^{\mathrm{dat}}_i(t)$,
- the physical-strain signal waveform $z^{\mathrm{phys}}_i(t)$.

From these it computes:

$$
\mathrm{sSNR}_i = \sum_t \left(z^{\mathrm{sig}}_i(t)\right)^2,
$$

$$
\mathrm{snr}_i = \sum_t \left(z^{\mathrm{dat}}_i(t)\right)^2,
$$

$$
\mathrm{null}_i = \sum_t \left(z^{\mathrm{dat}}_i(t) - z^{\mathrm{sig}}_i(t)\right)^2.
$$

The per-IFO cross energy is then defined as the geometric mean:

$$
\mathrm{xSNR}_i = \sqrt{\max(\mathrm{snr}_i \cdot \mathrm{sSNR}_i, 0)}.
$$

This waveform reconstruction is used for the final cluster-level quantities because it preserves the full wavelet overlap structure better than a pixel-only approximation.

### Time and Frequency Centroids

The code computes energy-weighted time and frequency centroids for each IFO waveform using `getWFtime` and `getWFfreq` (equivalent to C++ `get_CWtime` / `get_CWfreq`):

**Time centroid:**

$$
t_c = \frac{\sum_t t \cdot z^2(t)}{\sum_t z^2(t)}.
$$

**Frequency centroid** (via FFT):

$$
f_c = \frac{\sum_k f_k \,|Z(f_k)|^2}{\sum_k |Z(f_k)|^2},
$$

where $Z(f_k)$ is the FFT of the reconstructed waveform.

The cluster-level `c_time` and `c_freq` are arrays of these centroids across IFOs.

## Xtalk Double-Loop in `fill_detection_statistic`

After waveform reconstruction, the code refines pixel-level flags and energies using a double loop over pixel pairs with xtalk corrections. This is the computationally expensive $O(n_{\mathrm{core}}^2)$ section.

### Pixel Flag Assignment (First Pass)

Each pixel is classified into three categories based on coherent energy and mask:

- **Core pixel**: $ec_j > 0$ and coherent energy above threshold
- **Likelihood pixel**: contributes to likelihood but below core threshold
- **Null pixel**: contributes to null stream only

### Xtalk Energy Corrections (Second Pass)

For the null stream, the code iterates over all pixels $i$ flagged as "null," and for each computes:

$$
E^{\mathrm{null}}_i = \sum_{k \in \mathcal{K}(i)} \left[x_{ik}^{(00)}\, p_k p_i + x_{ik}^{(01)}\, p_k q_i + x_{ik}^{(10)}\, q_k p_i + x_{ik}^{(11)}\, q_k q_i\right],
$$

where $\mathcal{K}(i)$ is the set of pixels in the null-flag set that are in the xtalk neighborhood of $i$. The same structure applies to the likelihood-flagged pixel set.

This double loop has complexity $O(n_{\mathrm{core}} \times |\mathcal{K}|)$ where $|\mathcal{K}|$ is the average neighborhood size from the xtalk catalog (typically 5–20 neighbors per pixel).

## Cluster-Level Detection Statistics

Once the detailed best-sky quantities are available, `fill_detection_statistic(...)` writes the final event-level fields into `cluster.cluster_meta`.

The main definitions used by the code are:

### Subnetwork Statistic

Let $E_{\max}$ be the largest per-IFO signal energy and let

$$
E_{\mathrm{sub}} = \sum_i \mathrm{sSNR}_i - E_{\max}.
$$

The code then uses

$$
E_{\mathrm{sub}} \leftarrow E_{\mathrm{sub}} \left(1 + \frac{2 R_c E_{\mathrm{sub}}}{E_{\max}}\right)
$$

and

$$
N_{\max} = G_n + N_p - N_{\mathrm{eff}} (n_{\mathrm{IFO}} - 1)
$$

to define

$$
\mathrm{subnet} = \frac{E_{\mathrm{sub}}}{E_{\mathrm{sub}} + N_{\max}}.
$$

### Time-Domain Chi-Squared and Correlations

Let:

- $E_w = \sum_i \mathrm{snr}_i$,
- $N_w = \sum_i \mathrm{null}_i$,
- $N_{\mathrm{eff}}$ be the effective number of pixels.

The waveform-domain chi-squared-like factor is:

$$
\chi_{\mathrm{TD}} = \frac{N_w + G_n}{N_{\mathrm{eff}} n_{\mathrm{IFO}}}.
$$

The code defines a correlation correction for the reduced network correlation:

$$
c_{C_r} = \begin{cases}
1 + (\chi_{\mathrm{TD}} - 1) 2 (1 - R_c), & \chi_{\mathrm{TD}} > 1 \\
1, & \chi_{\mathrm{TD}} \le 1
\end{cases}
$$

and then computes:

$$
C_p = \frac{E_c R_c}{E_c R_c + (D_c + N_w + G_n) - N_{\mathrm{eff}} (n_{\mathrm{IFO}} - 1)},
$$

$$
C_r = \frac{E_c R_c}{E_c R_c + (D_c + N_w + G_n)c_{C_r} - N_{\mathrm{eff}} (n_{\mathrm{IFO}} - 1)}.
$$

These become:

- `net_cc = C_p`
- `sky_cc = C_r`

### Packet Norm

The final normalization stored in the cluster is based on

$$
\mathrm{norm}_{\mathrm{TD}} = \max\left(\frac{E_o - E_h}{E_w}, 1\right),
$$

and the code stores `cluster_meta.norm = 2 * norm_TD`.

### Rho

In standard 2G mode, the code stores a reduced rho:

$$
\rho_{\mathrm{red}} = \frac{\rho}{\sqrt{\max(\chi_{\mathrm{TD}}, 1)}}.
$$

Then:

- `net_rho = rho_red`
- `net_rho2 = raw rho`

In XGB mode the convention is swapped to mirror the C++ event output:

- `net_rho = raw rho`
- `net_rho2 = rho / sqrt(max(chi_TD, 1))`

## Chirp Mass Estimation (`get_chirp_mass`)

The chirp mass is estimated via a Hough-transform-based algorithm that searches for a frequency evolution consistent with an inspiraling binary system. This section was previously undocumented.

### Hough Transform over Mass Grid

Given $n_{\mathrm{pix}}$ pixels with central times $t_j$ and frequencies $f_j$, the algorithm evaluates a grid of 1001 trial chirp masses $\mathcal{M}_k$ logarithmically spaced between configurable bounds.

For each trial mass $\mathcal{M}_k$, the predicted time at pixel $j$ given its frequency is:

$$
t^{\mathrm{pred}}_j(\mathcal{M}_k) = C \left(\frac{\mathcal{M}_k}{1.22}\right)^{-5/3} f_j^{-8/3},
$$

where $C$ is a numerical constant from the leading-order inspiral formula. The Hough accumulator counts the number of pixels whose observed time matches the prediction within a tolerance.

The code evaluates a $\chi^2$-like statistic for each trial mass:

$$
\chi^2(\mathcal{M}_k) = \sum_j \frac{\left(t_j - t^{\mathrm{pred}}_j(\mathcal{M}_k) - t_c\right)^2}{f_j^{-2}},
$$

where $t_c$ is the coalescence time and the $f_j^{-2}$ weighting accounts for the larger time uncertainty at lower frequencies. The mass that minimizes $\chi^2$ is selected as the best estimate.

### Fine Grid Refinement

Around the coarse-grid minimum, a fine grid of 1001 sub-steps is evaluated, and the final $\chi^2$ minimum determines the chirp mass estimate stored in `cluster_meta.chirp`.

### PCA Ellipticity

After the chirp mass fit, the code performs PCA on the time-frequency distribution of pixels weighted by SNR:

$$
\Sigma = \begin{pmatrix}
\sigma_{tt} & \sigma_{tf} \\
\sigma_{tf} & \sigma_{ff}
\end{pmatrix}
$$

The eigenvalue ratio gives an ellipticity measure:

$$
e = \frac{\lambda_2}{\lambda_1},
$$

where $\lambda_1 \ge \lambda_2$ are the eigenvalues. This ellipticity is stored in `cluster_meta.ellipticity` and `SkyMapStatistics.nEllipticity`.

### Chirp-Adjusted Rho

If the chirp mass fit quality is good, `net_rho2` is updated using the chirp-weighted pixel ellipticity, providing a chirp-informed detection statistic.

## Error Region (`get_error_region`)

This function is currently a **stub** (implementation is `pass`). It is a placeholder for future sky-localization error-region computation.

## Threshold Cuts

The cluster is rejected if `threshold_cut(...)` finds any failed condition.

### Standard 2G Mode

The code requires:

$$
L_m > 0,
$$

$$
E_o - E_h > 0,
$$

$$
\frac{E_c R_c}{c_c} \ge \mathrm{netEC}_{\mathrm{thr}},
$$

$$
N_{\mathrm{eff}} \ge 1.
$$

### XGB Mode

The third condition is replaced by

$$
\rho \ge |\mathrm{netRHO}|.
$$

The effective-pixel requirement remains the same.

## Big-Cluster Sky Thinning

The code contains a direct analogue of the C++ `bBB` logic. If the cluster is large enough,

$$
n_{\mathrm{pix}} > n_{\mathrm{RES}} \cdot c_{\mathrm{size}},
$$

and coarse-grid sky tables are available, the sky scan switches to a coarser healpix grid. Importantly, the pixel set is not truncated; only the sky grid changes.

## Stored Outputs

The detailed outputs are split across two dataclasses.

### `SkyMapStatistics`

This stores arrays over the entire sky grid:

- `nAntennaPrior`
- `nAlignment`
- `nLikelihood`
- `nNullEnergy`
- `nCorrEnergy`
- `nCorrelation`
- `nSkyStat`
- `nDisbalance`
- `nNetIndex`
- `nEllipticity`
- `nPolarisation`
- `nProbability`

### `SkyStatistics`

This stores the best-sky quantities only, including:

- coherent and null energies,
- per-pixel masks,
- delayed detector packets,
- reconstructed signal and null packets,
- per-pixel coherent energies,
- polarization-plane projections,
- waveform-domain SNR support arrays.

### `ClusterMeta`

The final event-level fields written by the likelihood stage include:

- `energy`, `energy_sky`, `like_net`, `like_sky`
- `net_ecor`, `norm_cor`, `net_null`, `net_ed`
- `net_cc`, `sky_cc`
- `sub_net`, `sub_net2`
- `net_rho`, `net_rho2`
- `g_net`, `a_net`, `i_net`
- `c_time`, `c_freq`
- `wave_snr`, `signal_snr`, `cross_snr`
- `signal_energy_physical`, `null_energy`
- `chirp`, `ellipticity`

These are later consumed by event-output code such as `network_event.py`.

---

## Loop Dimension Estimates

This section provides typical sizes for each major loop dimension, useful for estimating computational cost and identifying GPU acceleration targets.

| Dimension | Symbol | Typical Range | Description |
|-----------|--------|---------------|-------------|
| Detectors | $n_{\mathrm{IFO}}$ | 2–8 (usually 2–3) | Number of GW interferometers |
| Pixels per cluster | $n_{\mathrm{pix}}$ | 100–10,000 | Time-frequency pixels in a cluster |
| Sky directions | $n_{\mathrm{sky}}$ | 49,152–196,608 | HEALPix order 6 → 49k; order 7 → 196k |
| Time-delay samples | $n_{\mathrm{delay}}$ | 12–60 (`TDSize`) | Delay taps per IFO per pixel |
| Core pixels | $n_{\mathrm{core}}$ | 10–2,000 | Subset passing coherent-energy cut |
| Xtalk neighbors | $|\mathcal{K}|$ | 5–20 per pixel | Average neighbors from xtalk catalog |
| Hough mass grid | $n_m$ | 1,001 (coarse) + 1,001 (fine) | Chirp mass trial values |
| MRA WDM layers | $n_{\mathrm{layers}}$ | ~5–10 | Distinct WDM resolutions |
| Detectors × Pixels | $n_{\mathrm{IFO}} \times n_{\mathrm{pix}}$ | 200–80,000 | Common inner-loop extent |

### Per-Function Loop Costs

| Function | Outer Loop | Inner Loop | Complexity |
|----------|-----------|------------|------------|
| `find_optimal_sky_localization` | $n_{\mathrm{sky}}$ (`prange`) | 4 kernels × $n_{\mathrm{pix}} \times n_{\mathrm{IFO}}$ | $O(n_{\mathrm{sky}} \cdot n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `calculate_dpf` | $n_{\mathrm{sky}}$ (`prange`) | `dpf_np_loops_vec`: $n_{\mathrm{pix}} \times n_{\mathrm{IFO}}$ | $O(n_{\mathrm{sky}} \cdot n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `calculate_sky_statistics` | 1 sky point | $n_{\mathrm{pix}} \times n_{\mathrm{IFO}}$ (many kernels) | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `avx_GW_ps` | $n_{\mathrm{pix}}$ | $n_{\mathrm{IFO}}$ projections + regularization | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `avx_ort_ps` | $n_{\mathrm{pix}}$ | $n_{\mathrm{IFO}}$ inner products | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `avx_stat_ps` | $n_{\mathrm{pix}}$ | $n_{\mathrm{IFO}}$ coherent tests | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `avx_packet_ps` | $n_{\mathrm{IFO}}$ | $n_{\mathrm{pix}}$ accumulation | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `packet_norm_numpy` | $n_{\mathrm{pix}}$ | $|\mathcal{K}| \times n_{\mathrm{IFO}}$ xtalk lookups | $O(n_{\mathrm{pix}} \cdot |\mathcal{K}| \cdot n_{\mathrm{IFO}})$ |
| `fill_detection_statistic` (xtalk) | $n_{\mathrm{core}}$ | $n_{\mathrm{core}} \cap \mathcal{K}$ | $O(n_{\mathrm{core}} \cdot |\mathcal{K}|)$ |
| `fill_detection_statistic` (MRA) | $n_{\mathrm{IFO}}$ | 3× `get_MRA_wave` per IFO | $O(n_{\mathrm{IFO}} \cdot n_{\mathrm{layers}} \cdot n_{\mathrm{pix}})$ |
| `get_chirp_mass` (Hough) | $n_m = 1001$ | $n_{\mathrm{pix}}$ | $O(n_m \cdot n_{\mathrm{pix}})$ |
| `avx_pol_ps` | $n_{\mathrm{pix}}$ | $n_{\mathrm{IFO}}$ dot products | $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ |
| `xtalk_energy_sum_numpy` | $n_{\mathrm{pix}}$ | $|\mathcal{K}| \times n_{\mathrm{IFO}}$ | $O(n_{\mathrm{pix}} \cdot |\mathcal{K}| \cdot n_{\mathrm{IFO}})$ |

### Dominant Cost

The total cost of the sky scan is:

$$
C_{\mathrm{total}} \approx n_{\mathrm{sky}} \times n_{\mathrm{pix}} \times n_{\mathrm{IFO}} \times c_{\mathrm{kernel}},
$$

where $c_{\mathrm{kernel}} \approx 4$ (the number of sequential kernel calls per sky point in `find_optimal_sky_localization`). With $n_{\mathrm{sky}} = 196{,}608$, $n_{\mathrm{pix}} = 1{,}000$, and $n_{\mathrm{IFO}} = 3$, this gives approximately $2.4 \times 10^9$ floating-point operations per cluster — the dominant computational cost.

---

## GPU Acceleration Analysis

This section identifies optimization opportunities for GPU (CUDA/ROCm via JAX or CuPy) acceleration, ordered by expected impact.

### Tier 1: Sky Scan (`find_optimal_sky_localization`) — Highest Impact

**Current implementation:** Numba `@njit(parallel=True)` with `prange(n_sky)`. Each sky direction is independent.

**GPU mapping:**
- **Grid dimension:** $n_{\mathrm{sky}}$ sky directions → GPU blocks (49k–196k blocks, excellent occupancy).
- **Block dimension:** $n_{\mathrm{pix}}$ pixels → GPU threads within each block (100–10k threads, needs tiling).
- **Per-thread work:** Inner products over $n_{\mathrm{IFO}}$ detectors (2–8 values, register-level).

**Memory requirements per sky point:**
- $v^{00}, v^{90}$: $n_{\mathrm{IFO}} \times n_{\mathrm{pix}} \times 4$ bytes each
- DPF arrays: $n_{\mathrm{pix}} \times n_{\mathrm{IFO}} \times 4$ bytes
- For $n_{\mathrm{pix}} = 1000$, $n_{\mathrm{IFO}} = 3$: ~24 KB per sky point → fits in shared memory

**Challenges:**
- Time-delay slicing (`td00[ml[i,l]+o, i, :]`) is a scattered gather — needs coalescing strategy.
- Four sequential kernels (`load_data_from_td → dpf → avx_GW_ps → avx_ort_ps → avx_stat_ps`) could be fused into a single GPU kernel to avoid repeated global memory round-trips.
- The `FP_t, FX_t` antenna pattern tables are read-only and shared across all pixels → texture memory or constant memory candidate.

**Expected speedup:** 50–200× over Numba on CPU (est. from similar GPU sky-scan implementations in GW analysis).

### Tier 2: DPF Computation (`calculate_dpf`) — High Impact

**Current implementation:** Numba `@njit(parallel=True)` with `prange(n_sky)`.

**GPU mapping:** Near-identical structure to the sky scan. Each sky point independently calls `dpf_np_loops_vec` over $n_{\mathrm{pix}} \times n_{\mathrm{IFO}}$.

**Optimization note:** Can be fused with Tier 1 if the DPF is computed as the first stage of each sky-scan iteration rather than as a separate pre-scan pass.

### Tier 3: Xtalk Norm Computation (`packet_norm_numpy`, `xtalk_energy_sum_numpy`)

**Current implementation:** Plain NumPy loops in `packet_norm_numpy` (`@njit`); vectorized NumPy in `xtalk_energy_sum_numpy`.

**GPU mapping:**
- Xtalk lookups are a sparse-matrix–vector product: the xtalk catalog defines a sparse adjacency matrix with 4 coefficients per non-zero entry.
- GPU implementation: convert xtalk catalog to CSR format → use batched SpMV (cuSPARSE) or a custom kernel.
- $n_{\mathrm{pix}} \times |\mathcal{K}| \times n_{\mathrm{IFO}}$ → with $n_{\mathrm{pix}} = 10{,}000$, $|\mathcal{K}| = 10$, $n_{\mathrm{IFO}} = 3$: ~300k operations, moderate parallelism.

**Note:** This is called only at the best sky point (once per cluster), so the absolute time saving is smaller than Tier 1–2. However, for the xtalk double-loop in `fill_detection_statistic`, the $O(n_{\mathrm{core}}^2)$ scaling can become significant for large clusters.

### Tier 4: Hough Transform (`get_chirp_mass`)

**Current implementation:** Python/NumPy loop over 1001 mass values × $n_{\mathrm{pix}}$ pixels.

**GPU mapping:**
- Embarrassingly parallel over the mass grid: each of 1001 trial masses independently evaluates a $\chi^2$ over all pixels.
- 1001 × 10,000 = ~10⁷ operations — good for a single GPU kernel.

**Note:** Only called once per cluster, after all other statistics. Low priority unless chirp mass estimation is on the critical path.

### Tier 5: MRA Waveform Reconstruction

**Current implementation:** Per-IFO sequential WDM inverse transforms via `get_MRA_wave`.

**GPU mapping:**
- Each IFO's waveform reconstruction is independent → $n_{\mathrm{IFO}}$ parallel streams.
- Within each IFO, the WDM inverse transform is a batched 1D operation over resolution layers.
- This is called 3× per IFO (signal_whitened, data_whitened, signal_physical).

**Note:** Moderate benefit. The WDM inverse transform should be ported to JAX/CuPy as part of the `wdm-wavelet` package rather than optimized in isolation.

### Memory Budget Summary

| Array | Shape | Size (float32) | Notes |
|-------|-------|-----------------|-------|
| `td00`, `td90` | $(n_{\mathrm{delay}}, n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | $2 \times 60 \times 3 \times 10{,}000 \times 4 = 14.4$ MB | Input data — must reside on GPU for sky scan |
| `FP_t`, `FX_t` | $(n_{\mathrm{sky}}, n_{\mathrm{IFO}})$ | $2 \times 196{,}608 \times 3 \times 4 = 4.7$ MB | Antenna patterns — read-only |
| `ml` | $(n_{\mathrm{IFO}}, n_{\mathrm{sky}})$ | $3 \times 196{,}608 \times 4 = 2.4$ MB | Delay indices — read-only |
| Sky-loop outputs | $(n_{\mathrm{sky}},) \times 10$ | $196{,}608 \times 10 \times 4 = 7.9$ MB | Per-sky statistics |
| **Total GPU memory** | | **~30 MB** | Well within any modern GPU |

### Recommended Strategy

1. **Phase 1:** Port `find_optimal_sky_localization` to JAX `vmap` + `jit` over sky directions. This covers Tier 1 and Tier 2 simultaneously since the DPF can be fused. Use `jax.lax.map` or manual scan for the time-delay gather.

2. **Phase 2:** Convert xtalk catalog to a sparse JAX representation and port `packet_norm_numpy` + `xtalk_energy_sum_numpy` to sparse JAX ops.

3. **Phase 3:** Port `get_chirp_mass` Hough transform to batched JAX.

4. **Phase 4:** Port `get_MRA_wave` WDM inverse transform to JAX (coordinate with `wdm-wavelet` package).

Note that Phase 1 alone is expected to provide the majority of the speedup since the sky scan dominates total runtime (>90% for large sky grids).

---

## Implementation Notes

### Why the Code Uses Float32 in the Hot Path

The sky loop and DPF kernels are Numba-compiled and operate mostly on `float32` arrays. This matches the original AVX-oriented design closely enough while keeping memory traffic and compilation overhead down.

### Why Setup Is Separated From Per-Cluster Likelihood

`setup_likelihood(...)` isolates computations that are constant across clusters in the same job segment:

- sky-delay tables,
- antenna-pattern tables,
- sky coordinates,
- pretransposed `FP` and `FX` arrays,
- threshold and regulator constants.

This is critical because the expensive part of the likelihood is the repeated sky loop, not the per-cluster Python dispatch.

### Why Waveform Reconstruction Is Used for Final Energy Outputs

The final event quantities are not taken only from the pixel-domain packet sums. Instead the code reconstructs time-domain waveforms with `get_MRA_wave(...)` because wavelet basis overlap makes naive pixel-only energy sums incomplete. This is especially important for:

- `energy`
- `null_energy`
- `signal_energy_physical`
- `hrss` and strain-like outputs downstream

### Why Some Formulas Look Redundant

Several statistics exist in both TF-domain and waveform-domain forms. This is intentional and mirrors the original cWB separation between:

- sky-loop selection statistics,
- packet-domain coherent quantities,
- waveform-domain final event outputs.

The code therefore keeps multiple related energies, each with a distinct role.

## Summary

`likelihoodWP` implements a coherent burst likelihood based on a sky scan over time-shifted, whitened detector data. For each sky direction it:

1. aligns detector TD amplitudes,
2. constructs a dominant polarization frame,
3. reconstructs coherent signal packets,
4. evaluates coherent versus null energy balance,
5. maximizes a sky statistic proportional to coherent likelihood times correlation.

At the best sky point it then computes packet-domain and waveform-domain statistics, applies threshold cuts, estimates chirp mass via Hough transform, and writes final cluster-level outputs used downstream by event reporting and reconstruction.

The Python implementation is not a simplified rewrite. It preserves the same algorithmic structure as the original cWB logic while using vectorized NumPy extraction, Numba-compiled kernels, and explicit waveform reconstruction for the final event energetics.

The dominant computational bottleneck is the sky scan over $O(10^5)$ sky directions, each evaluating $O(10^3{-}10^4)$ pixels across $O(1)$ detectors. GPU acceleration via JAX `vmap`/`jit` of this outer loop is the primary optimization target, with expected speedups of 50–200× and a GPU memory footprint under 30 MB.

---

## Iteration-Order Analysis for GPU Optimization

The current Numba CPU implementation uses loop orderings inherited from the original C++ AVX design. On GPU hardware, the optimal iteration order is fundamentally different due to the SIMT execution model, memory coalescing requirements, and the relative sizes of the three primary dimensions. This section analyzes the algorithmic restructuring needed to extract maximum GPU throughput.

### Dimension Size Hierarchy

The three primary dimensions have very different cardinalities:

| Dimension | Symbol | Typical Size | Role in Current Code |
|-----------|--------|-------------|---------------------|
| Sky directions | $n_{\mathrm{sky}}$ | 49,152–196,608 | Outermost parallel loop (`prange`) |
| Pixels per cluster | $n_{\mathrm{pix}}$ | 100–10,000 | Middle loop (sequential per sky) |
| Detectors | $n_{\mathrm{IFO}}$ | 2–3 (max 8) | Innermost loop (reduction/dot-product) |

On a GPU with $O(10^4)$ active threads, the key constraint is: **the parallelized dimensions must collectively provide $\gg 10^4$ independent work items to saturate occupancy**, while the innermost dimension should map to register-level operations or shared-memory reductions.

### Current Iteration Orders and Their GPU Implications

#### Sky Scan (`find_optimal_sky_localization`)

**Current CPU order:**
```
prange(n_sky)                          ← Parallel over sky
    for i in range(n_ifo):             ← Sequential: delay fetch
        v00[i] = td00[ml[i,l]+o, i]   ← Scattered gather
    load_data_from_td(v00, v90):
        for j in range(n_pix):         ← Sequential over pixels
            for i in range(n_ifo):     ← Reduction over IFOs
                ...accumulate energy
    dpf_np_loops_vec(FP[l], FX[l], rms):
        for j in range(n_pix):         ← Sequential over pixels
            for i in range(n_ifo):     ← Reduction over IFOs
                ...DPF rotation
    avx_GW_ps(v00, v90, f, F, ...):
        for j in range(n_pix):         ← Sequential over pixels
            for i in range(n_ifo):     ← Reduction/dot-product
                ...projection
    avx_ort_ps(ps, pS, mask):
        for j in range(n_pix):         ← Sequential over pixels
            for i in range(n_ifo):     ← Reduction
                ...orthogonalization
    avx_stat_ps(v00, v90, ps, pS, ...):
        for j in range(n_pix):         ← Sequential over pixels
            for i in range(n_ifo):     ← Reduction
                ...coherent statistics
```

**Problem on GPU:** Each sky direction processes $n_{\mathrm{pix}}$ pixels sequentially. If mapped as one GPU thread per sky direction, each thread does $O(n_{\mathrm{pix}} \cdot n_{\mathrm{IFO}})$ work — too much per thread, and the pixel-level parallelism is wasted.

#### Per-Pixel Kernels (inside sky loop)

**Current order in `load_data_from_td`, `avx_GW_ps`, `avx_ort_ps`, `avx_stat_ps`:**
```
for j in range(n_pix):          ← OUTER: pixel
    for i in range(n_ifo):      ← INNER: IFO reduction
        accumulate(...)
    per_pixel_scalar_ops(...)
```

**Current order in `dpf_np_loops_vec`** (Stage 1):
```
for j in range(n_ifo):          ← OUTER: IFO (wrong for C-order!)
    for i in range(n_pix):      ← INNER: pixel
        f[i, j] = rms[i, j] * Fp0[j]
```

This Stage 1 loop writes `f[i, j]` column-by-column in a row-major array — **stride-1 access is along the pixel axis but the write pattern is column-strided**. On GPU this would cause severe uncoalesced writes.

#### Packet-Level Kernels (`avx_packet_ps`)

**Current order:**
```
for j in range(n_ifo):          ← OUTER: IFO
    for i in range(n_pix):      ← INNER: pixel accumulation
        aa[j] += mk[i] * v00[j][i]²
```

This is an IFO-outer, pixel-inner reduction — natural for CPU (accumulate per-IFO scalar), but on GPU the pixel loop should be the parallel dimension with a per-IFO warp-level reduction.

### Proposed GPU Iteration Orders

#### Strategy 1: 2D Grid — Sky × Pixel (Recommended for Sky Scan)

**Proposed GPU layout:**

$$
\text{Grid:} \quad (\lceil n_{\mathrm{sky}} / B_s \rceil,\; \lceil n_{\mathrm{pix}} / B_p \rceil) \\
\text{Block:} \quad (B_s,\; B_p) \quad \text{e.g. } (1, 256) \text{ or } (4, 64)
$$

Each thread computes one (sky, pixel) pair. The IFO dimension ($n_{\mathrm{IFO}} = 2{-}3$) is fully unrolled in registers.

```
# GPU kernel: fused DPF + GW projection + orthogonalization + statistics
kernel[grid, block](sky_idx, pix_idx):
    # Phase 1: Time-delay fetch (per-thread, register-held)
    for i in range(n_ifo):                        # UNROLLED (2-3 iters)
        v00_i = td00[ml[i, sky_idx] + o, i, pix_idx]
        v90_i = td90[ml[i, sky_idx] + o, i, pix_idx]

    # Phase 2: DPF (per-pixel, antenna patterns from sky_idx)
    for i in range(n_ifo):                        # UNROLLED
        f_i = rms[pix_idx, i] * FP[sky_idx, i]
        F_i = rms[pix_idx, i] * FX[sky_idx, i]
    ... DPF rotation (all in registers) ...

    # Phase 3: GW projection (per-pixel dot products)
    xp, XP, xx, XX = 0, 0, 0, 0
    for i in range(n_ifo):                        # UNROLLED
        xp += v00_i * f_i
        ...
    ... regularization (scalar per pixel) ...

    # Phase 4: Per-pixel energy and coherent statistics
    ... (all scalar, register-held) ...

    # Phase 5: Block-level reduction over pixels → per-sky aggregates
    Eo = block_reduce_sum(energy_j)               # Shared-memory reduction
    Ec = block_reduce_sum(ec_j)
    No = block_reduce_sum(rn_j)
    ...
```

**Why this is optimal:**

1. **IFO unrolling:** With $n_{\mathrm{IFO}} = 2{-}3$, the IFO loop is compiled away entirely. Each register holds one IFO's contribution. No loop overhead, no branch divergence.

2. **Pixel parallelism:** 256 threads per block process 256 pixels simultaneously. The block-level reduction (`Eo`, `Ec`, `No`) uses shared memory and completes in $O(\log n)$ steps.

3. **Sky parallelism:** Grid-level parallelism across sky directions. With 196k sky directions and 256 threads/block, the grid has $\sim$196k blocks — excellent occupancy.

4. **Memory coalescing:** Adjacent threads in a warp access adjacent pixels. The key requirement is that the **pixel axis must be the last (contiguous) axis** in all arrays.

**Required layout changes:**

| Array | Current Shape | GPU-Optimal Shape | Reason |
|-------|--------------|-------------------|--------|
| `td00`, `td90` | $(n_{\mathrm{delay}}, n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | $(n_{\mathrm{delay}}, n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | **Already optimal** — pixel-last is contiguous |
| `v00`, `v90` | $(n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | $(n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | **Already optimal** — pixel-last |
| `f`, `F` (DPF) | $(n_{\mathrm{pix}}, n_{\mathrm{IFO}})$ | $(n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | **Must transpose** — current layout has IFO-last; warp-coalesced reads require pixel-last |
| `rms` | $(n_{\mathrm{pix}}, n_{\mathrm{IFO}})$ | $(n_{\mathrm{IFO}}, n_{\mathrm{pix}})$ | **Must transpose** — same reasoning |
| `FP`, `FX` | $(n_{\mathrm{sky}}, n_{\mathrm{IFO}})$ | $(n_{\mathrm{IFO}}, n_{\mathrm{sky}})$ | **Must transpose** — each warp reads the same sky index, so IFO-first allows broadcast from L1 cache |
| `ml` | $(n_{\mathrm{IFO}}, n_{\mathrm{sky}})$ | $(n_{\mathrm{IFO}}, n_{\mathrm{sky}})$ | **Already optimal** — constant across pixels |
| `mask`, `energy`, `ec`, `gn`, `rn` | $(n_{\mathrm{pix}},)$ | $(n_{\mathrm{pix}},)$ | 1D — naturally coalesced |

**The general rule:** arrays with a pixel dimension should have pixel as the **last axis** (contiguous in C-order) so that adjacent GPU threads access adjacent memory. Arrays with a sky dimension but no pixel dimension (e.g. `FP`, `FX`, `ml`) are accessed uniformly within a block, so their layout matters less.

#### Strategy 2: Batched Sky-Vectorized Formulation (JAX `vmap`)

Instead of a 2D CUDA grid, express the computation as pixel-parallel operations `vmap`-ed over sky:

```python
# JAX formulation
def per_sky_kernel(sky_idx, td00, td90, ml, FP, FX, rms, REG):
    # Gather delayed data: (n_ifo, n_pix) per sky direction
    v00 = td00[ml[:, sky_idx] + offset, jnp.arange(n_ifo)[:, None], jnp.arange(n_pix)]
    v90 = td90[ml[:, sky_idx] + offset, jnp.arange(n_ifo)[:, None], jnp.arange(n_pix)]

    # DPF: vectorized over pixels, IFO reduced via dot products
    f = rms * FP[sky_idx][:, None]      # (n_ifo, n_pix) — broadcast
    F = rms * FX[sky_idx][:, None]
    ... (all pixel-parallel operations) ...

    # Reductions over pixels
    Eo = jnp.sum(energy * mask) / 2
    ...
    return sky_statistics

# vmap over sky directions
batched = jax.vmap(per_sky_kernel, in_axes=(0, None, None, None, None, None, None, None))
results = batched(jnp.arange(n_sky), td00, td90, ml, FP, FX, rms, REG)
```

**Effective iteration order after XLA compilation:**

$$
\text{Outer: sky (vmap → batch dim)} \to \text{Middle: pixel (vectorized)} \to \text{Inner: IFO (reduced)}
$$

This is equivalent to Strategy 1 but expressed declaratively. XLA will fuse the operations and select the optimal GPU grid dimensions automatically.

**Advantages over explicit CUDA:**
- Automatic kernel fusion (no manual shared-memory management)
- Device-agnostic (CPU fallback for testing, TPU for future)
- Easier to maintain alongside the Numba CPU path

**Disadvantage:**
- The time-delay gather (`td00[ml[:, sky_idx] + offset, ...]`) is a dynamic index that XLA may not optimize as well as a hand-written CUDA kernel with texture-memory fetches.

#### Strategy 3: Pixel-Outer with Sky Batching (Alternative for Memory-Constrained Cases)

For very large clusters ($n_{\mathrm{pix}} > 10{,}000$), the $(n_{\mathrm{sky}}, n_{\mathrm{pix}})$ intermediate arrays may exceed GPU memory. In this case, chunk the sky loop into batches:

```
for sky_batch in range(0, n_sky, SKY_BATCH_SIZE):
    # Process SKY_BATCH_SIZE sky directions at once
    # Each sky direction: full pixel parallelism
    # Accumulate per-sky outputs
```

With `SKY_BATCH_SIZE = 4096` and $n_{\mathrm{pix}} = 10{,}000$, intermediate arrays per batch are:

$$
4096 \times 10{,}000 \times 3 \times 4 \text{ bytes} = 491 \text{ MB}
$$

This fits comfortably on a 4+ GB GPU. For $n_{\mathrm{pix}} = 50{,}000$ (extreme case), reduce batch size to 1024 → 614 MB.

### Per-Kernel Optimal Iteration Orders

This table summarizes the recommended GPU iteration order for each kernel, compared to the current CPU order:

| Kernel | Current CPU Order | Optimal GPU Order | Parallelism | Reduction |
|--------|------------------|-------------------|-------------|-----------|
| `load_data_from_td` | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO in registers |
| `dpf_np_loops_vec` (Stage 1) | `ifo → pix` | `pix ∥ (ifo unrolled)` | Pixel-parallel | Broadcast `Fp0[j]` |
| `dpf_np_loops_vec` (Stage 2–4) | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO in registers |
| `avx_GW_ps` | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO dot products in registers |
| `avx_ort_ps` | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO in registers |
| `avx_stat_ps` | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO in registers |
| `avx_packet_ps` | `ifo → pix` | `pix ∥ → block_reduce per ifo` | Pixel-parallel | Shared-mem reduction → per-IFO scalars |
| `avx_pol_ps` | `pix → ifo` | `pix ∥ (ifo unrolled)` | Pixel-parallel | IFO in registers |
| `packet_norm_numpy` | `pix → ifo` (with gather) | `pix ∥ (ifo unrolled, sparse gather)` | Pixel-parallel | Xtalk neighbors via shared mem |
| `xtalk_energy_sum_numpy` | `pix → neighbors → ifo` | `pix ∥ (neighbors in shared mem)` | Pixel-parallel | Neighbor gather + IFO unroll |

**Key observation:** In every kernel, the optimal GPU inner dimension is the **pixel axis** (parallel), and the IFO dimension should be **unrolled into registers**. The sky dimension is the **batch/grid dimension**. This is a complete inversion of the original C++ design, which used AVX to vectorize the IFO dimension (4–8 wide SIMD lanes) and looped over pixels and sky sequentially.

### Detailed Restructuring: `avx_packet_ps`

`avx_packet_ps` is the most interesting case because its current loop order (`ifo → pix`) is the exact opposite of what GPU wants. It accumulates per-IFO statistics over all pixels:

$$
aa_i = \sum_j m_j (v^{00}_{i,j})^2
$$

**Current CPU layout (efficient for CPU cache):**
```
for i in range(n_ifo):           # IFO outer — one accumulator in register
    for j in range(n_pix):       # Pixel inner — sequential memory sweep
        aa[i] += mk[j] * v00[i][j]²
```

**Optimal GPU layout:**
```
# Phase 1: Per-pixel, per-IFO partial products (pixel-parallel)
# Each thread j computes mk[j] * v00[i][j]² for all i
partial[i, j] = mk[j] * v00[i, j]²     # (n_ifo, n_pix) — pixel ∥

# Phase 2: Parallel reduction over pixels (shared memory)
aa[i] = block_reduce_sum(partial[i, :])  # One sum per IFO
```

The reduction itself is $O(\log n_{\mathrm{pix}})$ using shared memory, vs $O(n_{\mathrm{pix}})$ on CPU. For $n_{\mathrm{pix}} = 10{,}000$, this is a 13× reduction in serial steps (though the constant factor is different).

The second phase (per-IFO rotation applied to all pixels) maps naturally to pixel-parallel:

```
# Phase 3: Apply rotation (pixel-parallel, IFO unrolled)
for i in range(n_ifo):                              # UNROLLED
    v00_new[i, j] = mk[j] * (v00[i,j]*co[i] + v90[i,j]*si[i]) * a_inv[i]
```

### Detailed Restructuring: Time-Delay Gather

The time-delay fetch is the most challenging operation for GPU coalescing:

```python
# Current: per-IFO sequential gather
for i in range(n_ifo):
    v00[i] = td00[ml[i, sky_idx] + offset, i, :]
```

Each IFO has a **different delay index** `ml[i, sky_idx]`, so the gather addresses are:

$$
\mathrm{addr}(i, j) = \bigl(ml(i, l) + o\bigr) \cdot n_{\mathrm{IFO}} \cdot n_{\mathrm{pix}} + i \cdot n_{\mathrm{pix}} + j
$$

For a fixed sky index and IFO, **adjacent pixels $j, j+1$ are at adjacent addresses** — this is coalesced. But the IFO dimension introduces a stride of $n_{\mathrm{pix}}$ between fetches.

**GPU strategy:** Since $n_{\mathrm{IFO}} = 2{-}3$, issue $n_{\mathrm{IFO}}$ coalesced loads per thread:

```
thread j:
    v00_0 = td00[ml[0, sky_idx] + o, 0, j]   # Coalesced across threads
    v00_1 = td00[ml[1, sky_idx] + o, 1, j]   # Coalesced across threads
    v00_2 = td00[ml[2, sky_idx] + o, 2, j]   # Coalesced (if 3 IFOs)
```

Each load is fully coalesced because adjacent threads access adjacent pixel indices. The delay offset and IFO index are uniform within a warp (all threads in the same block share the same `sky_idx`). This yields $n_{\mathrm{IFO}}$ coalesced global memory transactions per thread — **optimal**.

**Alternative layout** `(n_ifo, n_delay, n_pix)` would allow a single contiguous read per IFO but requires transposing the input. The current `(n_delay, n_ifo, n_pix)` layout is already near-optimal because the stride between different IFOs at the same delay is only $n_{\mathrm{pix}} \times 4$ bytes, which the L2 cache handles well.

### Detailed Restructuring: DPF Kernel Fusion

The DPF computation (`dpf_np_loops_vec`) has 4 sequential stages that access the same `(n_pix, n_ifo)` data. On CPU, these must be separate loops because each stage depends on the previous one's per-pixel output. On GPU, **stages 1–3 can be fused into a single kernel** because each pixel's computation is independent:

```
# Fused GPU kernel: one thread per pixel
thread j:
    # Stage 1: Weighted antenna patterns (IFO unrolled)
    f_0 = rms[0, j] * Fp0[0]
    f_1 = rms[1, j] * Fp0[1]
    F_0 = rms[0, j] * Fx0[0]
    F_1 = rms[1, j] * Fx0[1]

    # Stage 2: Dot products → rotation angle (register arithmetic)
    ff = f_0² + f_1²
    FF = F_0² + F_1²
    fF = f_0*F_0 + f_1*F_1
    ... compute si, co, fp ...

    # Stage 3: Rotate (in-place in registers)
    f_0, F_0 = f_0*co + F_0*si, F_0*co - f_0*si
    f_1, F_1 = f_1*co + F_1*si, F_1*co - f_1*si

    # Stage 4: Orthogonalize
    fF_new = (f_0*F_0 + f_1*F_1) / (fp + ε)
    F_0 -= f_0 * fF_new
    F_1 -= f_1 * fF_new
    fx = F_0² + F_1²
    ni = (f_0⁴ + f_1⁴) / (fp² + ε)

    # Write outputs
    f_out[0, j] = f_0; f_out[1, j] = f_1
    F_out[0, j] = F_0; F_out[1, j] = F_1
    fp_out[j] = fp; fx_out[j] = fx; ni_out[j] = ni
```

**Savings:** Zero intermediate array writes for `f`, `F` between stages. On CPU, the current code writes $4 \times n_{\mathrm{pix}} \times n_{\mathrm{IFO}} \times 4$ bytes to L1/L2 between stages. On GPU with fusion, everything stays in registers.

### Fused Sky-Scan Mega-Kernel

The ultimate optimization is fusing all five per-sky kernels into a single GPU kernel that processes one (sky, pixel) pair per thread and reduces over pixels using shared memory:

$$
\text{load\_data} \to \text{dpf} \to \text{avx\_GW} \to \text{avx\_ort} \to \text{avx\_stat}
$$

**Register budget per thread** (for $n_{\mathrm{IFO}} = 3$):

| Variables | Count | Registers (float32) |
|-----------|-------|-------------------|
| `v00[i]`, `v90[i]` | 6 | 6 |
| `f[i]`, `F[i]` | 6 | 6 |
| `fp`, `fx`, `ni` | 3 | 3 |
| `si`, `co` (DPF) | 2 | 2 |
| `xp`, `XP`, `xx`, `XX` | 4 | 4 |
| `au`, `AU`, `av`, `AV` | 4 | 4 |
| `p[i]`, `q[i]` (signal packet) | 6 | 6 |
| `si_ort`, `co_ort`, `ee`, `EE` | 4 | 4 |
| `ec`, `gn`, `rn`, mask, energy | 5 | 5 |
| Temporaries | ~6 | 6 |
| **Total** | | **~46** |

At 46 registers per thread, a GPU SM with 65,536 registers supports 1,424 concurrent threads — enough for 5–6 full warps, which is adequate occupancy. For $n_{\mathrm{IFO}} = 2$, register count drops to ~34, allowing 8 warps per SM.

The shared memory needed for the per-sky reduction:

$$
\text{Shared mem} = B_p \times k_{\mathrm{reduce}} \times 4 \text{ bytes}
$$

where $k_{\mathrm{reduce}}$ is the number of per-pixel quantities that need block-level reduction (`Eo`, `NN`, `Mo`, `Lo`, `Cr`, `Ec`, `Mp`, `No` = 8 values). With $B_p = 256$:

$$
\text{Shared mem} = 256 \times 8 \times 4 = 8{,}192 \text{ bytes} = 8 \text{ KB}
$$

Well within the 48–164 KB shared memory budget per SM on modern GPUs.

### Memory-Bandwidth Analysis

The fused kernel's global memory traffic per (sky, pixel) pair:

**Reads:**
- `td00`, `td90`: $2 \times n_{\mathrm{IFO}} \times 4$ bytes = 24 B (for 3 IFOs)
- `rms`: $n_{\mathrm{IFO}} \times 4$ B = 12 B
- `FP`, `FX`: $2 \times n_{\mathrm{IFO}} \times 4$ B = 24 B (cached — uniform per block)
- `ml`: $n_{\mathrm{IFO}} \times 4$ B = 12 B (cached — uniform per block)
- `REG`: 12 B (constant memory)

**Writes:**
- Per-pixel: `ec`, `gn`, `rn`, `mask`, `energy`, `f[i]`, `F[i]`, `p[i]`, `q[i]` — only needed if this is the best sky direction (can be deferred)
- Per-sky outputs: 12 float32 values = 48 B (after reduction)

**Per-pixel effective bandwidth:**
- Reads: 24 + 12 = 36 B/pixel (per-pixel data) + 36 B/block-amortized = ~37 B/pixel
- Writes: 48 B/block (negligible per pixel)

**Total bandwidth for one sky scan:**

$$
B = n_{\mathrm{sky}} \times n_{\mathrm{pix}} \times 37 \text{ B}
$$

With $n_{\mathrm{sky}} = 196{,}608$ and $n_{\mathrm{pix}} = 1{,}000$:

$$
B = 196{,}608 \times 1{,}000 \times 37 = 7.3 \text{ GB}
$$

On an A100 (2 TB/s HBM bandwidth), this takes ~3.6 ms. On a consumer RTX 4090 (1 TB/s), ~7.3 ms. **The sky scan becomes memory-bandwidth-limited**, not compute-limited — which is the ideal regime for GPU.

For comparison, the current Numba CPU implementation at ~10 GFLOPS effective throughput takes:

$$
t_{\mathrm{CPU}} \approx \frac{196{,}608 \times 1{,}000 \times 3 \times 20}{10^{10}} \approx 1.2 \text{ s}
$$

The theoretical GPU speedup is thus **160–330×**.

### Summary of Layout Recommendations

For the GPU-optimized implementation, the following array layout conventions should be adopted:

1. **Pixel axis last** (contiguous): All per-pixel arrays should have shape `(..., n_pix)` in C-order. This includes `v00`, `v90`, `td00`, `td90` (already correct), and `f`, `F`, `rms` (currently `(n_pix, n_ifo)`, should become `(n_ifo, n_pix)`).

2. **IFO axis unrolled**: The IFO dimension never appears as a GPU parallel axis. It is always either:
   - Unrolled in registers (inside per-pixel computation), or
   - Reduced via warp-shuffle / shared-memory (for `avx_packet_ps`-style accumulations).

3. **Sky axis as batch dimension**: Sky directions map to GPU blocks (or `vmap` batch in JAX). Within each block, all pixel-level work is parallelized.

4. **No intermediate materialization**: On CPU, the current code writes `v00`, `v90` (extracted per sky direction), then `f`, `F` (DPF output), then `ps`, `pS` (GW projection output). On GPU, all intermediates should live in registers within the fused kernel. Only the final per-sky statistics and the best-sky pixel arrays are written to global memory.

### Precision Decision

All inputs and accumulators should use **FP32** for the initial GPU implementation. The dominant speedup (160–330×) comes from the GPU port itself, not from precision reduction. FP32 guarantees exact numerical parity with the Numba CPU reference, avoids DPF angle quantization errors, and requires no per-tensor scale factors or injection-based validation. Reduced-precision storage (BF16 for `td00`, `td90`, `FP`, `FX`) is a worthwhile follow-on optimization after the FP32 baseline is validated, since the conversion instruction overhead is hidden behind HBM latency in this bandwidth-limited kernel. `rms` must remain FP32 unconditionally — it directly determines DPF rotation angles and is the most precision-sensitive input.
