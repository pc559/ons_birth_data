## # Despite the averaging, there is still a large signal at 75 and 125 days.
## # Is this real, eg more babies born in some season...?
## # Should probably add k-dep error bars, coming from the fact this is only
## # the average of a few years.
## # It's probably that, because the xmas signal is still there, and very strong,
## # it hasn't been averaged away yet.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

data = pd.read_csv('ons_births_data.csv')
print(data.head())

#data.plot('date','average')
#plt.show()

N = len(data['average'])
FFT = np.fft.fftfreq(N)*N

birth_data = data['average']
birth_data_smthd = savgol_filter(birth_data, 33, 3)
days = np.arange(len(birth_data))
## # It makes no sense to pretend this is continuous,
## # but it at leasts allows for sanity checking the
## # FFT result.
cont_days = np.linspace(0, len(birth_data), 10*len(birth_data))

birth_fft = np.fft.fft(birth_data)
## # Could use ifft for this,
## # but wanted to explicitly check
## # the k-space interpretation is correct.
expon = np.einsum('i,j->ij', cont_days, FFT)*2*np.pi*1j/len(FFT)
recon = np.einsum('i,ji->j', birth_fft, np.exp(expon))/len(FFT)
## # Why are there imag parts at rel 1e-5?
recon = np.real_if_close(recon, tol=1e-4*np.mean(np.abs(recon.real)))
assert (np.isreal(recon)).all()
## # Calling this a "power spectrum" doesn't
## # really make sense, but whatever.
est_PS = np.abs(birth_fft)**2
est_PS = est_PS[1:]

birth_fft_smthd = np.fft.fft(birth_data_smthd)
est_PS_smthd = np.abs(birth_fft_smthd)**2
est_PS_smthd = est_PS_smthd[1:]

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.plot(days, birth_data, 'o', markersize=2, label='Birth Data')
ax1.plot(days, birth_data_smthd, '-', label='Smoothed')
ax1.plot(cont_days, recon, '-', label='Reconstructed from FFT (consistency test)')
ax1.legend()

ks = np.arange(len(FFT))/len(FFT)
xs = 1./ks[1:]
skip_ends = slice(1, -1)

ax2.plot(xs[skip_ends], est_PS[skip_ends], '-', label='Birth Data $|FFT|^2$')
ax2.plot(xs[skip_ends], est_PS_smthd[skip_ends], '-', label='Smoothed')
ax2.plot(xs[skip_ends], est_PS[skip_ends]-est_PS_smthd[skip_ends], '-', label='Difference')
ax2.vlines(7, np.min(est_PS[skip_ends]), np.max(est_PS[skip_ends]), colors='k', linestyles='dashed', label='7 days', alpha=0.5)
ax2.legend()

plt.show()

