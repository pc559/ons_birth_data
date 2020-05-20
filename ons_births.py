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
est_PS = np.abs(birth_fft)**2
est_PS = est_PS[1:]

birth_fft_smthd = np.fft.fft(birth_data_smthd)
est_PS_smthd = np.abs(birth_fft_smthd)**2
est_PS_smthd = est_PS_smthd[1:]

ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ks = np.arange(len(FFT))/len(FFT)
xs = 1./ks[1:]
ax1.plot(xs, est_PS, '-', label='Birth Data Power Spectrum')
ax1.plot(xs, est_PS_smthd, '-', label='Smoothed')
ax1.plot(xs, est_PS-est_PS_smthd, '-', label='Difference')
ax1.legend()

ax2.plot(cont_days, recon, '-', label='Reconstructed from FFT')
ax2.plot(days, birth_data, 'o', markersize=2, label='Birth Data')
ax2.plot(days, birth_data_smthd, '-', label='Smoothed')
ax2.legend()

plt.show()

