import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


window='blackman'
nperseg=1000
noverlap=800
nfft=10000
scaling='density'



T = 10
sampleRate = 1e3

#signals
t = np.arange(0,T,1/sampleRate)
f1 = 100
f2 = 75
sig1 = 1*np.sin(2*math.pi*f1*t)
sig2 = 10*np.sin(2*math.pi*f2*t + math.pi/2)

#noise term
variance = 1
sig = np.sqrt(variance)
rng = np.random.default_rng(seed =0)
noise1 = rng.normal(loc = 0, scale = sig, size = t.shape)
noise2 = rng.normal(loc = 0, scale = sig, size = t.shape)
noise3 = rng.normal(loc = 0, scale = sig, size = t.shape)

dataSize = len(t)


#plotting
fig, (ax_f, ax_w) = plt.subplots(1, 2, figsize=(16, 4))
fig.suptitle(f"Window: {window}, nperseg = {nperseg}, noverlap = {noverlap}, nfft = {nfft}, scaling: {scaling}, Length of Data = {T}", fontsize=14)

#fft
for sig, label in [(noise1,'real1'),(noise2,'real2'),(noise3,'real3')]:
    Y = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(dataSize, d=1/sampleRate)

    P = np.abs(Y)**2/(sampleRate*dataSize)
    db = 10*np.log10(P)
    ax_f.plot(freqs, db, label='MultiSignal(FFT)')
    
ax_f.set_xlabel('Frequency (Hz)')
ax_f.set_ylabel('Power (dB)')
ax_f.set_title('Spectrum (dB)')
ax_f.set_xlim(0, sampleRate/2)
ax_f.grid(True)


#welsh
for sig, label in [(noise1,'real1'),(noise2,'real2'),(noise3,'real3')]:
    f, Pxx = signal.welch(sig, fs=sampleRate, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling=scaling)  
    ax_w.plot(f, Pxx, label='MultiSignal(Welsh)')
    
ax_w.set_xlabel('Frequency (Hz)')
ax_w.set_ylabel('Power (dB)')
ax_w.set_title('Spectrum (dB) Welsh')
ax_w.set_xlim(0, sampleRate/2)
ax_w.grid(True)



plt.tight_layout()
plt.show()