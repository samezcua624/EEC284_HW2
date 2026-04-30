"""
EEC 284 Assignment 3 Problem 2
TFO Signal Analysis
Author: Sam Amezcua
Instructions:
  - Complete all "TODO" sections
  - Do not modify function signatures
  - Submit this file with all functions completed and working
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, find_peaks, sosfiltfilt


def estimate_carrier_frequencies(signal, Fs):
    
    #input and setup
    f1 = 0.0
    f2 = 0.0
    f1_index = 0
    f2_index = 0
    signal = signal.ravel()
    N = len(signal)
    

    # hpf above 10hz (I was having issues with a DC peak at 0 getting recognized)
    sos = butter(4, 10/(0.5*Fs), btype='highpass', output='sos')
    hpf_sig = sosfiltfilt(sos,signal)
    
    #set up fft
    Y = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1/Fs)

    P = np.abs(Y)**2/(Fs*N)
    db = 10*np.log10(P)
    
    #peak finder setup
    df = freqs[1] - freqs[0]
    min_prominence_hz = 1.0
    
    #tuning the peak finder
    min_prom = np.median(db) * 6.5 
    distance = int(np.ceil(min_prominence_hz / df))
    
    #find peaks and sort them by magnitude
    peaks, props = find_peaks(db, prominence=min_prom, distance=distance)
    sorted = np.argsort(-db[peaks])[:2]
    f1f2 = peaks[sorted][:2]
    f1 = freqs[f1f2[0]]
    f2 = freqs[f1f2[1]]

    return f1,f2


def iq_demodulate(signal, fc, Fs):
    
    #input
    signal = signal.ravel()
    N = len(signal)
    T = N/Fs
    
    #set up time axis
    t = np.arange(0,T,1/Fs)
    
    
    #set up bandpass around fc (+/- 10Hz)
    low = fc - 15
    high = fc + 15
    nyquist = 0.5 * Fs
    
    #bandpass
    sos_bp = butter(4, [low/nyquist, high/nyquist], btype='band', output='sos')
    bp_sig = sosfiltfilt(sos_bp,signal)

    
    # I and Q setup
    I = np.sin(2*np.pi*fc*t)
    Q = np.cos(2*np.pi*fc*t)
    
    I_mod = I*bp_sig
    Q_mod = Q*bp_sig
    
    
    
    #LPF I&Q
    I_mod_lpf = lowpass_filter(I_mod,15,Fs)
    Q_mod_lpf = lowpass_filter(Q_mod,15,Fs)
    
    mag = I_mod_lpf**2 + Q_mod_lpf**2
    
    demod = np.sqrt(mag)
    
    return demod

def lowpass_filter(data, cutoff, Fs):
    
    nyquist = 0.5 * Fs
    
    #LPF
    sos_lp = butter(4, cutoff/nyquist, btype='lowpass', output='sos')
    filtered = sosfiltfilt(sos_lp,data)
    
    return filtered

def plot_spectrogram(signal, Fs, window_sec, overlap):
    
    #convert window and overlap to samples
    nperseg = int(window_sec * Fs)
    noverlap = int(nperseg * overlap)
    
    f, t, Sxx = spectrogram(signal, Fs, nperseg = nperseg , noverlap = noverlap, mode = 'psd')
    
    #convert to db
    Sxx_db = 10*np.log10(Sxx)
    
    # dB range for nice plotting
    dynamic_range_db = 80
    vmax = Sxx_db.max()
    vmin = vmax - dynamic_range_db
    
    #colormesh plot
    plt.figure()
    plt.pcolormesh(f, t, Sxx_db.T, shading='auto', cmap='plasma', vmin = vmin)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Time (s)')
    plt.colorbar(label='Power (dB)')
    plt.xlim(0, 5)
    plt.show()
    
    return


# ------------------ File Path and Macros ------------------

filename = 'tfo_dataset.csv'  
Fs = 8000  # Sampling frequency in Hz
SSD = [1.5, 3, 4.5, 7, 10]  # Source-to-Detector Distances

# ------------------ Load Data ------------------

T = pd.read_csv(filename)
detector_data = T[['ch4Volts']].to_numpy().T

# ------------------ Estimate Carrier Frequencies ------------------

f1, f2 = estimate_carrier_frequencies(detector_data, Fs)
print(f"Estimated carrier frequencies: f1 = {f1:.2f} Hz, f2 = {f2:.2f} Hz")

# ------------------ Demodulate with I/Q Method ------------------

WL1_D4_demod = iq_demodulate(detector_data, f1, Fs)
WL2_D4_demod = iq_demodulate(detector_data, f2, Fs)

# # ------------------ Down-Sample by 100x ------------------

WL1_D4_down = WL1_D4_demod[::100]
WL2_D4_down = WL2_D4_demod[::100]

# # ------------------ Plot Spectrogram ------------------

plot_spectrogram(WL2_D4_down, Fs / 100, 60, 0.5)

# ------------------ Local Functions ------------------


