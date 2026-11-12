## 4 Features extraction for PPG - PWA/Sys/Dias/Area
import argparse
from scipy import signal
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
import neurokit2 as nk

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# @ray.remote
def ppg_extraction(raw_ppg, hz):
    #RR Interval Extraction

    sampling = hz
    ppg = raw_ppg
    ppg_clean = nk.ppg_clean(ppg,sampling_rate=sampling)

    info = nk.ppg_findpeaks(ppg_clean,sampling_rate=sampling)
    peaks = info["PPG_Peaks"]
    rr_interval = (np.diff(peaks)/sampling)*1000

    #Signal Filtering - Find Extrema
    sos = signal.cheby2(2, 0.1, 20, 'hp', fs=hz, output='sos')
    filtered = signal.sosfilt(sos, raw_ppg)
    filtered_ma= moving_average(filtered,hz//2)

    local_minima = argrelextrema(filtered_ma, np.less)[0]
    local_maxima = argrelextrema(filtered_ma, np.greater)[0]


    rm_index=[]
    rm2_index=[]
    for k in range(0,len(np.diff(local_minima)),1):
        if(np.diff(local_minima)[k]<30):
            rm_index.append(k)
            rm2_index.append(k+1)
    try:
        sel_min = np.delete(local_minima, rm_index)
        sel_max = np.delete(local_maxima,rm2_index)
    except:
        sel_min = local_minima
        sel_max = local_maxima

    pwa_list = []
    systole_list=[]
    diastole_list=[]
    area_list =[]

    count_min =0
    for mini in sel_min:
        count_min = count_min+1
        count_max = 0
        for maxi in sel_max:
            count_max = count_max+1
            if(maxi>mini):
                pwa_list.append(filtered_ma[maxi]-filtered_ma[mini])
                systole_list.append(maxi -mini)
                if(count_min+1 <len(sel_min)):
                    if len(sel_max) > count_max:
                        diastole_list.append(sel_min[count_min+1]-sel_max[count_max])
                    else:
                        diastole_list.append(sel_min[count_min+1]-sel_max[count_max-1])
                value = 0        
                for i in range(mini,maxi,1):
                    value+=filtered_ma[i]
                area_list.append(value)
                break
                
#     if
#     if len(pwa_list < 5):
#         exit()
#     if len(systole_list <5):
#         exit()
#     if len(rr_interval < 5):
#         exit()
#     if len(area_list <5):
#         exit()
#     if (diastole_list <5):
#         exit()
    
    pwa_list = signal.resample(pwa_list, 60)
    systole_list = signal.resample(systole_list, 60)
    diastole_list = signal.resample(diastole_list, 60)
    area_list = signal.resample(area_list, 60)
    rr_list = signal.resample(rr_interval,60)
    diff_rr_list = signal.resample(np.diff(rr_interval),60)
    diff_pwa_list = signal.resample(np.diff(pwa_list),60)


#     pwa_list = signal.resample(np.difff(pwa_list), 60)
#     systole_list = signal.resample(np.diff(systole_list), 60)
#     diastole_list = signal.resample(np.diff(diastole_list), 60)
#     area_list = signal.resample(np.diff(area_list), 60)
#     rr_list = signal.resample(np.diff(rr_interval),60)
    
    
#     print(rr_list.shape,area_list.shape,diastole_list.shape,systole_list.shape,pwa_list.shape)
    return rr_list,area_list,diastole_list,systole_list,pwa_list, diff_pwa_list, diff_rr_list

from pathlib import Path
import glob
# from rich.console import Console
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from rhodes.core.utils.signals.filters import filter_chebychevII_bandpass, hampel_filter
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Union, Literal
from numba import jit

from scipy.signal import filtfilt
from scipy import signal

def filter_signal(ppg, fs, fL=0, fH=0, order = 4, sm_wins_ppg = 30):

    if fL==0:
        b,a = signal.cheby2(order, 20, [fH], 'low', fs=fs)
    else:
        b, a = signal.cheby2(order, 20, [fL,fH], 'bandpass', fs=fs)

    ppg_cb2 = filtfilt(b, a, ppg)
    win = round(fs * sm_wins_ppg/1000)
    B = 1 / win * np.ones(win)
    ppg = filtfilt(B, 1, ppg_cb2)

    if fs >= 75:
        win = round(fs * sm_wins_ppg/1000)
        B = 1 / win * np.ones(win)
        ppg = filtfilt(B, 1, ppg_cb2)
    else:
        ppg=ppg_cb2
    

    return ppg
import numpy as np
def slpsamp(t, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow):
    while t > tt_2:
        prevVal = 0
        if (tt_2 >= 0) and (tt_2-1 >= 0) and (tt_2 < len(data)) and (tt_2-1 < len(data)):
            val2 = data[tt_2 - 1]
            val1 = data[tt_2]
        else:
            val2 = prevVal
            val1 = val2
        prevVal = val2
        dy = val1-val2
        dy = 0 if dy < 0 else dy
        tt_2 += 1
        M = round((tt_2 % (BUFLN-1)) + 1)
        et = dy
        ebuf[M] = et
        aet = 0
        for i in range(SLPwindow):
            p = M - i
            p = p + (BUFLN-1) if p <= 0 else p
            aet += ebuf[p]
        lbuf[M] = aet
    M3 = round((t % (BUFLN-1)) + 1)
    beat1 = lbuf[M3]
    return beat1, ebuf, lbuf, tt_2, aet

def fastqppg(data: np.double, fs: int):
    """
    Find PPG onset position from the input ppg data.

    Parameters
    ----------
    data: np.double
        PPG signal in numpy double precision.
    fs: int

    Returns
    -------
    onset_pos: np.ndarray
        Pulse onset positions on `data`.
    """

    ppgOnsets= list()
    beat_n = 0

    sps = fs  # Sampling Frequency

    BUFLN = 4096           # /* must be a power of 2, see slpsamp() */
    # /* eye-closing period is set to 0.34 sec (340 ms) for PPG */
    EYE_CLS = 0.34
    LPERIOD = sps*8       # /* learning period is the first LPERIOD samples */
    SLPW = 0.17            # /* Slope width (170ms) for PPG */
    NDP = 2.5             # /* adjust threshold if no pulse found in NDP seconds */
    TmDEF = 5              # /* minimum threshold value (default) */
    Tm = TmDEF

    # Re-scale data to ~ +/- 2000 range if data points is less than 5 mins
    if len(data) < 5*60*sps:
        data = (data-data.min())/(data.max()-data.min())*4000-2000
    else:
        # Find max/min every 5 minute, rescaling the data base on the median of obtained maxs/mins.
        max_data, min_data = list(), list()
        for i in range(0, len(data), 5*60*sps):
            max_data.append(data[i:min(i+5*60*sps-1, len(data))].max())
            min_data.append(data[i:min(i+5*60*sps-1, len(data))].min())
        max_data, min_data = np.array(max_data), np.array(min_data)
        data = (data-np.median(min_data)) / \
            (np.median(max_data)-np.median(min_data))*4000-2000

    EyeClosing = round(sps * EYE_CLS)   # /* set eye-closing period */
    ExpectPeriod = round(sps * NDP)     # /* maximum expected RR interval */
    SLPwindow = round(sps * SLPW)       # /* slope window size */
    timer = 0

    ebuf = np.zeros(BUFLN)
    lbuf = np.zeros(BUFLN)
    tt_2 = 0
    aet = 0

    t1 = 8*sps
    T0 = 0
    n = 0
    for t in range(t1):
        temp, ebuf, lbuf, tt_2, aet = slpsamp(
            t, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)
        T0 = T0+temp
        n += 1
    T0 = T0/n  # T0=T0/t1
    Ta = 3 * T0

    learning = True
    t = 0
    while t < len(data):
        if (learning):
            if (t > LPERIOD):
                learning = False
                T1 = T0
                t = 0  # /* start over */
            else:
                T1 = 2*T0

        temp, ebuf, lbuf, tt_2, aet = slpsamp(
            t, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)

        if (temp > T1):  # /* found a possible ABP pulse near t */
            timer = 0
            # /* used for counting the time after previous ABP pulse */
            maxd = temp
            mind = maxd
            tmax = t
            for tt in range(t, int(t + EyeClosing - 1)):
                temp2, ebuf, lbuf, tt_2, aet = slpsamp(
                    tt, data, BUFLN,  ebuf, lbuf, tt_2, aet, SLPwindow)
                if temp2 > maxd:
                    maxd = temp2
                    tmax = tt
            if maxd == temp:
                t += 1
                continue
            for tt in range(tmax, int(t - EyeClosing / 2 + 1), -1):
                temp2, ebuf, lbuf, tt_2, aet = slpsamp(
                    tt, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)
                if temp2 < mind:
                    mind = temp2
            if maxd > mind+10:
                onset = (maxd-mind)/100+2
                tpq = t - round(0.04*fs)
                maxmin_2_3_threshold = (maxd-mind)*2.0/3
                for tt in range(tmax, int(t - EyeClosing / 2 + 1), -1):
                    temp2, ebuf, lbuf, tt_2, aet = slpsamp(
                        tt, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)
                    if temp2 < maxmin_2_3_threshold:
                        break
                for tt in range(tt, int(t - EyeClosing / 2 + round(0.024*fs)), -1):
                    temp2, ebuf, lbuf, tt_2, aet = slpsamp(
                        tt, data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)
                    temp3, ebuf, lbuf, tt_2, aet = slpsamp(
                        tt-round(0.024*fs), data, BUFLN, ebuf, lbuf, tt_2, aet, SLPwindow)
                    if temp2-temp3 < onset:
                        tpq = tt-round(0.016*fs)
                        break

                # % find valley from the original data around 0.25s of tpq
                valley_v = round(tpq)
                for valley_i in range(round(max(2, tpq-round(0.20*fs))), round(min(tpq+round(0.05*fs), len(data)-1))):
                    # % If vally is too low, it cannot serve as an index, so move to the next time.
                    if valley_v <= 0:
                        t += 1
                        continue

                    if (data[valley_v] > data[valley_i]) and (data[valley_i] <= data[valley_i-1]) and (data[valley_i] <= data[valley_i+1]):
                        valley_v = valley_i

                if not learning:
                    # If we are looking for the first peak
                    if beat_n == 0:
                        # If the proposed peak index > 0
                        if round(valley_v) > 0:
                            ppgOnsets.append(round(valley_v))
                            beat_n += 1
                    else:
                        # Check if rounded valley_v is greater than the prior beat index
                        if round(valley_v) > ppgOnsets[beat_n-1]:
                            ppgOnsets.append(round(valley_v))
                            beat_n += 1

                # Adjust thresholds
                Ta = Ta + (maxd - Ta)/10
                T1 = Ta / 3

                # /* Lock out further detections during the eye-closing period */
                t = tpq+EyeClosing
        else:
            if not learning:
                # Once past the learning period, decrease threshold if no pulse
                # was detected recently.
                timer += 1
                if (timer > ExpectPeriod) and (Ta > Tm):
                    Ta = Ta - 1
                    T1 = Ta / 3
        t += 1
    return ppgOnsets

def pulse_peaks_from_onsets(onsets, sig):
    if len(onsets) == 0:
        peaks = []
    else:
        peaks = np.full((len(onsets)-1,1), np.nan)
        for wave_no in range(len(onsets)-1):
            temp, idx = max(enumerate(sig[onsets[wave_no]:onsets[wave_no+1]]), key=lambda x: x[1])
            peaks[wave_no] = temp + onsets[wave_no] - 1
    return peaks

# def pulse_peaks_from_onsets(onsets, sig):
#     try:
#         peaks = np.full((len(onsets)-1, 1), np.nan)
#         for wave_no in range(len(onsets)-1):
#             temp, idx = max(enumerate(sig[onsets[wave_no]:onsets[wave_no+1]]), key=lambda x: x[1])
#             peaks[wave_no] = temp
#     except ValueError:
#         peaks = np.full((len(onsets)-1, 1), np.nan)
#     return peaks

def tidy_peaks_and_onsets(peaks, onsets):
    local_min = None
    finished = None
    els_to_remove = None
    peak_no = None
    rel_els = None
    peak_vals = None
    el_to_remove = None
    # (i) No two points at the same time
    # remove any repeated peaks (or onsets)
    peaks = np.unique(peaks)
    onsets = np.unique(onsets)

    # If there is a peak and onset at the same index, then remove them both
    repeated_vals = np.intersect1d(peaks, onsets)
    peaks = np.setxor1d(peaks, repeated_vals)
    onsets = np.setxor1d(onsets, repeated_vals)
    del repeated_vals

    return peaks, onsets
def eliminate_wrongpeaks(peaks, sig):
    newpeaks = []
    for i in range(len(peaks)):
        if i < len(peaks)-1:
            x = sig[peaks.astype(int)][i+1] - sig[peaks.astype(int)][i]
            if x < 0:
                newpeaks.append(peaks[i])
    newpeaks = np.array(newpeaks)
    return newpeaks
def eliminate_wrongonsets(onsets, t):
    newonsets = []
    for i in range(len(onsets)):
        if i < len(onsets)-1:
            x_t = t[onsets.astype(int)][i+1] - t[onsets.astype(int)][i]
            if x_t > 0.5:
                newonsets.append(onsets[i+1])
    newonsets = np.array(newonsets)
    return newonsets

# Find onsets and peaks of signal
def detect_onsets_peaks(signal, fs):
    filt = filter_signal(signal, fs, fL=0.5, fH=5)
    t = np.linspace(0,len(filt)/fs,len(filt))
    onsets = fastqppg(filt, fs)
    peaks = pulse_peaks_from_onsets(onsets,filt)
    peaks, onsets = tidy_peaks_and_onsets(peaks, onsets)
    onsets = eliminate_wrongonsets(onsets, t)
    return filt, t, onsets, peaks

from scipy.signal import firls 
import numpy as np
def get_dicrotic_notch (ppg, fs, peaks: np.array, onsets: list):
        """
        Dicrotic Notch function estimate the location of dicrotic notch in between the systolic and diastolic peak

        :param peaks: peaks of the signal
        :type peaks: 1-d array
        :param onsets: onsets of the signal
        :type onsets: list

        :return: location of dicrotic notches, 1-d array
        """

        ## The 2nd derivative and Hamming low pass filter is calculated.
        dxx = np.diff(np.diff(ppg))
        fs = fs

        # Make filter
        Fn = fs / 2                                 # Nyquist Frequency
        FcU = 20                                    # Cut off Frequency: 20 Hz
        FcD = FcU + 5                               # Transition Frequency: 5 Hz

        n = 21                                      # Filter order
        f = [0, (FcU / Fn), (FcD / Fn), 1]          # Frequency band edges
        a = [1, 1, 0, 0]                            # Amplitudes
        b = firls(n, f, a)

        lp_ppg = filtfilt(b, 1,  dxx)          # Low pass filtered signal with 20 cut off Frequency and 5 Hz Transition width

        ## The weighting is calculated and applied to each beat individually
        def t_wmax(i, peaks,onsets):
            if i < 3:
                HR = np.mean(np.diff(peaks))/fs
                t_wmax = -0.1 * HR + 0.45
            else:
                t_wmax = np.mean(peaks[i - 3:i]-onsets[i - 3:i])/fs
            return t_wmax

        dic_not=[]
        for i in range(0,len(onsets)-1):
            nth_beat = lp_ppg[onsets[i].astype(int):onsets[i + 1].astype(int)]

            i_Pmax=peaks[i].astype(int)-onsets[i].astype(int)
            t_Pmax=(peaks[i].astype(int)-onsets[i].astype(int))/fs
            t=np.linspace(0,len(nth_beat)-1,len(nth_beat))/fs
            T_beat=(len(nth_beat)-1)/fs
            tau=(t-t_Pmax)/(T_beat-t_Pmax)
            tau[0:i_Pmax] = 0
            beta=5

            if len(peaks)>1:
                t_w=t_wmax(i, peaks, onsets)
            else:
                t_w=np.NaN

            if t_w!=T_beat:
                tau_wmax=(t_w-t_Pmax)/(T_beat-t_Pmax)
            else:
                tau_wmax=0.9

            alfa=(beta*tau_wmax-2*tau_wmax+1)/(1-tau_wmax)
            if (alfa > 4.5) or (alfa < 1.5):
                HR = np.mean(np.diff(peaks))/fs
                t_w = -0.1 * HR + 0.45
                tau_wmax = (t_w - t_Pmax) / (T_beat - t_Pmax)
                alfa = (beta * tau_wmax - 2 * tau_wmax + 1) / (1 - tau_wmax)

            ## Calculate the Dicrotic Notch for each heart cycle using the weighted window
            if alfa>1:
                w = tau ** (alfa - 1) * (1 - tau) ** (beta - 1)
            else:
                w = tau * (1 - tau) ** (beta - 1)

            pp=w*nth_beat
            pp = pp[np.where(~np.isnan(pp))]
            max_pp_v = np.max(pp)
            max_pp_i=np.where(pp==max_pp_v)[0][0]
            ## NOTE!! Shifting with 26 ms. FIX IT!
            shift=int(fs*0.026)
            dic_not.append(max_pp_i+onsets[i]+shift)
        dic_not = np.array(dic_not)
        dp = [0] * len(dic_not)
        for i in range(len(dic_not)):
            dp[i] = dic_not[i] + 0.2
        dp = np.array(dp)
        return dic_not,dp


# %%
# PD, PRT, DT = cal_fiducial_point(peaks,t,dp)
def cal_fiducial_point(peaks, onsets,t,dp):
    PD = []
    for i in range(len(peaks)-1):
        PD.append(t[peaks[i+1].astype(int)]-t[peaks[i].astype(int)])
    PRT = []
    for i in range(len(onsets)):
        PRT.append(t[peaks[i].astype(int)]-t[onsets[i].astype(int)])
    DT = []
    for i in range(len(dp)):
        DT.append(t[dp[i].astype(int)]-t[peaks[i].astype(int)])
    return PD, PRT, DT

# %% [markdown]
# ### Calculation of amplitude-based features by using z-score normalization to each ppg signal

# %%
def z_score_norm(ppg):
    ppg_new = np.array([])
    for i in range(len(ppg)):
        ppg_new = np.append(ppg_new, (ppg[i] - np.mean(ppg)) / np.std(ppg))
    return ppg_new

def cal_amp_z_score_norm(ppg, fs):
    y = ppg
    y = filter_signal(y, fs, fL=0.05, fH=20)
    y_new = z_score_norm(y)
    t = np.linspace(0, len(y_new)/fs, len(y_new))
    onsets = fastqppg(y_new, fs)
    peaks = pulse_peaks_from_onsets(onsets,y_new)
    peaks, onsets = tidy_peaks_and_onsets(peaks,onsets)
    # peaks = eliminate_wrongpeaks(peaks, y)
    onsets = eliminate_wrongonsets(onsets, t)
    dc, dp = get_dicrotic_notch(y_new, fs, peaks, onsets)
    # Amplitude of the systolic peak
    Sys_Amp = []
    x = min(len(onsets),len(peaks))
    for i in range(len(onsets)):
        Sys_Amp.append(y_new[peaks[i].astype(int)]-y_new[onsets[i].astype(int)])
    # Amplitude of the diastolic peak
    Dia_Amp = []
    y = min(len(onsets),len(dp))
    for i in range(y):
        Dia_Amp.append(y_new[dp[i].astype(int)]-y_new[onsets[i].astype(int)])
    return Sys_Amp, Dia_Amp

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def duration_normalization(ppg_signal, num_intervals=100, fs=256):
    # Find peaks of the PPG signal
    # Assuming you have functions 'filter_signal', 'fastqppg', 'pulse_peaks_from_onsets',
    # 'tidy_peaks_and_onsets', 'eliminate_wrongonsets' defined elsewhere
    y = filter_signal(ppg_signal, fs, fL=0.05, fH=20)
    y_new = z_score_norm(y)
    t = np.linspace(0, len(y_new)/fs, len(y_new))
    onsets = fastqppg(y_new, fs)
    peaks = pulse_peaks_from_onsets(onsets, y_new)
    peaks, onsets = tidy_peaks_and_onsets(peaks, onsets)
    # peaks = eliminate_wrongpeaks(peaks, y)
    onsets = eliminate_wrongonsets(onsets, t)

    # Create a new array to store normalized pulses and peaks
    normalized_pulses = []

    # Interpolate each pulse to have the same number of intervals
    for i in range(len(onsets) - 1):
        pulse_start = onsets[i].astype(int)
        pulse_end = onsets[i + 1].astype(int)
        pulse = y_new[pulse_start:pulse_end]

        # Interpolate the pulse using cubic spline
        x_old = np.linspace(0, 1, len(pulse))
        x_new = np.linspace(0, 1, num_intervals)
        cs = CubicSpline(x_old, pulse)
        interpolated_pulse = cs(x_new)

        # Append the interpolated pulse to the list
        normalized_pulses.append(interpolated_pulse)
        

    return np.array(normalized_pulses)

# %%
import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian_to_pulse(pulse, x):
    # Define initial guess for Gaussian parameters
    initial_guess = [np.max(pulse), np.argmax(pulse), 1]  # Amplitude, mean, standard deviation

    # Perform curve fitting to fit Gaussian to pulse
    popt, _ = curve_fit(gaussian, x, pulse, p0=initial_guess)

    return popt  # Return the optimized parameters for the Gaussian

def compute_residual_curve(pulse, x):
    # Fit Gaussian to systolic arm of the pulse
    gaussian_params = fit_gaussian_to_pulse(pulse, x)

    # Compute Gaussian curve
    gaussian_curve = gaussian(x, *gaussian_params)

    # Subtract Gaussian from the main pulse to obtain residual curve
    residual_curve = pulse - gaussian_curve

    return residual_curve

# %%
import numpy as np
from scipy.signal import find_peaks

def find_dicrotic_notch(pulse):
    # Fit a Gaussian to the systolic arm of the pulse (not implemented here)
    # Subtract the Gaussian from the main pulse to get the residual curve (not implemented here)
    x = np.arange(len(pulse))  # Assuming x-axis is just the index of the sample
    residual_curve = compute_residual_curve(pulse, x)

    # Calculate the first derivative of the residual curve
    first_derivative_residual = np.gradient(residual_curve)

    # Calculate the first derivative of the original pulse
    first_derivative_pulse = np.gradient(pulse)

    # Find peaks in the first derivative of the residual curve
    peaks_residual, _ = find_peaks(first_derivative_residual)

    # Find peaks in the first derivative of the original pulse
    peaks_pulse, _ = find_peaks(first_derivative_pulse)
    # peaks_pulse = np.argmax(first_derivative_pulse)
    
    # Create a window between the second peak of the first derivative of the residual curve
    # and the third peak of the first derivative of the original pulse
    window_start = peaks_residual[1]
    window_end = peaks_pulse[2]

    # Calculate the second derivative of the original pulse within the window
    second_derivative_pulse = np.gradient(first_derivative_pulse[window_start:window_end])

    # Find the largest peak in the second derivative of the original pulse within the window
    dicrotic_notch_index = window_start + np.argmax(second_derivative_pulse)

    return dicrotic_notch_index


# %%
from scipy.signal import find_peaks
import numpy as np

def find_diastolic_peak(pulse, dicrotic_notch_index, descending_wave_fraction=0.2):
    descending_wave_index = dicrotic_notch_index + int(descending_wave_fraction * len(pulse))

    # Define the window for the diastolic peak search
    window_start = dicrotic_notch_index
    window_end = descending_wave_index
    # x = pulse[window_start:window_end]
    peaks, _ = find_peaks(pulse[window_start:window_end])
    dp = peaks + dicrotic_notch_index

    return dp

def sleep_feature(ppg, fs):
    ppg = ppg * -1
    ppg_filtered = filter_signal(ppg, fs, fL=0.05, fH=20)
    t = np.linspace(0, len(ppg_filtered)/fs, len(ppg_filtered))
    onsets = fastqppg(ppg_filtered, fs)
    peaks = pulse_peaks_from_onsets(onsets, ppg_filtered)
    peaks, onsets = tidy_peaks_and_onsets(peaks, onsets)
    onsets = eliminate_wrongonsets(onsets, t)
    dc, dp = get_dicrotic_notch(ppg_filtered, fs, peaks, onsets)
    PD, PRT, DT = cal_fiducial_point(peaks, onsets, t, dp)
    Sys_Amp, Dia_Amp = cal_amp_z_score_norm(ppg_filtered, fs)
    # PD = np.array(PD)
    # PRT = np.array(PRT)
    # DT = np.array(DT)
    # Sys_Amp = np.array(Sys_Amp)
    # Dia_Amp = np.array(Dia_Amp)
    lengths = [len(PD), len(PRT), len(DT), len(Sys_Amp), len(Dia_Amp)]
    minlen = np.min(lengths)
    if minlen > 0:
        dp = signal.resample(dp, minlen)
        dc = signal.resample(dc, minlen)
        PD = signal.resample(PD, minlen)
        PRT = signal.resample(PRT, minlen)
        DT = signal.resample(DT, minlen)
        Sys_Amp = signal.resample(Sys_Amp, minlen)
        Dia_Amp = signal.resample(Dia_Amp, minlen)
        subject = {}
        subject['sys_onset'] = onsets
        subject['sys_peak'] = peaks
        subject['dia_peak'] = dp
        subject['dicrotic_notch'] = dc
        subject['PD'] = PD
        subject['PRT'] = PRT
        subject['DT'] = DT
        subject['Sys_Amp'] = Sys_Amp
        subject['Dia_Amp'] = Dia_Amp
        # normalized_pulses = duration_normalization(ppg_filtered)
        # deltaT_norm_l = []
        # PRT_norm_l = []
        # Rslope_l = []
        # RI_l = []
        # for j in range(len(normalized_pulses)):
        #     # plt.plot(normalized_pulses[j])
        #     # plt.show()
        #     try:
        #         ppg_n = normalized_pulses[j]
        #         t = np.linspace(0, len(ppg_n), len(ppg_n))
        #         pulse = ppg_n
        #         dicrotic_notch_index = find_dicrotic_notch(pulse)
        #         diastolic_peak_index = find_diastolic_peak(pulse, dicrotic_notch_index)
        #         sys_peak_time = t[np.argmax(pulse)]
        #         dia_peak_time = t[diastolic_peak_index.item()]
        #         deltaT_norm = np.abs(sys_peak_time - dia_peak_time)
        #         onsets_time = t[0]
        #         deltaT_norm_l.append(deltaT_norm)
        #         PRT_norm = np.abs(onsets_time - sys_peak_time)
        #         PRT_norm_l.append(PRT_norm)
        #         Rslope = Sys_Amp[j] / PRT_norm
        #         RI = Dia_Amp[j] / Sys_Amp[j]
        #         Rslope_l.append(Rslope)
        #         RI_l.append(RI)
        #     except Exception as e:
        #         pass
        # deltaT_norm_l = np.array(deltaT_norm_l)
        # PRT_norm_l = np.array(PRT_norm_l)
        # Rslope_l = np.array(Rslope_l)
        # RI_l = np.array(RI_l)
        # subject['deltaT_norm'] = deltaT_norm_l
        # subject['PRT_norm'] = PRT_norm_l
        # subject['Rslope'] = Rslope_l
        # subject['RI'] = RI_l
        for i in subject:
            subject[i] = signal.resample(subject[i], 120)
        return subject
    else:
        return


# CUSleep
import os
import pandas as pd
import pickle
import numpy as np
# Apsense
def extract_features(meta_file, data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fs_ppg = 256
    metadata = pd.read_csv(f'{meta_file}')
    skip_subjects = [2934,2416]

    for i in range(len(metadata)):
        if metadata['overall5'][i] == 7.0 and metadata['mesaid'][i] not in skip_subjects:
            id_o = metadata['mesaid'][i]
            print('ID:', id_o)
            path_file = f'{data_dir}/s_{id_o}.pkl'
            with open(path_file, 'rb') as f:
                data = pickle.load(f)    
            subject = []
            window = []
            rr = []
            area = []
            dias = []
            syst = []
            pwa = []
            diff_pwa = []
            diff_rr = []
            label = []
            for i in range(len(data)):
                # if data['Event'][i] == 'Normal':
                print(id_o, i)
                try:
                    ppg = data['ppg'][i]
                    rr_list,area_list,diastole_list,systole_list,pwa_list, diff_pwa_list, diff_rr_list = ppg_extraction(ppg, hz=fs_ppg)
                    sub = None
                    if sub == None:
                        subject.append(id_o)
                        window.append(i)
                        rr.append(rr_list)
                        area.append(area_list)
                        dias.append(diastole_list)
                        syst.append(systole_list)
                        pwa.append(pwa_list)
                        diff_pwa.append(diff_pwa_list)
                        diff_rr.append(diff_rr_list)
                        label.append(data['label'][i])
                except Exception as e:
                    print(e)
                    pass
            df = pd.DataFrame({
                'subject': subject,
                'window ': window,
                'rr': rr,
                'area': area,
                'diastole': dias,
                'systole': syst,
                'pwa': pwa,
                'diff_pwa': diff_pwa,
                'diff_rr': diff_rr,
                'label': label,
            })
            path = f'{output_dir}/s_{id_o}.pkl'
            with open(path, 'wb') as f:
                pickle.dump(df, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Extract features from PPG data.')
    p.add_argument('--meta', type=str, default='metadata.csv', help='Path to the metadata CSV file.')
    p.add_argument('--data_dir', type=str, default='data', help='Directory containing the PPG data files.')
    p.add_argument('--output_dir', type=str, default='output', help='Directory to save the extracted features.')
    args = p.parse_args()
    extract_features(args.meta, args.data_dir, args.output_dir)
    
# python preprocess.py \
#     --meta /home/Data/MESA/datasets/mesa-sleep-dataset-0.5.0.csv \
#     --data_dir /home/work/Apnea/ApSense/benchmark/preprocess/MESA/segments \
#     --output_dir /home/work/Apnea/ApSense/benchmark/preprocess/MESA/feature