import numpy as np
from scipy import signal, butter, filtfilt

data_in = np.zeros((37, 2000))
fs = 500

eeg_data = data_in[np.array([3, 4, 12, 13, 14, 21, 28, 29]), :]

eeg_data = signal.resample(eeg_data, int(eeg_data.shape[1] / 10), axis=1)  # Resample along axis 1


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


b, a = butter_highpass(1, fs, order=2)
eeg_data = filtfilt(b, a, eeg_data, axis=1)

EEG_prevwin = eeg_data[:, 0:99]
EEG = eeg_data[:, 100:199]

numchans = EEG.shape[0]
points = 0
for channum in range(numchans):
    Y = np.fft.fft(EEG_prevwin[channum])
    L = Y.shape[0]
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2]
    P1[1:-2] = 2 * P1[1:-2]
    f = fs * (np.arange(0, (L / 2) + 1) / L)
    selected_indices = np.logical_and(f > 3.9, f < 4.1)
    selected_values = P1[selected_indices]
    pow4Hz_prevwin = np.mean(selected_values)
    selected_indices = np.logical_and(f > 3, f < 5)
    selected_values = P1[selected_indices]
    relpow4Hz_prevwin = pow4Hz_prevwin - np.mean(selected_values)
    selected_indices = np.logical_and(f > 6.9, f < 7.1)
    selected_values = P1[selected_indices]
    pow7Hz_prevwin = np.mean(selected_values)
    selected_indices = np.logical_and(f > 6, f < 8)
    selected_values = P1[selected_indices]
    relpow7Hz_prevwin = pow7Hz_prevwin - np.mean(selected_values)

    Y = np.fft.fft(EEG[channum])
    L = Y.shape[0]
    P2 = np.abs(Y / L)
    P1 = P2[:L // 2]
    P1[1:-2] = 2 * P1[1:-2]
    f = fs * (np.arange(0, (L / 2) + 1) / L)
    selected_indices = np.logical_and(f > 3.9, f < 4.1)
    selected_values = P1[selected_indices]
    pow4Hz = np.mean(selected_values)
    selected_indices = np.logical_and(f > 3, f < 5)
    selected_values = P1[selected_indices]
    relpow4Hz = pow4Hz - np.mean(selected_values)
    selected_indices = np.logical_and(f > 6.9, f < 7.1)
    selected_values = P1[selected_indices]
    pow7Hz = np.mean(selected_values)
    selected_indices = np.logical_and(f > 6, f < 8)
    selected_values = P1[selected_indices]
    relpow7Hz = pow7Hz - np.mean(selected_values)

    if relpow4Hz > relpow4Hz_prevwin:
        if relpow7Hz < relpow7Hz_prevwin:
            points = points+1
    else:
        if relpow4Hz < relpow4Hz_prevwin:
            if relpow7Hz > relpow7Hz_prevwin:
                points = points-1
    
if points>0:
    output = 'left'
elif points<0:
    output = 'right'
else:
    output = 'stay'
