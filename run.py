from pylsl import StreamInlet, resolve_stream
import numpy as np
from scipy import signal, butter, filtfilt  
from decode_attention import butter_highpass

def decode_eeg(data_in):

    fs = 500
    eeg_data = data_in[np.array([3, 4, 12, 13, 14, 21, 28, 29]), :]

    eeg_data = signal.resample(eeg_data, int(eeg_data.shape[1] / 10), axis=1)  # Resample along axis 1
    fs = 50

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
    for chan_num in range(numchans):
        # Previous window
        data_prevwin = EEG_prevwin[chan_num, :]
        t = np.arange(1, len(data_prevwin) + 1) / fs

        # Generate sinusoids at 4 Hz and 7 Hz
        sinusoid4Hz = np.sin(2 * np.pi * 4 * t)
        sinusoid7Hz = np.sin(2 * np.pi * 7 * t)

        # Compute the cross-correlation
        cross_corr_4Hz_prevwin = np.correlate(data_prevwin, sinusoid4Hz, mode='full')
        cross_corr_7Hz_prevwin = np.correlate(data_prevwin, sinusoid7Hz, mode='full')

        # Current window
        data = EEG[chan_num, :]
        t = np.arange(1, len(data) + 1) / fs

        # Generate sinusoids at 4 Hz and 7 Hz
        sinusoid4Hz = np.sin(2 * np.pi * 4 * t)
        sinusoid7Hz = np.sin(2 * np.pi * 7 * t)

        # Compute the cross-correlation
        cross_corr_4Hz = np.correlate(data, sinusoid4Hz, mode='full')
        cross_corr_7Hz = np.correlate(data, sinusoid7Hz, mode='full')

        if cross_corr_4Hz > cross_corr_4Hz_prevwin:
            if cross_corr_7Hz < cross_corr_7Hz_prevwin:
                points = points+1
        else:
            if cross_corr_4Hz < cross_corr_4Hz_prevwin:
                if cross_corr_7Hz > cross_corr_7Hz_prevwin:
                    points = points-1
    
    if points>0:
        output = 'left'
    elif points<0:
        output = 'right'
    else:
        output = 'stay'

    return output


def main():
    
    # first resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    
    n_channels = 30
    # initialize a buffer for 4 seconds of data
    buffer = np.zeros((n_channels, 4 * 500))  # assuming the EEG data has 30 channels

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        
        # add the sample to the buffer and remove the oldest sample
        buffer = np.roll(buffer, -1, axis=1)
        buffer[:, -1] = sample
        
        # decode every 4 seconds of data
        if timestamp % 4 == 0:
            command = decode_eeg(buffer)
            print(f"Decoded command: {command}")


      
if __name__ == "__main__":
    main()