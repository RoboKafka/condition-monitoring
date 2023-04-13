import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
class VibrationAnalyzer:
    def __init__(self, filename, axis='x', callback=None):
        self.data = pd.read_csv(filename)
        self.data.columns = ['timestamp','x','y','z']
        self.sampling_freq = 20000
        self.axis = axis.lower()
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError("Invalid axis specified. Please specify either 'x', 'y', or 'z'.")
        # Define the filter parameters
        order = 3  # filter order
        cutoff_freq = 7000  # cutoff frequency in Hz
        nyquist_freq = 0.5 * self.sampling_freq  # Nyquist frequency
        normalized_cutoff_freq = cutoff_freq / nyquist_freq  # normalized cutoff frequency
        self.b, self.a = butter(order, normalized_cutoff_freq, btype='lowpass')  # filter coefficients
        self.data_filtered = pd.DataFrame(columns=['timestamp','x','y','z'])
        self.callback = callback
    
    def preprocess(self):
        self.data_filtered['timestamp'] = self.data['timestamp']
        self.data_filtered[self.axis] = filtfilt(self.b, self.a, self.data[self.axis], axis=0)
        self.amplitude = np.sqrt(np.sum(np.square(self.data_filtered.iloc[:, 1:]), axis=1))
        self.envelope = np.abs(filtfilt(self.b, self.a, self.amplitude, axis=0))
        self.rms = np.sqrt(np.mean(np.square(self.amplitude)))
        self.fft = np.fft.fft(self.data_filtered[self.axis])
        self.freq = np.fft.fftfreq(len(self.data_filtered), 1/self.sampling_freq)
        self.psd = np.abs(self.fft)**2 / len(self.data_filtered)
    
    def plot_time_domain(self):
        plt.plot(self.data_filtered['timestamp'], self.data_filtered[self.axis])
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (m/s^2)")
        plt.title("Vibration Envelope")
        if self.callback is None:
            plt.show()
        else:
            self.callback(plt)
    
    def plot_freq_domain(self):
        plt.semilogx(self.freq, 10*np.log10(self.psd))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (dB)")
        plt.title("Vibration Spectrum")
        if self.callback is None:
            plt.show()
        else:
            self.callback(plt)
