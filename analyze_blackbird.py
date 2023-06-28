import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy import io

from frame_elements import setup_indexSet_step, evaluate_linear_combination, setup_indexSet
from frame_operations import analysis_operator_sample, setup_gramian3, frame_algorithm

amsel_samplerate, amsel_data = wavfile.read("blackbird.wav")
amsel_length = amsel_data.shape[0] / amsel_samplerate
amsel_time = np.linspace(0., amsel_length, amsel_data.shape[0])
#plt.figure(1)
#plt.plot(amsel_time, amsel_data)
#plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
#plt.title("Amsel")
#plt.show()

I = [0, amsel_length]
m = 3
alpha = 0.25
epsilon = 0.25
J = 8000
indexSet = setup_indexSet(I, m, -J, J, alpha, epsilon)
print(len(indexSet))

c0 = analysis_operator_sample(m, indexSet, amsel_data, amsel_time, alpha, epsilon)
#io.savemat('coeffs_m3_a00_J8000_blackbird.mat', {"data": c0})