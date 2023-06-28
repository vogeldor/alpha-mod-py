import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

from frame_elements import setup_indexSet, evaluate_linear_combination, eval_frame_element
from frame_operations import analysis_operator_sample, setup_gramian3, frame_algorithm

# read audio samples

amsel_samplerate, amsel_data = wavfile.read("blackbird.wav")
#amsel_data = amsel_data[0:1000]

amsel_length = amsel_data.shape[0] / amsel_samplerate
print(f"length = {amsel_length}s")

amsel_time = np.linspace(0., amsel_length, amsel_data.shape[0])


#plt.figure(1)
#plt.plot(amsel_time, amsel_data)
#plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
#plt.title("Amsel")
#plt.show()


I = [0, amsel_length]
m = 2
alpha = 0.0
epsilon = 0.25
J = 150
indexSet = setup_indexSet(I, m, -J, J, alpha, epsilon)

def f(t):
    return eval_frame_element(t, 10000, 0, m, alpha, epsilon)

h = 0.0001
t = np.arange(amsel_time[0], amsel_time[-1] + h, h)

plt.plot(amsel_time, amsel_data, t, f(t))
plt.show()

#amsel_data = f(amsel_time)
#c0 = analysis_operator_sample(m, indexSet, amsel_data, amsel_time, alpha, epsilon)


#gramian = setup_gramian3(m, indexSet, alpha, epsilon)
#sparse.save_npz('gramian_m1_j150_a00_blackbird.npz', gramian)
#gramian = sparse.load_npz('gramian_m1_j150_a00_blackbird.npz')


#relax = 0.005
#iters = 10 ** 2
#coeffs = frame_algorithm(relax, iters, c0, gramian)


#reconstruction = evaluate_linear_combination(m, coeffs, indexSet, amsel_time, alpha, epsilon)
#plt.plot(amsel_time, reconstruction, amsel_time, amsel_data, linestyle = 'None',)
#plt.show()
