import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

from frame_elements import setup_indexSet, evaluate_linear_combination
from frame_operations import analysis_operator_sample, setup_gramian3, frame_algorithm

# read audio samples

amsel_samplerate, amsel_data = wavfile.read("blackbird.wav")
#print(f"Abtastrate={amsel_samplerate}.")

amsel_length = amsel_data.shape[0] / amsel_samplerate
print(f"length = {amsel_length}s")

amsel_time = np.linspace(0., amsel_length, amsel_data.shape[0])
#print(f"data[0:10]={amsel_data[0:10]}")
#print(f"Laenge(data)={amsel_data.shape}")

#plt.figure(1)
#plt.plot(amsel_time, amsel_data)
#plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
#plt.title("Amsel")
#plt.show()

I = [0, amsel_length]
m = 3
alpha = 0.5
epsilon = 0.25
jmin = -5
jmax = 5
indexSet = setup_indexSet(I, m, jmin, jmax, alpha, epsilon)


c0 = analysis_operator_sample(m, indexSet, amsel_data, amsel_time, alpha, epsilon)

gramian = setup_gramian3(m, indexSet, alpha, epsilon)
#gramian = sparse.load_npz('gramian2.npz')

relax = 0.1
iters = 10 ** 5
coeffs = frame_algorithm(relax, iters, c0, gramian)
sparse.save_npz('gramian_amsel.npz', gramian)

print('error =', np.linalg.norm(gramian.dot(coeffs)-c0))

reconstruction = evaluate_linear_combination(m, coeffs, indexSet, amsel_time, alpha, epsilon)
plt.plot(amsel_time, reconstruction, amsel_time, amsel_data)
plt.show()

L2error = np.sqrt(np.trapz(abs(reconstruction - amsel_data) ** 2, amsel_time))
print('L2-error = ', L2error)