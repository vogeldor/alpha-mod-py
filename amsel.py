import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

from frame_elements import setup_indexSet, evaluate_linear_combination, setup_indexSet_step, support
from frame_operations import analysis_operator_sample, setup_gramian3, frame_algorithm

# read audio samples

amsel_samplerate, amsel_data = wavfile.read("blackbird.wav")
amsel_length = amsel_data.shape[0] / amsel_samplerate
amsel_time = np.linspace(0., amsel_length, amsel_data.shape[0])


I = [0, amsel_length]
m = 3
alpha = 0.0
epsilon = 0.25
J = np.append(np.arange(-2250, -1750, 5), np.arange(1750, 2250, 5))
indexSet = setup_indexSet_step(I, m, J, alpha, epsilon)
#print(len(indexSet))


c0 = analysis_operator_sample(m, indexSet, amsel_data, amsel_time, alpha, epsilon)
#np.save('c0_m3_a00_blackbird.npy', c0)

gramian = setup_gramian3(m, indexSet, alpha, epsilon)
#sparse.save_npz('gramian_m3_a00_blackbird.npz', gramian)
#gramian = sparse.load_npz('gramian_blackbird.npz')

relax = 0.1
iters = 10 ** 5
coeffs = frame_algorithm(relax, iters, c0, gramian)
#np.save('coeffs_m3_j60_a00_blackbird.npy', coeffs)


print('residual error =', np.linalg.norm(gramian.dot(coeffs)-c0))

reconstruction = evaluate_linear_combination(m, coeffs, indexSet, amsel_time, alpha, epsilon)
plt.plot(amsel_time, reconstruction, amsel_time, amsel_data)
plt.show()

L2error = np.sqrt(np.trapz(abs(reconstruction - amsel_data) ** 2, amsel_time))
print('L2-error = ', L2error)