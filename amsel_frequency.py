import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.io import wavfile
from scipy.fft import fft, fftfreq

from frame_elements import setup_indexSet_step, evaluate_linear_combination
from frame_operations import analysis_operator_sample, setup_gramian3, frame_algorithm

amsel_samplerate, amsel_data = wavfile.read("blackbird.wav")
#print(f"Abtastrate={amsel_samplerate}.")
amsel_length = amsel_data.shape[0] / amsel_samplerate
#print(f"length = {amsel_length}s")
amsel_time = np.linspace(0., amsel_length, amsel_data.shape[0])
#print(f"data[0:10]={amsel_data[0:10]}")
#print(f"Laenge(data)={amsel_data.shape}")

plt.figure(1)
plt.plot(amsel_time, amsel_data)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Amsel")
plt.show()

#plt.figure(2)
#N_amsel=amsel_data.shape[0]
#yf = fft(amsel_data)
#xf = fftfreq(N_amsel, 1 / amsel_samplerate)

#plt.plot(xf,np.abs(yf))
#plt.show()


amsel_data_section = amsel_data[int(np.floor(amsel_samplerate*2.9)):int(np.ceil(amsel_samplerate*3.2))]
amsel_time_section = amsel_time[int(np.floor(amsel_samplerate*2.9)):int(np.ceil(amsel_samplerate*3.2))]

plt.figure(1)
plt.plot(amsel_time_section, amsel_data_section)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Amsel")
plt.show()

plt.figure(2)
N_amsel=amsel_data_section.shape[0]
yf = fft(amsel_data_section)
xf = fftfreq(N_amsel, 1 / amsel_samplerate)

plt.plot(xf,np.abs(yf))
plt.show()






I = [2.9, 3.2]
m = 2
alpha = 0.0
epsilon = 0.25

J = np.append(np.arange(-3100, -1900, 1), np.arange(1900, 3100, 1))
indexSet = setup_indexSet_step(I, m, J, alpha, epsilon)



c0 = analysis_operator_sample(m, indexSet, amsel_data, amsel_time, alpha, epsilon)
np.save('c0_m2_a00_blackbird.npy', c0)

gramian = setup_gramian3(m, indexSet, alpha, epsilon)
sparse.save_npz('gramian_m2_a00_blackbird.npz', gramian)
#gramian = sparse.load_npz('gramian_blackbird.npz')

relax = 0.1
iters = 10 ** 5
coeffs = frame_algorithm(relax, iters, c0, gramian)
np.save('coeffs_m2_a00_blackbird.npy', coeffs)


print('residual error =', np.linalg.norm(gramian.dot(coeffs)-c0))

reconstruction = evaluate_linear_combination(m, coeffs, indexSet, amsel_time_section, alpha, epsilon)
plt.plot(amsel_time_section, reconstruction, amsel_time_section, amsel_data_section)
plt.show()

L2error = np.sqrt(np.trapz(abs(reconstruction - amsel_data_section) ** 2, amsel_time_section))
print('L2-error = ', L2error)
