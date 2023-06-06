from scipy import io
import numpy as np

coeffs = np.array(np.load('coeffs_m3_a00_e25_j60_sinparab5.npy'))
io.savemat('coeffs_m3_a00_e25_j60_sinparab5.mat', {"data": coeffs})

coeffs = np.array(np.load('coeffs_m3_a05_e25_j60_sinparab5.npy'))
io.savemat('coeffs_m3_a05_e25_j60_sinparab5.mat', {"data": coeffs})