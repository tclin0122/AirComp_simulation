import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# Sample measured data
snr_db = np.array([0, -20*np.log10(0.9), -20*np.log10(0.8), -20*np.log10(0.7), -20*np.log10(0.6), -20*np.log10(0.5), -20*np.log10(0.4), -20*np.log10(0.3)])  # Example SNR values in dB
ber_measured = np.array([10**(-1.200742), 10**(-1.326684), 10**(-1.369249), 10**(-1.375042), 10**(-1.631296), 10**(-3.139604), 10**(-4.170177), 10**(-6.161712)])  # Example measured BER values

# Calculate theoretical BER for BPSK
snr_linear = 10**(snr_db / 10)  # Convert SNR from dB to linear scale
ber_theoretical = 0.5 * erfc(np.sqrt(snr_linear))  # Theoretical BER for BPSK

# Calculate theoretical BER for 3-ASK
M = 3
ber_theoretical_3ask = (2 * (M - 1) / (M * np.log2(M))) * erfc(np.sqrt((6 * np.log2(M) / (M**2 - 1)) * snr_linear) / np.sqrt(2))

# Shannon theoretical BER
shannon_snr_linear = 10**(snr_db / 10)  # Convert to linear scale
shannon_ber_theory = 0.5 * erfc(np.sqrt(2 * shannon_snr_linear))  # Shannon theory BER (approximation)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.semilogy(snr_db, ber_measured, 'o-', label='Simulate AirComp BPSK BER')
plt.semilogy(snr_db, ber_theoretical, 's--', label='BPSK Theoretical BER')
plt.semilogy(snr_db, ber_theoretical_3ask, 'd-.', label='3-ASK Theoretical BER')
#plt.axvline(x=shannon_limit_db, color='r', linestyle='--', label='Shannon Limit')
plt.semilogy(snr_db, shannon_ber_theory, 'g:', label='Shannon Theory')


plt.xlabel('SNR (dB)',fontsize=20)
plt.ylabel('BER',fontsize=20)
#plt.title('BER vs SNR for AirComp BPSK, BPSK, 3-ASK and Shannon Theory')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()
