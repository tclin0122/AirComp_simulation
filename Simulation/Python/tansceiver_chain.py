
#Eb/N0 Vs BER for BPSK over AWGN (complex baseband model)
# © Author: Mathuranathan Viswanathan (gaussianwaves.com)
import numpy as np #for numerical computing
import matplotlib.pyplot as plt #for plotting functions
from scipy.special import erfc #erfc/Q function

#---------Input Fields------------------------
nSym = 10**5 # Number of symbols to transmit
EbN0dBs = np.arange(start=-4,stop = 13, step = 1) # Eb/N0 range in dB for simulation
BER_sim = np.zeros(len(EbN0dBs)) # simulated Bit error rates

M=2 #Number of points in BPSK constellation
m = np.arange(0,M) #all possible input symbols
constellation = 2 * m - 1  #reference constellation for BPSK

#------------ Transmitter---------------
inputSyms_1 = np.random.randint(low=0, high = M, size=nSym) #Random 1's and 0's as input to BPSK modulator
inputSyms_2 = np.random.randint(low=0, high = M, size=nSym) #Random 1's and 0's as input to BPSK modulator
inputSyms = inputSyms_1 + inputSyms_2
s_1 = constellation[inputSyms_1] #modulated symbols
s_2 = constellation[inputSyms_2]
s = s_1 + s_2
fig, ax1 = plt.subplots(nrows=1,ncols = 1)
ax1.plot(np.real(s),np.imag(s),'*')

#----------- Channel --------------
#Compute power in modulatedSyms and add AWGN noise for given SNRs
for j, EbN0dB in enumerate(EbN0dBs):
    # Convert Eb/N0 from dB to linear scale
    EbN0 = 10**(EbN0dB / 10)

    # Energy per bit is 1 for BPSK
    Eb = 1
    N0 = Eb / EbN0  # Noise spectral density

    # Generate AWGN noise
    noise = np.sqrt(N0 / 2) * np.random.randn(nSym)  # Gaussian noise with variance N0/2

    # Received signal (transmitted signal + noise)
    r = s + noise

    #-------------- Receiver ------------
    #detectedSyms = (r <= 0).astype(int) #thresolding at value 0
    detectedSyms = np.zeros_like(r, dtype=int)
    detectedSyms[r < -1] = -2  # map -2 to 0
    detectedSyms[(r >= -1) & (r < 1)] = 0  # map 0 to 1
    detectedSyms[r >= 1] = 2  # map 2 to 2
    BER_sim[j] = np.sum(detectedSyms != s)/nSym #calculate BER


BER_theory = 0.5*erfc(np.sqrt(10**(EbN0dBs/10)))

fig, ax = plt.subplots(nrows=1,ncols = 1)
ax.semilogy(EbN0dBs,BER_sim,color='r',marker='o',linestyle='',label='AirComp BPSK')
ax.semilogy(EbN0dBs,BER_theory,marker='',linestyle='-',label='BPSK Theory')
ax.set_xlabel('$E_b/N_0(dB)$');ax.set_ylabel('BER ($P_b$)')
ax.set_title('Probability of Bit Error for BPSK over AWGN channel')
ax.set_xlim(-5,13)
ax.grid(True)
ax.legend()
plt.show()
