import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift, fft


R = 20e6  # Bit rate
sps = 8  # Samples per symbol
num_symbols = 20
beta = 0.25
Ts = sps
num_taps = 101

# 2. Generate a binary stream for 20 symbols for 4-PAM modulation
# Generate a binary bitstream of length 40
bitstream = np.random.randint(2, size=40)

# Define the look up table(mapping dictionary)
look_up_table = {'00': -3, '01': -1, '10': 1, '11': 3}

# Convert bitstream to string and map two successive bits using the look up table
bitstream_str = ''.join(map(str, bitstream))
symbols = [look_up_table[bitstream_str[i:i+2]] for i in range(0, len(bitstream_str), 2)]

#  Generate square pulses
square_pulses = np.repeat(symbols, sps)
plt.figure(0)
plt.plot(square_pulses)
plt.title(' randomly generated square pulses')
plt.grid(True)

# 5. Generate samples  to Plot the transmit waveform for square pulse
samples = np.zeros(sps * len(symbols))
samples[::sps] = symbols
plt.figure(1)
plt.plot(samples)
plt.title('transmit waveform for square pulse')
plt.grid(True)


# 6.root raised-cosine filter for plotting transmit waveform for RRC puls with beta = 0.25
def h(t, Ts, beta):
    for t_i in t:
        if t_i == 0:
            return (1/Ts) * (1 + beta * ((4/np.pi) - 1))
        elif t_i == Ts/(4*beta) or t_i == -Ts/(4*beta):
            return (beta/(Ts * np.sqrt(2))) * ((1 + 2/np.pi) * (np.sin(np.pi/(4*beta))) + (1 - 2/np.pi) * (np.cos(np.pi/(4*beta))))
        else:
            return ((1/Ts) * (((np.sin(np.pi*t/Ts*(1-beta)) +  4*beta*t/Ts * np.cos(np.pi*t/Ts*(1+beta))))/((np.pi*(t/Ts))*(1-(4*beta*(t/Ts))**2))))


# Generate the time values
t = np.linspace(-10*Ts, 10*Ts, 160)

# Calculate the filter values
h_values = h(t, Ts, beta)
# Plot of  impulse response of root raised cosine filter
plt.figure(figsize=(10, 6))
plt.plot(t, h_values,".")
plt.title(' impulse response of root raised Cosine Filter')
plt.xlabel('Time (t)')
plt.ylabel('h(t)')
plt.grid(True)
plt.show()

#  Apply the RRC filter
x_shaped = np.convolve(samples, h_values)
plt.figure(2)
plt.plot(x_shaped, '.-')

# for i in range(num_symbols):
#     plt.plot([i*sps+num_taps//2,i*sps+num_taps//2], [0, x_shaped[i*sps+num_taps//2]])

plt.title('Transmit waveform for RRC pulse')
plt.grid(True)

# 7. Plot the spectrum of the transmitted signal using FFT
Fs = sps * (R/2) # Sampling rate
X = fftshift(fft(x_shaped, 800))
f_axis = np.arange(800) * Fs / 800 - Fs / 2
plt.figure(3)
plt.plot(f_axis / 1e6, 20 * np.log10(np.abs(X)))
plt.title('Spectrum of modulated signal')
plt.xlabel('f in MHz')
plt.ylabel('PSD in dB')
plt.grid(True)

# 7. Choose the appropriate matched filter
matched_filter = h_values[::-1]

# 8. Assume perfect synchronization at the receiver
# Apply the matched filter
y_matched = np.convolve(x_shaped, matched_filter)

# Sample at the symbol instants
received_symbols = y_matched[num_symbols*sps//2::sps]

# 9. Plot the constellation diagram
plt.figure(4)
plt.plot(received_symbols.real, received_symbols.imag, '.')
plt.title('Constellation Diagram with Matched Filter')
plt.grid(True)

plt.show()
