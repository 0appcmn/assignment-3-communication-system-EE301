# assignment-3-communication-system-EE301
chandrakant keshari(12140500)

problem statement 
1. Deadline on or before Thursday 23-11-2023 (No late submission allowed)
2. Generate a binary stream for 20 symbols for 4-PAM modulation
3. Bit rate of 20Mbps
4. sps 8 (Sample per symbol)
5. Plot the transmit waveform for square pulse
6. Plot the transmit waveform for RRC pulse with beta=0.25

7. Plot the spectrum of the transmitted signal using FFT function of matlab/Python

Hint for plotting spectrum: R=20 Mbps, which implies 10*10^6 symbols per second (2 bits per symbol for 4-PAM). Samples per symbol (sps) is 8. Thus the sampling rate is Fs=8*10*10^6.  Thus to plot FFT assume Fs=80MHz. Let vector xm is the samples of modulated signal, ie, 8*20=160 samples.

X=fftshift(fft(yc,800));
f_axis=(0:800-1)*Fs/800-Fs/2;
plot(f_axis/1e6,20*log10(abs(X)))
title('Spectrum of modulated signal')
xlabel('f in MHz')
ylabel('PSD in dB')
grid on

8. Choose the appropriate matched filter for 5 and 6
9. Assume perfect synchronization at the receiver and plot the constellation diagram for both 5&6 with matched filter mentioned in 8.
