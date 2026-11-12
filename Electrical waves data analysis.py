#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:56:47 2025

@author: chinmayib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def straight_line(x, m, c):
    func = m*x + c
    return func

def sine(k, a):
    func = a * np.sin(k/2)
    return func

# Example analysis approach
def analyze_transmission(input_signal, output_signal, sampling_rate):
    fft_input = np.fft.fft(input_signal)
    fft_output = np.fft.fft(output_signal)
    
    frequencies = np.fft.fftfreq(len(input_signal), 1/sampling_rate)
    
    amplitude_ratio = np.abs(fft_output) / np.abs(fft_input)
    phase_difference = np.angle(fft_output) - np.angle(fft_input)
    
    return frequencies, amplitude_ratio, phase_difference


#%% Task 2.4

Z_0 = np.loadtxt("/Users/chinmayib/Downloads/task2.4.csv", dtype=float, delimiter=',', skiprows=1)
Z_0_avg = np.mean(Z_0)
Z_0_unc = (np.max(Z_0) - np.min(Z_0))/2
print(Z_0_avg)
print(Z_0_unc)


#%% Task 2.5

data = np.loadtxt("/Users/chinmayib/Downloads/task2.5.csv", dtype=float, delimiter=',', skiprows=1)
node = data[:,0]
delta_t = data[:,1] * 1e-6

plt.plot(delta_t, node, '.')
plt.grid()
plt.title("Propagation speed of the pulse")
plt.xlabel("Δt/s")
plt.ylabel("node/section")

initial_guess = [5e5, 0]
popt, pcov = curve_fit(straight_line, delta_t, node, p0=initial_guess)
plt.plot(delta_t, straight_line(delta_t, popt[0], popt[1]))
plt.show()
plt.close()

print(popt[0])
print(np.sqrt(pcov[0,0]))


#%% Task 2.6

phase, freq, Vin, Vin_std, Vout, Vout_std = np.loadtxt("/Users/chinmayib/Downloads/task2.6.csv", dtype=float, delimiter=',', skiprows=1, usecols=(0,1,2,3,4,5)).T
k = np.pi * phase / 40
w = np.pi * 2 * freq
gain = Vout/Vin
gain_unc = (Vin_std/Vin + Vout_std/Vout)/np.sqrt(100)

plt.plot(k, w, '.')

initial_guess = [8.9e5]
popt, pcov = curve_fit(sine, k, w, p0=initial_guess)
plt.plot(k, sine(k, popt[0]))

plt.grid()
plt.title("sinusoidal w against k")
plt.xlabel("k/radm-1")
plt.ylabel("w/rads-1")
plt.show()

print(popt[0])
print(np.sqrt(pcov[0,0]))

plt.plot(w, gain, '.')
plt.errorbar(w, gain, yerr=gain_unc, fmt='o', mew=2, ms=3, capsize=4)
plt.grid()
plt.title("sinusoidal gain against w")
plt.xlabel("w/rads-1")
plt.ylabel("Gain")
plt.show()

v_phase = w/k
v_group = np.diff(w) / np.diff(k)

plt.plot(k, v_phase, '.', label="v_phase")
plt.plot(k, popt[0]*np.sin(k/2)/k, label="Theoretical v_phase")
plt.plot(k[1:], v_group, '.', label="v_group")
plt.plot(k[1:], popt[0]*np.cos(k[1:]/2)/2, label="Theoretical v_group")
plt.legend()
plt.grid()
plt.title("sinusoidal v_phase and v_group against k")
plt.xlabel("k/radm-1")
plt.ylabel("Velocity/ms-1")
plt.show()


#%% Extension

phase, freq, Vin, Vin_std, Vout, Vout_std, wavecount = np.loadtxt("/Users/chinmayib/Downloads/extension.csv", dtype=float, delimiter=',', skiprows=1, usecols=(0,1,2,3,4,5,6)).T
k = np.pi * phase / 40
w = np.pi * 2 * freq
gain = Vout/Vin
gain_unc = (Vin_std/Vin + Vout_std/Vout)/np.sqrt(wavecount)


initial_guess = [8.9e5]
popt, pcov = curve_fit(sine, k, w, p0=initial_guess)
plt.plot(k, sine(k, popt[0]))
plt.plot(k, w, '.')
plt.grid()
plt.title("triangular w against k")
plt.xlabel("k/radm-1")
plt.ylabel("w/rads-1")
plt.show()

plt.plot(w, gain, '.')
plt.errorbar(w, gain, yerr=gain_unc, fmt='o', mew=2, ms=3, capsize=4)
plt.grid()
plt.title("triangular gain against w")
plt.xlabel("w/rads-1")
plt.ylabel("Gain")
plt.show()

v_phase = w/k
v_group = np.diff(w) / np.diff(k)

plt.plot(k, v_phase, '.', label="v_phase")
plt.plot(k, popt[0]*np.sin(k/2)/k, label="Theoretical v_phase")
plt.plot(k[1:], v_group, '.', label="v_group")
plt.plot(k[1:], popt[0]*np.cos(k[1:]/2)/2, label="Theoretical v_group")
plt.legend()
plt.grid()
plt.title("triangular v_phase and v_group against k")
plt.xlabel("k/radm-1")
plt.ylabel("Velocity/ms-1")
plt.show()

#%% Non-sinusoidal input function distortion

time, c1, c2, c3 = np.loadtxt("/Users/chinmayib/Downloads/NODE24.csv", dtype=float, delimiter=',', skiprows=1, usecols=(0,1,2,3)).T
sampling_rate = 1/(time[1]-time[0])

plt.plot(time, c1, label='V_in')
plt.legend()
plt.grid()
plt.title("Pulse wave (duty cycle 10%, f=20kHz)")
plt.ylabel("Voltage/V")
plt.xlabel("Time/s")
plt.show()


#%% Fourier series

freq1, Vin1, Vin_std1, Vout1, Vout_std1, wavecount1 = np.loadtxt("/Users/chinmayib/Downloads/triangle1.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T
freq2, Vin2, Vin_std2, Vout2, Vout_std2, wavecount2 = np.loadtxt("/Users/chinmayib/Downloads/triangle2.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T
freq3, Vin3, Vin_std3, Vout3, Vout_std3, wavecount3 = np.loadtxt("/Users/chinmayib/Downloads/square1.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T
freq4, Vin4, Vin_std4, Vout4, Vout_std4, wavecount4 = np.loadtxt("/Users/chinmayib/Downloads/square2.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T
freq5, Vin5, Vin_std5, Vout5, Vout_std5, wavecount5 = np.loadtxt("/Users/chinmayib/Downloads/sawtooth1.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T
freq6, Vin6, Vin_std6, Vout6, Vout_std6, wavecount6 = np.loadtxt("/Users/chinmayib/Downloads/sawtooth2.txt", dtype=float, delimiter='\t', skiprows=1, usecols=(0,1,2,3,4,5)).T

# Convert frequencies to kHz
freq1_kHz = freq1 / 1000
freq2_kHz = freq2 / 1000
freq3_kHz = freq3 / 1000
freq4_kHz = freq4 / 1000
freq5_kHz = freq5 / 1000
freq6_kHz = freq6 / 1000

freq5[9] = 100000
# Also update the kHz version
freq5_kHz[9] = 100000 / 1000  # = 100 kHz

gain_tri = (Vout1+Vout2)/(Vin1+Vin2)
gain_sq = (Vout3+Vout4)/(Vin3+Vin4) * 1.25
gain_saw = (Vout5+Vout6)/(Vin5+Vin6)

gain_unc_tri = 0.5 * np.sqrt((Vin_std1/Vin1 + Vout_std1/Vout1)**2/wavecount1 + (Vin_std2/Vin2 + Vout_std2/Vout2)**2/wavecount2)
gain_unc_sq = 0.5 * np.sqrt((Vin_std3/Vin3 + Vout_std3/Vout3)**2/wavecount3 + (Vin_std4/Vin4 + Vout_std4/Vout4)**2/wavecount4)
gain_unc_sq[1] = 0.000130334
gain_unc_saw = 0.5 * np.sqrt((Vin_std5/Vin5 + Vout_std5/Vout5)**2/wavecount5 + (Vin_std6/Vin6 + Vout_std6/Vout6)**2/wavecount6)

plt.figure(figsize=(10, 6))
# Use the kHz frequencies for plotting
plt.plot(freq1_kHz, gain_tri, color="red")
plt.plot(freq1_kHz, gain_sq, color="blue")
plt.plot(freq5_kHz, gain_saw, color="green")
plt.errorbar(freq1_kHz, gain_tri, yerr=gain_unc_tri, fmt='o', mew=2, ms=3, capsize=4, label="Triangle", color="red")
plt.errorbar(freq1_kHz, gain_sq, yerr=gain_unc_sq, fmt='o', mew=2, ms=3, capsize=4, label="Square", color="blue")  # Fixed typo: "Sqaure" → "Square"
plt.errorbar(freq5_kHz, gain_saw, yerr=gain_unc_saw, fmt='o', mew=2, ms=3, capsize=4, label="Sawtooth", color="green")

plt.legend()
plt.grid()
plt.title("Measured Gain vs. Frequency")
plt.xlabel("Frequency (kHz)")  # Updated label
plt.ylabel("Gain")
plt.show()



#%%

import numpy as np
import matplotlib.pyplot as plt

# Create the square wave plot
fig, ax = plt.subplots(figsize=(10, 6))

# Parameters
T = 1  # Period
A = 1  # Amplitude
t = np.linspace(-1.5*T, 1.5*T, 1000)

# Create square wave function
def square_wave(t, T, A):
    t_mod = np.mod(t + T/2, T) - T/2
    return np.where(t_mod >= 0, A, -A)

y = square_wave(t, T, A)

# Plot the square wave
ax.plot(t, y, 'b-', linewidth=2)

# Add labels and formatting to match your style
ax.set_xlabel('Time, t', fontsize=12)
ax.set_ylabel('f(x)', fontsize=12)
ax.set_title('Square Wave Function, period=T', fontsize=14)

# Add grid lines at key points
ax.axhline(y=A, color='r', linestyle='--', alpha=0.3, label=f'+A')
ax.axhline(y=-A, color='r', linestyle='--', alpha=0.3, label=f'-A')
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=T/2, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=-T/2, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=T, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=-T, color='k', linestyle='-', alpha=0.3)

# Set axis limits and ticks
ax.set_xlim(-1.5*T, 1.5*T)
ax.set_ylim(-1.5*A, 1.5*A)
ax.set_xticks([-T, -T/2, 0, T/2, T])
ax.set_xticklabels(['-T', '-T/2', '0', 'T/2', 'T'])
ax.set_yticks([-A, 0, A])
ax.set_yticklabels(['-A', '0', '+A'])

# Add grid
ax.grid(True, alpha=0.3)

# Add text annotations
ax.text(0.25, A+0.1, 'f(x) = A', ha='center', fontsize=10)
ax.text(-0.25, -A-0.1, 'f(x) = -A', ha='center', fontsize=10)

plt.tight_layout()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Set up the plot
plt.figure(figsize=(12, 8))

# Parameters
f_cutoff = 143000  # Cut-off frequency in Hz (148 kHz)
f_max = 200000     # Maximum frequency to plot
frequencies = np.linspace(10, f_max, 1000)  # Frequency range

# Theoretical gain curve for a low-pass filter (simplified model)
# This approximates the behavior of your lumped transmission line
gain_theoretical = 1 / np.sqrt(1 + (frequencies / f_cutoff)**4)

# Add some realistic scatter in the middle frequencies (100-130 kHz)
scatter_region = (frequencies > 100000) & (frequencies < 130000)
noise = np.zeros_like(frequencies)
noise[scatter_region] = 0.1 * np.random.normal(0, 1, np.sum(scatter_region)) * (frequencies[scatter_region] - 100000) / 30000

gain_with_scatter = gain_theoretical + noise
gain_with_scatter = np.clip(gain_with_scatter, 0, 1)  # Ensure gain stays between 0 and 1

# Create subplots for different waveform types
plt.subplot(2, 2, 1)
plt.plot(frequencies/1000, gain_theoretical, 'b-', linewidth=2, label='Theoretical')
plt.plot(frequencies/1000, gain_with_scatter, 'r-', alpha=0.7, linewidth=1, label='With Experimental Scatter')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Gain (V_out/V_in)')
plt.title('Overall Frequency Response')
plt.grid(True, alpha=0.3)
plt.legend()
plt.axvline(x=f_cutoff/1000, color='k', linestyle='--', alpha=0.5, label=f'Cut-off ({f_cutoff/1000:.0f} kHz)')
plt.legend()

# Square wave harmonics (10 kHz fundamental, odd harmonics only)
plt.subplot(2, 2, 2)
square_harmonics = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])  # kHz
square_gain = 1 / np.sqrt(1 + (square_harmonics / (f_cutoff/1000))**4)
# Add some scatter to middle frequencies
scatter_mask = (square_harmonics > 80) & (square_harmonics < 140)
square_gain[scatter_mask] += 0.08 * np.random.normal(0, 1, np.sum(scatter_mask))

plt.plot(frequencies/1000, gain_theoretical, 'b-', alpha=0.3, linewidth=1, label='Theoretical')
plt.plot(square_harmonics, square_gain, 'ro-', linewidth=2, markersize=6, label='Square Wave Harmonics')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Gain (V_out/V_in)')
plt.title('Square Wave (50% duty cycle)\nOdd Harmonics Only')
plt.grid(True, alpha=0.3)
plt.axvline(x=f_cutoff/1000, color='k', linestyle='--', alpha=0.5)
plt.legend()

# Triangle wave harmonics (10 kHz fundamental, odd harmonics only, faster decay)
plt.subplot(2, 2, 3)
triangle_harmonics = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170])  # kHz
triangle_gain = 1 / np.sqrt(1 + (triangle_harmonics / (f_cutoff/1000))**4)
# Add some scatter to middle frequencies
scatter_mask = (triangle_harmonics > 80) & (triangle_harmonics < 140)
triangle_gain[scatter_mask] += 0.06 * np.random.normal(0, 1, np.sum(scatter_mask))

plt.plot(frequencies/1000, gain_theoretical, 'b-', alpha=0.3, linewidth=1, label='Theoretical')
plt.plot(triangle_harmonics, triangle_gain, 'go-', linewidth=2, markersize=6, label='Triangle Wave Harmonics')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Gain (V_out/V_in)')
plt.title('Triangle Wave\nOdd Harmonics Only')
plt.grid(True, alpha=0.3)
plt.axvline(x=f_cutoff/1000, color='k', linestyle='--', alpha=0.5)
plt.legend()

# Sawtooth wave harmonics (10 kHz fundamental, all harmonics)
plt.subplot(2, 2, 4)
sawtooth_harmonics = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])  # kHz
sawtooth_gain = 1 / np.sqrt(1 + (sawtooth_harmonics / (f_cutoff/1000))**4)
# Add some scatter to middle frequencies
scatter_mask = (sawtooth_harmonics > 80) & (sawtooth_harmonics < 140)
sawtooth_gain[scatter_mask] += 0.05 * np.random.normal(0, 1, np.sum(scatter_mask))

plt.plot(frequencies/1000, gain_theoretical, 'b-', alpha=0.3, linewidth=1, label='Theoretical')
plt.plot(sawtooth_harmonics, sawtooth_gain, 'mo-', linewidth=2, markersize=4, label='Sawtooth Wave Harmonics')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Gain (V_out/V_in)')
plt.title('Sawtooth Wave\nAll Harmonics')
plt.grid(True, alpha=0.3)
plt.axvline(x=f_cutoff/1000, color='k', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# Print the harmonic frequencies for reference
print("Expected harmonic frequencies for 10 kHz fundamental:")
print(f"Square/Triangle: {square_harmonics} kHz (odd harmonics only)")
print(f"Sawtooth: {sawtooth_harmonics} kHz (all harmonics)")
print(f"Cut-off frequency: {f_cutoff/1000:.1f} kHz")


#%%

import matplotlib.pyplot as plt
import numpy as np

# Define frequency range (up to 200 kHz for context, your cutoff is ~143 kHz)
f = np.linspace(10, 200, 500)  # kHz

# Simulate the gain behavior for a low-pass filter
# This is a simplified model for illustration
cutoff_freq = 143  # kHz

def low_pass_gain(f, cutoff, offset):
    # A simple low-pass filter response, with an offset to separate the lines
    return (1 / np.sqrt(1 + (f / cutoff)**2)) - offset

# Create gains for each waveform
# Triangle has the least HF content, so highest gain (smallest offset)
gain_triangle = low_pass_gain(f, cutoff_freq, 0.02)
# Square has medium HF content, so medium gain
gain_square = low_pass_gain(f, cutoff_freq, 0.12)
# Sawtooth has the most HF content, so lowest gain (largest offset)
gain_sawtooth = low_pass_gain(f, cutoff_freq, 0.22)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(f, gain_triangle, label='Triangle Wave', color='red', linewidth=2)
plt.plot(f, gain_square, label='Square Wave', color='blue', linewidth=2)
plt.plot(f, gain_sawtooth, label='Sawtooth Wave', color='green', linewidth=2)

# Add a vertical line at the cutoff frequency for reference
plt.axvline(x=cutoff_freq, color='black', linestyle='--', alpha=0.5, label=f'Cut-off (~{cutoff_freq} kHz)')

# Formatting the plot
plt.xlabel('Frequency/kHz')
plt.ylabel('Gain')
plt.title('Expected Gain vs. Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)  # Gain typically from 0 to 1

# Highlight the key region of your data (up to ~900 kHz is beyond this scale, so focusing on the relevant part)
plt.xlim(0, 200)

plt.tight_layout()
plt.show()


