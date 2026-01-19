"""Synthetic time series generators shared by validation scripts."""

import numpy as np


def generate_sine_wave(n=1000, freq=0.1):
    """Pure sine wave - simplest, most predictable."""
    t = np.arange(n)
    return np.sin(2 * np.pi * freq * t)


def generate_constant(n=1000, value=0.0):
    """Constant value - no dynamics, should have negative lambda."""
    return np.full(n, value)


def generate_multi_frequency(n=1000, freqs=None):
    """Multi-frequency wave - more complex spectrum but still periodic."""
    if freqs is None:
        freqs = [0.05, 0.1, 0.15]
    t = np.arange(n)
    signal = np.zeros(n)
    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t)
    return signal / len(freqs)


def generate_noisy_multi_frequency(n=1000, freqs=None, noise_std=0.05):
    """Noisy multi-frequency - complex + small stochastic component."""
    if freqs is None:
        freqs = [0.05, 0.1, 0.15]
    signal = generate_multi_frequency(n, freqs)
    noise = np.random.normal(0, noise_std, n)
    return signal + noise


def generate_lorenz(n=1000, dt=0.02, sigma=15, rho=28, beta=8/3):
    """Lorenz system - highly chaotic attractor."""
    x, y, z = 0.0, 1.0, 1.05
    trajectory = np.zeros(n)
    
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        
        x += dx * dt
        y += dy * dt
        z += dz * dt
        
        trajectory[i] = x  # Use x component as time series
    
    return trajectory


def generate_white_noise(n=1000):
    """White noise - maximum unpredictability."""
    return np.random.normal(0, 1, n)
