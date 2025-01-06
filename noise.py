import math
import random
import matplotlib.pyplot as plt
import numpy as np

# A basic implementation of a fade function (to ease the interpolation)
def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

# A basic implementation of linear interpolation
def lerp(a, b, t):
    return a + t * (b - a)

# A function to generate a gradient at a given coordinate (x, y)
def grad(hash, x, y):
    h = hash & 15  # Limit to 16 gradients
    grad = 1 + (h & 7)  # Gradient value is in range 1-8
    if h & 8:
        grad = -grad
    return grad * (x + y)

# A function to generate Perlin noise (without using pnoise2)
def perlin_noise(x, y, scale=100, base=42):
    random.seed(base)
    perm = [i for i in range(256)]
    random.shuffle(perm)
    perm += perm  # Duplicate to avoid wrapping

    X = int(x / scale) & 255
    Y = int(y / scale) & 255

    xf = (x / scale) - int(x / scale)
    yf = (y / scale) - int(y / scale)

    u = fade(xf)
    v = fade(yf)

    aa = perm[X + perm[Y]] % 256
    ab = perm[X + perm[Y + 1]] % 256
    ba = perm[X + 1 + perm[Y]] % 256
    bb = perm[X + 1 + perm[Y + 1]] % 256

    x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
    x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)
    return lerp(x1, x2, v)

# Function to generate Perlin noise with octaves (fractal noise)
def perlin_noise_with_octaves(x, y, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, base=42):
    total = 0.0
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += perlin_noise(x * frequency, y * frequency, scale, base) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity

    return total / max_value

# Function to generate a 2D grid of Perlin noise with octaves
def generate_perlin_noise_with_octaves(width, height, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, base=42):
    noise_grid = []
    min_value = float('inf')
    max_value = float('-inf')
    
    # Generate noise and find min/max values
    for y in range(height):
        row = []
        for x in range(width):
            noise_value = perlin_noise_with_octaves(x, y, scale, octaves, persistence, lacunarity, base)
            row.append(noise_value)
            min_value = min(min_value, noise_value)
            max_value = max(max_value, noise_value)
        noise_grid.append(row)

    # Normalize the noise to be in the range [-1, 1]
    normalized_grid = np.array(noise_grid)
    normalized_grid = 2 * (normalized_grid - min_value) / (max_value - min_value) - 1
    
    return normalized_grid

# Example usage: generate a 50x50 Perlin noise grid with octaves
width = 50
height = 50
scale = 10
octaves = 6
persistence = 0.5
lacunarity = 2.0
MIN = 0
MAX = 10**6
base = random.randint(MIN,MAX)
noise_grid = generate_perlin_noise_with_octaves(width, height, scale, octaves, persistence, lacunarity, base)

# Plot the Perlin noise using matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(noise_grid, cmap='viridis', origin='lower', interpolation='lanczos')
plt.colorbar()  # Show color scale
plt.title("Perlin Noise Visualization")
plt.show()
