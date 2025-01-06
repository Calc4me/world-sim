#THIS IS AN OVERVIEW THIS DOES NOT WORK WILL FIX LATER DO NOT RUN
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Constants for map dimensions and noise
MAP_WIDTH = 50
MAP_HEIGHT = 50
SCALE = 10  # Adjust to control the "zoom" level of terrain
BASE = random.randint(0,10**6)

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

# Generate Perlin noise-based map
def generate_terrain_map(width, height, scale):
    return generate_perlin_noise_with_octaves(width, height, scale, base=BASE)
def value_to_biome(value):
    if value <= -0.6:
        return 'ocean'
    elif value <= -0.2:
        return 'river'
    elif value <= 0.2:
        return 'plains'
    elif value <= 0.4:
        return 'forest'
    elif value <= 0.6:
        return 'hills'
    elif value <= 0.8:
        return 'mountain'
    else:
        return 'snow'

# Function to generate the biome map from the noise grid
def generate_biome_map(noise_grid):
    biome_map = []
    for row in noise_grid:
        biome_row = [value_to_biome(value) for value in row]
        biome_map.append(biome_row)
    return np.array(biome_map)

# Function to map biome names to numbers
def biome_to_number(biome_map):
    # Define a dictionary that maps biome names to unique integers
    biome_to_num = {
        'ocean': 0,
        'river': 1,
        'plains': 2,
        'forest': 3,
        'hills': 4,
        'mountain': 5,
        'snow': 6
    }
    # Convert biome names to numbers
    return np.vectorize(biome_to_num.get)(biome_map)

# Function to display the biome map with custom colors
def display_biome_map(biome_map):
    # Define the color mapping for each biome
    biome_colors = {
        'ocean': 'blue',
        'river': 'cyan',
        'plains': 'yellow',
        'forest': 'green',
        'hills': 'orange',
        'mountain': 'gray',
        'snow': 'white'
    }

    # Convert biome names to their corresponding color
    color_map = np.vectorize(biome_colors.get)(biome_map)

    # Create a colormap using the color list
    cmap = ListedColormap(list(biome_colors.values()))

    # Create a numerical representation of the biome map
    numerical_biome_map = biome_to_number(biome_map)

    # Display the biome map with custom colors using imshow
    plt.figure(figsize=(8, 8))
    plt.imshow(numerical_biome_map, cmap=cmap, origin='upper', interpolation='nearest')
    plt.colorbar(label='Biome Types')  # Add colorbar for reference
    plt.title("World Biome Map")
    plt.show()

ngrid = generate_terrain_map(MAP_WIDTH,MAP_HEIGHT,SCALE)
bmap = generate_biome_map(ngrid)
print(bmap)
display_biome_map(bmap)
OCEAN_THRESHOLD = 0.2
RIVER_THRESHOLD = 0.3
PLAINS_THRESHOLD = 0.45
FOREST_THRESHOLD = 0.6
HILLS_THRESHOLD = 0.7
MOUNTAIN_THRESHOLD = 0.85
SNOW_THRESHOLD = 1.0



