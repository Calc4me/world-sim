import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Define corresponding biome names
biome_names = [
    'deep_ocean', 'ocean', 'high_ocean', 'river', 
    'plains', 'forest', 'hills', 'mountain', 'snow', 'ice'
]

# Define biome thresholds and base colors
DEEP_OCEAN_THRESHOLD = -0.75
OCEAN_THRESHOLD = -0.5
HIGH_OCEAN_THRESHOLD = -0.25
RIVER_THRESHOLD = -0.15
PLAINS_THRESHOLD = 0.15
FOREST_THRESHOLD = 0.4
HILLS_THRESHOLD = 0.65
MOUNTAIN_THRESHOLD = 0.85
SNOW_THRESHOLD = 0.95
ICE_THRESHOLD = 1

BASE_BLEND = 1
VE_BLEND = 0.3
RAIN_BLEND = 0.4

BIOME_COLORS = {
    'deep_ocean': (5, 17, 36),   # Dark Blue
    'ocean': (0, 40, 72),        # Blue
    'high_ocean': (16, 62, 98), # Light Blue
    'river': (30, 88, 128),      # Cyan
    'plains': (60, 112, 45),     # Yellow
    'forest': (27, 59, 27),     # Forest Green
    'hills': (110, 83, 60),     # Brown
    'mountain': (74, 70, 67), # Gray
    'snow': (158, 154, 152),     # White
    'ice': (237, 242, 237)       # Light Cyan
}
BIOME_COLORS = {k: np.array(v) / 255.0 for k, v in BIOME_COLORS.items()}

# Define thresholds in a list (sorted)
thresholds = [
    DEEP_OCEAN_THRESHOLD, OCEAN_THRESHOLD, HIGH_OCEAN_THRESHOLD,
    RIVER_THRESHOLD, PLAINS_THRESHOLD, FOREST_THRESHOLD, 
    HILLS_THRESHOLD, MOUNTAIN_THRESHOLD, SNOW_THRESHOLD, ICE_THRESHOLD
]

#Map dimensions
MAP_WIDTH = 60
MAP_HEIGHT = 60
SCALE = 60  # Adjust to control the "zoom" level of terrain (high -> zoom in, low -> zoom out)
BASE = random.randint(0,10**6) # "seed"

# A basic implementation of a fade function (to ease the interpolation)
def fade(t):
    return 1 / (1 + 4 ** ((-1) * ((8 * (t - 0.5))/(0.9 - (t - 0.5) ** 2))))

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

def assign_biome_properties(noise_map, veg_map, rain_map):
    thresholds = [
        DEEP_OCEAN_THRESHOLD, OCEAN_THRESHOLD, HIGH_OCEAN_THRESHOLD,
        RIVER_THRESHOLD, PLAINS_THRESHOLD, FOREST_THRESHOLD, 
        HILLS_THRESHOLD, MOUNTAIN_THRESHOLD, SNOW_THRESHOLD, ICE_THRESHOLD
    ]
    biome_names = [
        'deep_ocean', 'ocean', 'high_ocean', 'river', 
        'plains', 'forest', 'hills', 'mountain', 'snow', 'ice'
    ]
    
    biome_indices = np.digitize(noise_map, thresholds) - 1
    biome_indices = np.clip(biome_indices, 0, len(biome_names) - 1)
    
    biome_array = []
    for x in range(noise_map.shape[0]):
        row = []
        for y in range(noise_map.shape[1]):
            biome = biome_names[biome_indices[x, y]]
            row.append({
                "biome": biome,
                "value": noise_map[x, y],
                "veg": veg_map[x, y],
                "rain": rain_map[x, y]
            })
        biome_array.append(row)
    
    return np.array(biome_array)

#Display map
def display_biome_map(biome_array):
    width, height = biome_array.shape
    
    # Create a color map for visualization
    color_map = np.zeros((width, height, 3))  # RGB map
    
    for x in range(width):
        for y in range(height):
            cell = biome_array[x, y]
            base_color = BIOME_COLORS[cell["biome"]]
            
            # Modulate base color by vegetation and rainfall
            veg_factor = 1 - ((cell["veg"] + 1) / 2 ) # Normalize [-1, 1] to [0, 1]
            rain_factor = (cell["rain"] + 1) / 2
            
            # Flip rainfall contribution: higher rain = darker
            inverted_rain_factor = 1 - rain_factor
            
            # Blend base color with vegetation (greenish) and rainfall (darker for high rain)
            blended_color = (
                BASE_BLEND * base_color + 
                VE_BLEND * np.array([0.2 * veg_factor, 0.4 * veg_factor, 0.2 * veg_factor]) +  # Green for vegetation
                RAIN_BLEND * np.array([0.2 * inverted_rain_factor, 0.3 * inverted_rain_factor, 1.0 * inverted_rain_factor])  # Blue for low rain
            )
            color_map[x, y] = np.clip(blended_color, 0, 1)  # Ensure RGB values are in range [0, 1]
    
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map, origin='upper', interpolation='nearest')
    plt.axis('off')
    plt.title("Biome Map with Vegetation and Rainfall (Flipped Rainfall)")
    plt.show(block=False)

def display_base_biome_map(biome_array):
    """
    Displays the base biome map using only the biome colors.
    """
    width, height = biome_array.shape

    # Create a color map for visualization
    color_map = np.zeros((width, height, 3))  # RGB map

    for x in range(width):
        for y in range(height):
            cell = biome_array[x, y]
            base_color = BIOME_COLORS[cell["biome"]]
            color_map[x, y] = base_color

    # Display the map
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map, origin='upper', interpolation='nearest')
    plt.axis('off')
    plt.title("Base Biome Map")
    plt.show(block=False)

def show_noise_maps(maps, titles):
    """
    Display multiple noise maps side by side for comparison.

    Parameters:
    - maps: List of 2D arrays (noise maps) to display.
    - titles: List of titles for the corresponding maps.
    - cmap: Colormap to use for the plots.
    """
    num_maps = len(maps)
    fig, axes = plt.subplots(1, num_maps, figsize=(5 * num_maps, 5))
    
    for i, ax in enumerate(axes):
        im = ax.imshow(maps[i], cmap='seismic', origin='upper')
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)
    
    plt.tight_layout()
    plt.show(block=False)

base_noise_map = generate_perlin_noise_with_octaves(MAP_WIDTH, MAP_HEIGHT, scale=SCALE, base=BASE)
vegetation_noise_map = generate_perlin_noise_with_octaves(MAP_WIDTH, MAP_HEIGHT, scale=(SCALE/5), base=BASE)
rainfall_noise_map = generate_perlin_noise_with_octaves(MAP_WIDTH, MAP_WIDTH, scale=SCALE*2, base=BASE)

biome_array = assign_biome_properties(base_noise_map, vegetation_noise_map, rainfall_noise_map)
print("Seed: " + str(BASE))
print("Size: " + str(MAP_HEIGHT) + "x" + str(MAP_WIDTH))
display_biome_map(biome_array)
display_base_biome_map(biome_array)
show_noise_maps([vegetation_noise_map, rainfall_noise_map], ["Vegetation", "Rainfall"])
plt.show()