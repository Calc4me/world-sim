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

biome_colors = {
    'deep ocean': 'darkblue',
    'ocean': 'blue',
    'beach': 'khaki',
    'grassland': 'limegreen',
    'savanna': 'yellow',
    'desert': 'gold',
    'forest': 'green',
    'rainforest': 'darkgreen',
    'swamp': 'olive',
    'steppe': 'tan',
    'tundra': 'lightgray',
    'hills': 'sienna',
    'plateau': 'peru',
    'mountain': 'gray',
    'snowy peaks': 'white',
    'snow': 'lightblue'
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

def generate_noise_grid(width, height, scale, octaves, persistence, lacunarity, base):
    grid = generate_perlin_noise_with_octaves(
        width, height, scale, octaves, persistence, lacunarity, base
    )
    # Normalize grid to [0, 1] for convenience
    min_value = np.min(grid)
    max_value = np.max(grid)
    return (grid - min_value) / (max_value - min_value)


def classify_biome_optimized(height, temperature, rainfall):
    if height < 0.2:
        if temperature < 0.4:
            return 'deep ocean'
        return 'ocean'
    elif height < 0.3:
        return 'beach'
    elif height < 0.5:
        if rainfall > 0.7:
            return 'swamp'
        if temperature > 0.6:
            return 'savanna'
        if temperature < 0.3:
            return 'tundra'
        return 'grassland'
    elif height < 0.7:
        if rainfall > 0.8:
            return 'rainforest'
        if rainfall < 0.4:
            return 'steppe'
        return 'forest'
    elif height < 0.85:
        return 'hills'
    elif height < 0.95:
        return 'plateau'
    elif height < 1.0:
        if temperature < 0.3:
            return 'snowy peaks'
        return 'mountain'
    else:
        return 'snow'

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

def generate_master_biome_map_optimized(heightmap, temperature_map, rainfall_map):
    rows, cols = heightmap.shape
    master_biome_map = np.empty((rows, cols), dtype='<U20')  # Preallocate with string type

    for y in range(rows):
        for x in range(cols):
            height = heightmap[y, x]
            temperature = temperature_map[y, x]
            rainfall = rainfall_map[y, x]
            # Classify biome based on the optimized function
            master_biome_map[y, x] = classify_biome_optimized(height, temperature, rainfall)

    return master_biome_map
#Display map
def display_biome_map_with_modifiers(biome_array, heightmap, temperature_map, rainfall_map):
    """
    Display a biome map with colors modulated by vegetation and rainfall factors.
    
    biome_array: 2D array of biome strings (e.g., 'forest', 'desert', etc.)
    heightmap, temperature_map, rainfall_map: 2D arrays of corresponding modifiers.
    """
    # Biome colors
    BIOME_COLORS = {
        'deep ocean': np.array([0.0, 0.0, 0.5]),       # Dark Blue
        'ocean': np.array([0.0, 0.0, 1.0]),           # Blue
        'beach': np.array([0.9, 0.9, 0.6]),           # Khaki
        'grassland': np.array([0.5, 1.0, 0.5]),       # Lime Green
        'savanna': np.array([1.0, 1.0, 0.0]),         # Yellow
        'desert': np.array([1.0, 0.8, 0.4]),          # Gold
        'forest': np.array([0.0, 0.6, 0.0]),          # Green
        'rainforest': np.array([0.0, 0.4, 0.0]),      # Dark Green
        'swamp': np.array([0.4, 0.4, 0.0]),           # Olive
        'steppe': np.array([0.7, 0.5, 0.3]),          # Tan
        'tundra': np.array([0.8, 0.8, 0.8]),          # Light Gray
        'hills': np.array([0.6, 0.4, 0.2]),           # Sienna
        'plateau': np.array([0.8, 0.6, 0.4]),         # Peru
        'mountain': np.array([0.5, 0.5, 0.5]),        # Gray
        'snowy peaks': np.array([1.0, 1.0, 1.0]),     # White
        'snow': np.array([0.9, 0.9, 1.0]),            # Light Blue
    }
    
    # Prepare the color map
    width, height = biome_array.shape
    color_map = np.zeros((width, height, 3))  # RGB map

    for x in range(width):
        for y in range(height):
            # Get biome and base color
            biome = biome_array[x, y]
            base_color = BIOME_COLORS[biome]

            # Get modifiers
            veg_factor = (heightmap[x, y] + 1) / 2       # Normalize heightmap [-1, 1] to [0, 1]
            rain_factor = (rainfall_map[x, y] + 1) / 2   # Normalize rainfall [-1, 1] to [0, 1]

            # Adjust the biome color based on vegetation and rainfall
            vegetation_color = np.array([0.0, 0.5, 0.0]) * veg_factor  # Greenish tint for vegetation
            rainfall_color = np.array([0.0, 0.0, 0.7]) * rain_factor   # Blueish tint for rainfall

            # Combine base color with vegetation and rainfall modifiers
            blended_color = base_color * 0.5 + vegetation_color * 0.3 + rainfall_color * 0.2
            color_map[x, y] = np.clip(blended_color, 0, 1)  # Ensure RGB values are in range [0, 1]

    # Display the map
    plt.figure(figsize=(10, 10))
    plt.imshow(color_map, origin='upper', interpolation='nearest')
    plt.axis('off')
    plt.title("Biome Map with Vegetation and Rainfall")
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

def display_map(grid, title, cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, origin='upper')
    plt.title(title)
    plt.colorbar()
    plt.show(block=False)

#base_noise_map = generate_perlin_noise_with_octaves(MAP_WIDTH, MAP_HEIGHT, scale=SCALE, base=BASE)
vegetation_noise_map = generate_perlin_noise_with_octaves(MAP_WIDTH, MAP_HEIGHT, scale=(SCALE/5), base=BASE)
heightmap = generate_noise_grid(MAP_WIDTH, MAP_HEIGHT, SCALE, octaves=6, persistence=0.5, lacunarity=2.0, base=BASE)
temperature_map = generate_noise_grid(MAP_WIDTH, MAP_HEIGHT, SCALE, octaves=4, persistence=0.6, lacunarity=2.5, base=BASE)
rainfall_map = generate_noise_grid(MAP_WIDTH, MAP_HEIGHT, scale=SCALE*2, octaves=5, persistence=0.7, lacunarity=2.0, base=BASE)

# biome_array = assign_biome_properties(base_noise_map, vegetation_noise_map, rainfall_noise_map)
# print("Seed: " + str(BASE))
# print("Size: " + str(MAP_HEIGHT) + "x" + str(MAP_WIDTH))
# display_biome_map(biome_array)
# display_base_biome_map(biome_array)
# show_noise_maps([vegetation_noise_map, rainfall_noise_map], ["Vegetation", "Rainfall"])
# plt.show()
show_noise_maps([heightmap, temperature_map, rainfall_map, vegetation_noise_map], ["Heightmap", "Temp map", "Rain map", "Veg. Map"])
master_biome_map = generate_master_biome_map_optimized(vegetation_noise_map, temperature_map, rainfall_map)
biome_to_num = {biome: i for i, biome in enumerate(biome_colors.keys())}
numerical_biome_map = np.vectorize(biome_to_num.get)(master_biome_map)
display_biome_map_with_modifiers(master_biome_map, heightmap, temperature_map, rainfall_map)
# Display the master biome map
cmap = ListedColormap(list(biome_colors.values()))
plt.figure(figsize=(8, 8))
plt.imshow(numerical_biome_map, cmap=cmap, origin='upper', interpolation='nearest')
plt.title("Biome Map")
plt.colorbar(ticks=range(len(biome_colors)), label="Biome Types")
plt.clim(-0.5, len(biome_colors) - 0.5)
plt.show(block=False)
plt.show()