import random
import math

# Representing relationships as a dictionary
diplomatic_relations = {
    ("NationA", "NationB"): {"relationship_score": 50, "treaties": ["trade"], "grievances": []},
    ("NationB", "NationC"): {"relationship_score": -30, "treaties": [], "grievances": ["border_dispute"]},
}

def update_relationship(nation1, nation2, change):
    # Update relationship score with limits
    pair = (nation1, nation2)
    if pair in diplomatic_relations:
        diplomatic_relations[pair]["relationship_score"] = max(-100, min(100, 
            diplomatic_relations[pair]["relationship_score"] + change))
    else:
        diplomatic_relations[pair] = {"relationship_score": change, "treaties": [], "grievances": []}

def propose_alliance(nation1, nation2):
    pair = (nation1, nation2)
    if diplomatic_relations.get(pair, {}).get("relationship_score", 0) > 60:
        diplomatic_relations[pair]["treaties"].append("alliance")
        print(f"{nation1} and {nation2} have formed an alliance!")
    else:
        print(f"{nation2} rejects the alliance proposal.")

def declare_war(nation1, nation2):
    pair = (nation1, nation2)
    diplomatic_relations[pair]["relationship_score"] = -100
    diplomatic_relations[pair]["treaties"] = []
    print(f"{nation1} has declared war on {nation2}!")

def ai_diplomatic_decision(nation, other_nations):
    for other in other_nations:
        if diplomatic_relations.get((nation, other), {}).get("relationship_score", 0) > 60:
            propose_alliance(nation, other)
        elif diplomatic_relations.get((nation, other), {}).get("relationship_score", 0) < -50:
            declare_war(nation, other)

events = [
    {"name": "Earthquake", "impact": "population_loss", "severity": random.randint(5, 15)},
    {"name": "Plague", "impact": "population_loss", "severity": random.randint(10, 30)},
    {"name": "Scientific Breakthrough", "impact": "tech_boost", "severity": random.randint(5, 20)}
]

def trigger_event():
    event = random.choice(events)
    print(f"Event Triggered: {event['name']}! Impact: {event['impact']} Severity: {event['severity']}")


class Tile:
    def __init__(self, x, y, owner=None):
        self.x = x
        self.y = y
        self.owner = owner  # City or empire owning the tile


# Define a city with cultural attributes
class City:
    def __init__(self, name, x, y, population, culture_strength):
        self.name = name
        self.x = x  # Map coordinates
        self.y = y
        self.population = population
        self.culture_strength = culture_strength  # Determines influence
        self.influence_radius = math.sqrt(culture_strength)  # Simple scaling

    def expand_culture(self):
        # Culture grows based on population and strength
        growth_factor = random.uniform(0.01, 0.05)  # Random growth
        self.culture_strength += int(self.population * growth_factor)
        self.influence_radius = math.sqrt(self.culture_strength)

    def generate_research(self, education_level, wealth):
        # Research points generation formula
        self.research_points += education_level * wealth * 0.1

    def unlock_technology(self, tech):
        if self.research_points >= tech.cost:
            self.technologies.append(tech.name)
            self.research_points -= tech.cost
            print(f"{self.name} unlocked {tech.name}!")

    def expand_territory(self, map_tiles):
        # Try to expand territory based on population
        expansion_attempts = int(self.population / 1000)  # Expansion depends on size
        for _ in range(expansion_attempts):
            adjacent_tile = random.choice(self.get_adjacent_tiles(map_tiles))
            if adjacent_tile and not adjacent_tile.owner:
                adjacent_tile.owner = self.name
                self.territory.append((adjacent_tile.x, adjacent_tile.y))

    def get_adjacent_tiles(self, map_tiles):
        # Find adjacent tiles to the city's territory
        adjacent_tiles = []
        for tile in map_tiles:
            for city_tile in self.territory:
                if abs(tile.x - city_tile[0]) <= 1 and abs(tile.y - city_tile[1]) <= 1:
                    adjacent_tiles.append(tile)
        return adjacent_tiles



# Define technology levels and cities with tech attributes
class Technology:
    def __init__(self, name, cost):
        self.name = name
        self.cost = cost  # Research points needed to unlock


# Calculate cultural influence on a neighboring city
def calculate_influence(city1, city2):
    distance = math.sqrt((city1.x - city2.x)**2 + (city1.y - city2.y)**2)
    if distance <= city1.influence_radius:
        influence = city1.culture_strength / (distance + 1)  # Prevent divide by zero
        return influence
    return 0

# Generate map tiles
map_tiles = [Tile(x, y) for x in range(50) for y in range(50)]

# Example city expanding
city_d = City("Rome", 25, 25, 5000)
city_d.expand_territory(map_tiles)
print(f"{city_d.name} controls tiles: {city_d.territory}")


# Example cities
city_a = City("Athens", 10, 10, 5000, 1000)
city_b = City("Sparta", 15, 15, 3000, 500)

# Expand culture and calculate influence
city_a.expand_culture()
influence_on_b = calculate_influence(city_a, city_b)

print(f"{city_a.name}'s influence on {city_b.name}: {influence_on_b:.2f}")

# Define technologies
wheel = Technology("Wheel", 500)
agriculture = Technology("Agriculture", 300)

# Example city
city_c = City("Babylon")
city_c.generate_research(education_level=10, wealth=20)  # Adjust education/wealth

print(f"{city_c.name} has {city_c.research_points} research points.")
city_c.unlock_technology(agriculture)

