

# %%
import pygame
import heapq
import math
import random

# Initialize Pygame
pygame.init()



# %%
# Constants
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
BROWN = (139, 69, 19)
LIGHT_BLUE = (173, 216, 230)
GRAY = (128, 128, 128)

# Terrain types and their costs
FLAT = 0
FOREST = 1
MOUNTAIN = 2
WATER = 3
TRAIL = 4
SCENIC = 5

# Cost for each terrain type
TERRAIN_COSTS = {
    FLAT: 1.0,
    FOREST: 2.0,
    MOUNTAIN: 5.0,
    WATER: 10.0,
    TRAIL: 0.8,
    SCENIC: 0.9
}

# Colors for each terrain type
TERRAIN_COLORS = {
    FLAT: (240, 240, 200),
    FOREST: (34, 139, 34),
    MOUNTAIN: (139, 137, 137),
    WATER: (64, 164, 223),
    TRAIL: (210, 180, 140),
    SCENIC: (152, 251, 152)
}



# %%
class HikingNavigator:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hiking Trail Navigator - A* Pathfinding")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        # Create terrain map
        self.terrain = self.generate_terrain()
        self.elevation = self.generate_elevation()
        
        # Pathfinding
        self.start_point = None
        self.end_point = None
        self.current_path = []
        self.visited_nodes = set()
        
        # Preferences
        self.distance_weight = 1.0
        self.elevation_weight = 1.0
        self.scenic_weight = 0.5
        self.avoid_water = True
        
        # Path metrics
        self.path_distance = 0
        self.elevation_gain = 0
        self.calculation_time = 0
        self.nodes_explored = 0
        
        # UI state
        self.show_elevation = True
        self.show_path = True
        self.show_visited = False
        
        # Adding some trails and points of interest
        self.add_trails()
        self.points_of_interest = self.add_points_of_interest()
        
        # Running flag
        self.running = True
        
    def generate_terrain(self):
        """Generating a simple terrain map"""
        terrain = [[FLAT for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Adding some forests (clusters)
        for _ in range(5):
            x, y = random.randint(5, GRID_WIDTH-6), random.randint(5, GRID_HEIGHT-6)
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if random.random() < 0.7 and 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                        terrain[y+dy][x+dx] = FOREST
        
        # Adding some mountains
        for _ in range(3):
            x, y = random.randint(5, GRID_WIDTH-6), random.randint(5, GRID_HEIGHT-6)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if random.random() < 0.8 and 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                        terrain[y+dy][x+dx] = MOUNTAIN
        
        # Adding some water
        for _ in range(2):
            x, y = random.randint(5, GRID_WIDTH-6), random.randint(5, GRID_HEIGHT-6)
            length = random.randint(5, 15)
            direction = random.choice([(1, 0), (0, 1), (1, 1), (-1, 1)])
            
            for i in range(length):
                nx, ny = x + i * direction[0], y + i * direction[1]
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    terrain[ny][nx] = WATER
                    # Add some width
                    for w in range(-1, 2):
                        if 0 <= ny+w < GRID_HEIGHT and random.random() < 0.7:
                            terrain[ny+w][nx] = WATER
        
        return terrain
    
    def generate_elevation(self):
        """Generate elevation data"""
        elevation = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        
        # Adding some hills
        for _ in range(5):
            x, y = random.randint(5, GRID_WIDTH-6), random.randint(5, GRID_HEIGHT-6)
            peak = random.randint(500, 1500)
            radius = random.randint(5, 10)
            
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    if 0 <= x+dx < GRID_WIDTH and 0 <= y+dy < GRID_HEIGHT:
                        distance = math.sqrt(dx*dx + dy*dy)
                        if distance <= radius:
                            # Higher in the center, lower at edges
                            factor = (radius - distance) / radius
                            elevation[y+dy][x+dx] += int(peak * factor * factor)
        
        # Mountains have higher elevation
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.terrain[y][x] == MOUNTAIN:
                    elevation[y][x] += random.randint(300, 700)
        
        return elevation
    
    def add_trails(self):
        """Add hiking trails to the map"""
        # Create a few random trails
        for _ in range(3):
            # Random start point
            x, y = random.randint(2, GRID_WIDTH-3), random.randint(2, GRID_HEIGHT-3)
            
            # Random length
            length = random.randint(10, 30)
            
            # Create trail
            for i in range(length):
                # Random direction change
                if random.random() < 0.3:
                    dx, dy = random.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                else:
                    # Continue in same direction
                    dx, dy = (0, 1) if i == 0 else (dx, dy)
                
                # Update position
                x, y = x + dx, y + dy
                
                # Check bounds
                if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT:
                    if self.terrain[y][x] != WATER:  # Don't put trails on water
                        self.terrain[y][x] = TRAIL
    
    def add_points_of_interest(self):
        """Add scenic viewpoints to the map"""
        points = []
        
        # Add some scenic viewpoints (typically on higher elevation)
        for _ in range(10):
            # Find a spot with good elevation
            candidates = []
            for y in range(GRID_HEIGHT):
                for x in range(GRID_WIDTH):
                    if self.elevation[y][x] > 800 and self.terrain[y][x] != WATER:
                        candidates.append((x, y))
            
            if candidates:
                x, y = random.choice(candidates)
                points.append((x, y))
                self.terrain[y][x] = SCENIC
        
        return points
    
    def handle_events(self):
        """Process user input"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_e:
                    self.show_elevation = not self.show_elevation
                elif event.key == pygame.K_p:
                    self.show_path = not self.show_path
                elif event.key == pygame.K_v:
                    self.show_visited = not self.show_visited
                elif event.key == pygame.K_1:
                    self.distance_weight = max(0.1, self.distance_weight - 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_2:
                    self.distance_weight = min(5.0, self.distance_weight + 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_3:
                    self.elevation_weight = max(0.1, self.elevation_weight - 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_4:
                    self.elevation_weight = min(5.0, self.elevation_weight + 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_5:
                    self.scenic_weight = max(0.0, self.scenic_weight - 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_6:
                    self.scenic_weight = min(2.0, self.scenic_weight + 0.1)
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_w:
                    self.avoid_water = not self.avoid_water
                    if self.start_point and self.end_point:
                        self.calculate_path()
                elif event.key == pygame.K_n:
                    # Generate new map
                    self.terrain = self.generate_terrain()
                    self.elevation = self.generate_elevation()
                    self.add_trails()
                    self.points_of_interest = self.add_points_of_interest()
                    self.start_point = None
                    self.end_point = None
                    self.current_path = []
                    self.visited_nodes = set()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position and convert to grid coordinates
                mouse_pos = pygame.mouse.get_pos()
                grid_x, grid_y = mouse_pos[0] // GRID_SIZE, mouse_pos[1] // GRID_SIZE
                
                # checking if valid position
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    # Set start/end points
                    if event.button == 1:  # Left click
                        if not self.start_point:
                            self.start_point = (grid_x, grid_y)
                        elif not self.end_point:
                            self.end_point = (grid_x, grid_y)
                            self.calculate_path()
                        else:
                            # Reset and set new start point
                            self.start_point = (grid_x, grid_y)
                            self.end_point = None
                            self.current_path = []
                            self.visited_nodes = set()
    
    def calculate_path(self):
        """Calculating optimal hiking path using A*"""
        import time
        
        # Reset metrics
        self.visited_nodes = set()
        self.nodes_explored = 0
        
        # Time the calculation timing
        start_time = time.time()
        self.current_path = self.a_star_pathfinding(self.start_point, self.end_point)
        end_time = time.time()
        
        self.calculation_time = end_time - start_time
        
        # Calculating path metrics
        self.calculate_path_metrics()
    
    def a_star_pathfinding(self, start, end):
        """A* pathfinding algorithm optimized for hiking"""
        if not start or not end:
            return []
        
        # Reset nodes explored counter
        self.nodes_explored = 0
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Start with initial node
        # Priority queue with: (f_score, position, path)
        heapq.heappush(open_set, (0, start, []))
        
        # Track g scores (cost from start)
        g_scores = {start: 0}
        
        while open_set:
            # Getting node with lowest f_score
            _, current, path = heapq.heappop(open_set)
            
            # Counting nodes explored for metrics
            self.nodes_explored += 1
            
            # Adding to visited for visualization
            self.visited_nodes.add(current)
            
            # Check if reached goal
            if current == end:
                return path + [current]
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check neighbors (including diagonals)
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # Check if valid position
                if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                    continue
                
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate movement cost
                move_cost = self.calculate_movement_cost(current, neighbor)
                
                # Diagonal movement costs more
                if dx != 0 and dy != 0:
                    move_cost *= 1.414
                
                # Skip if movement is blocked (like water if avoid_water is on)
                if move_cost == float('inf'):
                    continue
                
                # Calculate tentative g score
                tentative_g = g_scores[current] + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # This path is better, record it
                    g_scores[neighbor] = tentative_g
                    
                    # Calculate f score (g + heuristic)
                    h_score = self.hiking_heuristic(neighbor, end)
                    f_score = tentative_g + h_score
                    
                    # Add to open set with updated path
                    heapq.heappush(open_set, (f_score, neighbor, path + [current]))
        
        # No path found
        return []
    
    def calculate_movement_cost(self, p1, p2):
        """Calculating cost of movement between two points"""
        x1, y1 = p1
        x2, y2 = p2
        
        # Get terrain and elevation
        terrain1 = self.terrain[y1][x1] if 0 <= x1 < GRID_WIDTH and 0 <= y1 < GRID_HEIGHT else FLAT
        terrain2 = self.terrain[y2][x2] if 0 <= x2 < GRID_WIDTH and 0 <= y2 < GRID_HEIGHT else FLAT
        elev1 = self.elevation[y1][x1] if 0 <= x1 < GRID_WIDTH and 0 <= y1 < GRID_HEIGHT else 0
        elev2 = self.elevation[y2][x2] if 0 <= x2 < GRID_WIDTH and 0 <= y2 < GRID_HEIGHT else 0
        
        # Base cost from terrain
        base_cost = TERRAIN_COSTS.get(terrain2, 1.0)
        
        # Blocking water if avoid_water is on
        if self.avoid_water and terrain2 == WATER:
            return float('inf')
        
        # Elevation factor (uphill is harder)
        elev_diff = elev2 - elev1
        if elev_diff > 0:  # Going uphill
            elevation_factor = 1.0 + (elev_diff / 100.0) * self.elevation_weight
        else:  # Going downhill (easier but not free)
            elevation_factor = 1.0 + (abs(elev_diff) / 300.0) * self.elevation_weight
        
        # Scenic preference (reduced cost for scenic areas / or rest areas)
        scenic_factor = 1.0
        if terrain2 == SCENIC:
            scenic_factor = max(0.5, 1.0 - self.scenic_weight)
        
        # Final cost calculation
        return base_cost * elevation_factor * scenic_factor
    
    def hiking_heuristic(self, current, goal):
        """Heuristic function for A* optimized for hiking"""
        # Base distance (straight line)
        dx = current[0] - goal[0]
        dy = current[1] - goal[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Adjust heuristic based on distance weight
        return distance * self.distance_weight
    
    def calculate_path_metrics(self):
        """Calculate metrics for the current path"""
        if not self.current_path or len(self.current_path) < 2:
            self.path_distance = 0
            self.elevation_gain = 0
            return
        
        # Calculate distance and elevation gain
        self.path_distance = 0
        self.elevation_gain = 0
        
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            
            # Calculate distance
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            distance = math.sqrt(dx*dx + dy*dy) * 100  # Convert to meters
            self.path_distance += distance
            
            # Calculate elevation gain
            elev1 = self.elevation[p1[1]][p1[0]]
            elev2 = self.elevation[p2[1]][p2[0]]
            if elev2 > elev1:
                self.elevation_gain += elev2 - elev1
        
        # Convert distance to kilometers
        self.path_distance /= 1000
    
    def draw(self):
        """Rendering the application"""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Draw terrain
        self.draw_terrain()
        
        # Draw pathfinding visualization
        if self.show_visited:
            self.draw_visited_nodes()
        
        if self.show_path:
            self.draw_path()
        
        # Draw start and end points
        self.draw_points()
        
        # Draw UI
        self.draw_ui()
        
        # Update display
        pygame.display.flip()
    
    def draw_terrain(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
                
                if self.show_elevation:
                    # Calculate color based on elevation
                    elevation = self.elevation[y][x]
                    normalized = min(1.0, elevation / 1500)
                    
                    if normalized < 0.5:
                        # Green to brown
                        r = int(34 + (139 - 34) * normalized * 2)
                        g = int(139 + (69 - 139) * normalized * 2)
                        b = int(34 + (19 - 34) * normalized * 2)
                    else:
                        # Brown to white
                        factor = (normalized - 0.5) * 2
                        r = int(139 + (255 - 139) * factor)
                        g = int(69 + (255 - 69) * factor)
                        b = int(19 + (255 - 19) * factor)
                    
                    color = (r, g, b)
                else:
                    # Use terrain color
                    color = TERRAIN_COLORS.get(self.terrain[y][x], (100, 100, 100))
                    
                pygame.draw.rect(self.screen, color, rect)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
    
    def draw_visited_nodes(self):
        """Drawing nodes visited by A*"""
        for x, y in self.visited_nodes:
            rect = pygame.Rect(x * GRID_SIZE + 5, y * GRID_SIZE + 5, GRID_SIZE - 10, GRID_SIZE - 10)
            pygame.draw.rect(self.screen, PURPLE, rect, 1)
    
    def draw_path(self):
        """Drawing the calculated path"""
        if not self.current_path:
            return
        
        # Drawing path lines
        for i in range(len(self.current_path) - 1):
            p1 = self.current_path[i]
            p2 = self.current_path[i+1]
            
            # Calculating screen positions
            start_pos = (p1[0] * GRID_SIZE + GRID_SIZE // 2, p1[1] * GRID_SIZE + GRID_SIZE // 2)
            end_pos = (p2[0] * GRID_SIZE + GRID_SIZE // 2, p2[1] * GRID_SIZE + GRID_SIZE // 2)
            
            # Draw line
            pygame.draw.line(self.screen, YELLOW, start_pos, end_pos, 3)
        
        # Drawing points along path
        for i, (x, y) in enumerate(self.current_path):
            center = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
            
            # Different size for intermediate points
            if i == 0 or i == len(self.current_path) - 1:
                pygame.draw.circle(self.screen, YELLOW, center, 5)
            else:
                pygame.draw.circle(self.screen, YELLOW, center, 2)
    
    def draw_points(self):
        """Draw start, end, and points of interest"""
        # Draw start and end points
        if self.start_point:
            start_rect = pygame.Rect(
                self.start_point[0] * GRID_SIZE, 
                self.start_point[1] * GRID_SIZE,
                GRID_SIZE, GRID_SIZE
            )
            pygame.draw.rect(self.screen, GREEN, start_rect, 3)
        
        if self.end_point:
            end_rect = pygame.Rect(
                self.end_point[0] * GRID_SIZE, 
                self.end_point[1] * GRID_SIZE,
                GRID_SIZE, GRID_SIZE
            )
            pygame.draw.rect(self.screen, RED, end_rect, 3)
        
        # Drawing points of interest
        for x, y in self.points_of_interest:
            center = (x * GRID_SIZE + GRID_SIZE // 2, y * GRID_SIZE + GRID_SIZE // 2)
            pygame.draw.circle(self.screen, BLUE, center, GRID_SIZE // 2, 2)
    
    def draw_ui(self):
        """Drawing user interface"""
        # Drawing info box at bottom
        info_height = 80
        info_rect = pygame.Rect(0, HEIGHT - info_height, WIDTH, info_height)
        pygame.draw.rect(self.screen, (30, 30, 30), info_rect)
        
        # Drawing path metrics if path exists
        if self.current_path:
            metrics_x = 10
            metrics_y = HEIGHT - info_height + 10
            
            metrics_text = [
                f"Distance: {self.path_distance:.2f} km",
                f"Elevation Gain: {self.elevation_gain:.1f}m",
                f"Path Nodes: {len(self.current_path)}",
                f"Calculation Time: {self.calculation_time:.4f}s",
                f"Nodes Explored: {self.nodes_explored}"
            ]
            
            for i, text in enumerate(metrics_text):
                text_surf = self.font.render(text, True, WHITE)
                self.screen.blit(text_surf, (metrics_x, metrics_y + i * 20))
        
        # Drawing preference settings
        prefs_x = WIDTH - 300
        prefs_y = HEIGHT - info_height + 10
        
        prefs_text = [
            f"Distance Weight: {self.distance_weight:.1f} (Keys 1/2)",
            f"Elevation Weight: {self.elevation_weight:.1f} (Keys 3/4)",
            f"Scenic Weight: {self.scenic_weight:.1f} (Keys 5/6)",
            f"Avoid Water: {'Yes' if self.avoid_water else 'No'} (Key W)"
        ]
        
        for i, text in enumerate(prefs_text):
            text_surf = self.font.render(text, True, WHITE)
            self.screen.blit(text_surf, (prefs_x, prefs_y + i * 20))
    
    def run(self):
        """Main application loop"""
        while self.running:
            self.clock.tick(60)
            self.handle_events()
            self.draw()



# %% [markdown]
# **Run below cell to run hiking trail navigator**

# %%
if __name__ == "__main__":
    app = HikingNavigator()
    app.run()

# %%



