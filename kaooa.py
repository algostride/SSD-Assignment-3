import turtle
import math

# Constants
NODE_RADIUS = 20
TOKEN_RADIUS = 15
WIDTH = 800
HEIGHT = 800
OUTER_RADIUS = 200  # Radius of the pentagram's outer points
TOTAL_CROWS = 5  # Number of crows to be placed at the beginning
CROW = 'crow'
VULTURE = 'vulture'

# Colors
BACKGROUND_COLOR = '#fafafa'
LINE_COLOR = '#1a237e'
NODE_COLOR = 'white'
CROW_COLOR = 'black'
VULTURE_COLOR = '#b71c1c'
HIGHLIGHT_COLOR = '#2196f3'

# Initialize game state variables
occupant = [None] * 10  # List to track node occupations (None, CROW, VULTURE)
crow_reserve = TOTAL_CROWS
vulture_pos = None
turn = CROW  # CROW starts first
drop_phase = True  # Drop phase for placing crows
nodes = []
edges = []

# Initialize Turtle screen
screen = turtle.Screen()
screen.setup(width=WIDTH, height=HEIGHT)
screen.bgcolor(BACKGROUND_COLOR)
screen.title("Kaooa")

# Turtle setup for drawing
pen = turtle.Turtle()
pen.speed(0)
pen.hideturtle()

# Helper functions
def polar_to_cartesian(radius, angle):
    """Convert polar coordinates to cartesian (x, y)."""
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    return x, y

def build_pentagram_graph():
    """Build the graph of nodes (edges) for pentagram structure."""
    global nodes, edges
    nodes = []
    edges = []
    angles = [math.radians(i * 72) for i in range(5)]  # 72 degrees for pentagon
    
    # Outer points
    for angle in angles:
        x, y = polar_to_cartesian(OUTER_RADIUS, angle)
        nodes.append((x, y))
    
    # Inner intersection points (can be computed by intersection geometry)
    inner_radius = OUTER_RADIUS * 0.5
    for i in range(5):
        x, y = polar_to_cartesian(inner_radius, math.radians(i * 72 + 36))  # 36 for offset
        nodes.append((x, y))

    # Add edges (simplified for the explanation, you'll need to connect them according to the pentagram)
    for i in range(5):
        edges.append((i, (i + 1) % 5))  # Outer edges of the pentagram
        edges.append((i + 5, (i + 1) % 5 + 5))  # Inner connections
        # And the diagonals in the pentagram that connect outer nodes to inner intersections.

def draw_board():
    """Draw the pentagram board."""
    pen.penup()
    pen.color(LINE_COLOR)
    
    # Draw edges (lines between nodes)
    for edge in edges:
        pen.goto(nodes[edge[0]])
        pen.pendown()
        pen.goto(nodes[edge[1]])
        pen.penup()
    
    # Draw nodes
    for node in nodes:
        pen.goto(node[0], node[1] - NODE_RADIUS)
        pen.pendown()
        pen.begin_fill()
        pen.circle(NODE_RADIUS)
        pen.end_fill()
        pen.penup()

def draw_tokens():
    """Draw the tokens (crows, vulture)."""
    for i, token in enumerate(occupant):
        if token == CROW:
            pen.goto(nodes[i][0], nodes[i][1] - TOKEN_RADIUS)
            pen.color(CROW_COLOR)
            pen.dot(TOKEN_RADIUS * 2)
        elif token == VULTURE:
            pen.goto(nodes[i][0], nodes[i][1] - TOKEN_RADIUS)
            pen.color(VULTURE_COLOR)
            pen.dot(TOKEN_RADIUS * 2)

def show_info(msg):
    """Display the status message."""
    pen.goto(-WIDTH // 2 + 10, HEIGHT // 2 - 20)
    pen.color('black')
    pen.clear()
    pen.write(msg, align="left", font=("Arial", 16, "normal"))

def get_neighbors(node_idx):
    """Get neighboring nodes for the given node index."""
    neighbors = []
    for edge in edges:
        if edge[0] == node_idx:
            neighbors.append(edge[1])
        elif edge[1] == node_idx:
            neighbors.append(edge[0])
    return neighbors

def node_at_point(x, y):
    """Find the node index given a click position."""
    for i, (node_x, node_y) in enumerate(nodes):
        if math.sqrt((x - node_x) ** 2 + (y - node_y) ** 2) < NODE_RADIUS * 3:
            return i
    return None

def possible_crow_moves(node_idx):
    """Return list of possible valid moves for crows from the given node."""
    valid_moves = []
    for neighbor in get_neighbors(node_idx):
        if occupant[neighbor] is None:
            valid_moves.append(neighbor)
    return valid_moves

def possible_vulture_moves(node_idx):
    """Return list of valid moves for the vulture from the current node."""
    valid_moves = []
    for neighbor in get_neighbors(node_idx):
        if occupant[neighbor] is None:  # Simple move
            valid_moves.append(neighbor)
        else:
            # Check for capture opportunities
            capture = check_vulture_capture(node_idx, neighbor)
            if capture:
                valid_moves.append(capture)
    return valid_moves

def check_vulture_capture(start_idx, crow_idx):
    """Check if the vulture can capture a crow."""
    # Calculate the capture move: the node beyond the crow in the same line
    crow_pos = nodes[crow_idx]
    start_pos = nodes[start_idx]
    for neighbor in get_neighbors(crow_idx):
        if occupant[neighbor] is None:
            # If there's an empty spot beyond the crow in the same line, return the capture destination
            return neighbor
    return None

def handle_crow_turn(node_idx):
    """Handle the crow player's turn."""
    if occupant[node_idx] == None:
        show_info("Select a crow.")
    elif occupant[node_idx] == CROW:
        valid_moves = possible_crow_moves(node_idx)
        highlight_moves(valid_moves)
        show_info("Select destination.")

def highlight_moves(valid_moves):
    """Highlight valid moves for selected token."""
    for move in valid_moves:
        pen.goto(nodes[move][0], nodes[move][1] - NODE_RADIUS)
        pen.color(HIGHLIGHT_COLOR)
        pen.dot(NODE_RADIUS * 2)

def on_click(x, y):
    """Handle clicks on the board."""
    node = node_at_point(x, y)
    if node is None:
        show_info("Click on a node.")
    elif drop_phase:
        handle_crow_turn(node)
    else:
        if turn == CROW:
            handle_crow_turn(node)
        else:
            handle_vulture_turn(node)

def handle_vulture_turn(node_idx):
    """Handle the vulture player's turn."""
    if occupant[node_idx] == VULTURE:
        valid_moves = possible_vulture_moves(node_idx)
        highlight_moves(valid_moves)
        show_info("Select move.")

def restart():
    """Restart the game."""
    global crow_reserve, vulture_pos, turn, drop_phase, occupant
    crow_reserve = TOTAL_CROWS
    vulture_pos = None
    turn = CROW
    drop_phase = True
    occupant = [None] * 10
    draw_board()
    draw_tokens()
    show_info("Crows' turn: Select a crow to place.")

# Game loop
def main():
    build_pentagram_graph()
    draw_board()
    draw_tokens()
    show_info("Crows' turn: Select a crow to place.")
    
    # Set up mouse click handler
    screen.onclick(on_click)
    screen.onkey(restart, 'r')
    screen.listen()
    
    turtle.mainloop()

# Start the game
main()

