import turtle
import math
from collections import defaultdict

# ---------- CONFIG ----------
NODE_RADIUS = 12
TOKEN_RADIUS = 16
WIDTH, HEIGHT = 800, 800
OUTER_RADIUS = 320
CENTER = (0, 0)

COLORS = {
    "line": "#1a237e",  # deep blue
    "bg": "#fafafa",
    "crow": "black",
    "vulture": "#b71c1c",
    "highlight": "#2196f3",  # blue for possible moves
    "node": "#ffffff",
    "border": "#000000",
}

TOTAL_CROWS = 7
CROW, VULTURE = "crow", "vulture"

# ---------- GEOMETRY ----------
def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-9:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    def between(a,b,c): return min(a,b)-1e-6 <= c <= max(a,b)+1e-6
    if (between(x1,x2,px) and between(y1,y2,py) and
        between(x3,x4,px) and between(y3,y4,py)):
        return (px, py)
    return None

def dist(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
def approx_equal(a,b,eps=1e-6): return dist(a,b) < eps

# ---------- BUILD STAR ----------
def build_pentagram_graph():
    outer = []
    for i in range(5):
        ang = math.radians(-90 + i*72)
        outer.append((OUTER_RADIUS*math.cos(ang), OUTER_RADIUS*math.sin(ang)))
    segs = [(outer[i], outer[(i+2)%5]) for i in range(5)]
    pts = list(outer)
    for i in range(len(segs)):
        for j in range(i+1,len(segs)):
            p = line_intersection(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            if p and not any(approx_equal(p,q) for q in pts): pts.append(p)
    unique = []
    for p in pts:
        if not any(approx_equal(p,q) for q in unique):
            unique.append(p)
    pts = unique
    adj = defaultdict(set)
    def onseg(p,a,b):
        return abs((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) < 1e-6 \
               and min(a[0],b[0])-1e-6<=p[0]<=max(a[0],b[0])+1e-6 \
               and min(a[1],b[1])-1e-6<=p[1]<=max(a[1],b[1])+1e-6
    for a,b in segs:
        pts_on = [p for p in pts if onseg(p,a,b)]
        pts_on.sort(key=lambda p: dist(a,p))
        for i in range(len(pts_on)-1):
            p1,p2=pts_on[i],pts_on[i+1]
            adj[p1].add(p2); adj[p2].add(p1)
    cx=sum(p[0] for p in pts)/len(pts)
    cy=sum(p[1] for p in pts)/len(pts)
    pts.sort(key=lambda p: math.atan2(p[1]-cy,p[0]-cx))
    idx={p:i for i,p in enumerate(pts)}
    graph={i:set() for i in range(len(pts))}
    for p,n in adj.items():
        for q in n: graph[idx[p]].add(idx[q])
    return pts,graph

# ---------- GAME STATE ----------
coords, graph = build_pentagram_graph()
occupant = {i:None for i in range(len(coords))}
crow_reserve = TOTAL_CROWS
vulture_pos = None
turn = CROW
drop_phase = True
selected_node = None

# ---------- TURTLE SETUP ----------
screen = turtle.Screen()
screen.tracer(False)
screen.setup(WIDTH,HEIGHT)
screen.bgcolor(COLORS["bg"])
drawer = turtle.Turtle(visible=False)
drawer.speed(0)
drawer.pensize(3)

token_t = turtle.Turtle(visible=False)
token_t.speed(0)
token_t.penup()

info_t = turtle.Turtle(visible=False)
info_t.penup()
info_t.goto(-WIDTH//2+10,HEIGHT//2-30)

# ---------- DRAWING ----------
def draw_static_board():
    drawer.clear()
    drawer.color(COLORS["line"])
    for u in graph:
        for v in graph[u]:
            if u<v:
                drawer.penup(); drawer.goto(coords[u]); drawer.pendown()
                drawer.goto(coords[v])
    for p in coords:
        drawer.penup()
        drawer.goto(p[0],p[1]-NODE_RADIUS)
        drawer.pendown()
        drawer.fillcolor(COLORS["node"])
        drawer.begin_fill(); drawer.circle(NODE_RADIUS); drawer.end_fill()
        drawer.penup()
    screen.update()

def draw_tokens(highlights=None):
    token_t.clear()
    for i, p in enumerate(coords):
        occ = occupant[i]
        # Draw crows with special color if selected
        if occ == CROW:
            if i == selected_node:
                token_t.goto(p[0], p[1] - TOKEN_RADIUS / 2)
                token_t.dot(TOKEN_RADIUS, "green")  # Use green for selected crow
            else:
                token_t.goto(p[0], p[1] - TOKEN_RADIUS / 2)
                token_t.dot(TOKEN_RADIUS, COLORS["crow"])
        elif occ == VULTURE:
            token_t.goto(p)
            token_t.shape("circle")
            token_t.setheading(0)
            token_t.fillcolor(COLORS["vulture"])
            token_t.begin_fill()
            token_t.goto(p[0], p[1])
            token_t.dot(TOKEN_RADIUS, COLORS["vulture"])
            token_t.end_fill()
    
    # Draw highlights for possible moves
    if highlights:
        for h in highlights:
            x, y = coords[h]
            token_t.goto(x, y - TOKEN_RADIUS / 2)
            token_t.dot(TOKEN_RADIUS + 6, COLORS["highlight"])
    
    screen.update()

def show_info(msg):
    info_t.clear()
    info_t.write(msg,font=("Arial",14,"normal"))

# ---------- LOGIC ----------
def dist(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

def node_at_point(x,y):
    for i,p in enumerate(coords):
        if dist((x,y),p)<NODE_RADIUS*3:
            return i
    return None

def possible_crow_moves(node):
    return [n for n in graph[node] if occupant[n] is None]

def possible_vulture_moves(node):
    moves=[]
    for nb in graph[node]:
        if occupant[nb] is None:
            moves.append((nb,False,None))
        elif occupant[nb]==CROW:
            ax,ay=coords[node];bx,by=coords[nb];vx,vy=bx-ax,by-ay
            for nb2 in graph[nb]:
                if nb2==node: continue
                cx,cy=coords[nb2];wx,wy=cx-bx,cy-by
                if abs(vx*wy-vy*wx)<1e-6 and (vx*wx+vy*wy)>0 and occupant[nb2] is None:
                    moves.append((nb2,True,nb))
    return moves

def any_vulture_capture():
    if vulture_pos is None: return False
    return any(cap for _,cap,_ in possible_vulture_moves(vulture_pos))

def check_winner():
    captured = TOTAL_CROWS - (sum(1 for v in occupant.values() if v==CROW)+crow_reserve)
    if captured>=4: return VULTURE
    if crow_reserve==0 and not possible_vulture_moves(vulture_pos): return CROW
    return None

# ---------- EVENTS ----------
def on_click(x, y):
    global selected_node, turn, drop_phase, crow_reserve, vulture_pos
    node = node_at_point(x, y)
    if node is None:
        return

    # CROW TURN
    if turn == CROW:
        if drop_phase:
            if occupant[node]: 
                return show_info("Occupied spot.")
            occupant[node] = CROW
            crow_reserve -= 1
            if crow_reserve == 0:
                drop_phase = False
            turn = VULTURE
            show_info("Vulture's turn.")
            draw_tokens()
        else:
            if selected_node is None:
                if occupant[node] == CROW:
                    moves = possible_crow_moves(node)
                    if not moves:
                        return show_info("No moves.")
                    selected_node = node
                    draw_tokens(highlights=moves)
                    show_info("Select destination.")
                else:
                    show_info("Select a crow.")
            else:
                if selected_node == node:  # If the selected crow is clicked again, deselect it.
                    selected_node = None
                    draw_tokens()
                    show_info("Crow deselected.")
                elif node in possible_crow_moves(selected_node):  # Move the selected crow.
                    occupant[node] = CROW
                    occupant[selected_node] = None
                    selected_node = None
                    turn = VULTURE
                    show_info("Vulture's turn.")
                    draw_tokens()
                else:
                    show_info("Invalid move.")
        winner = check_winner()
        if winner:
            end_game(winner)

    # VULTURE TURN
    else:
        if vulture_pos is None:
            if occupant[node]:
                return show_info("Occupied.")
            occupant[node] = VULTURE
            vulture_pos = node
            turn = CROW
            show_info("Crow's turn.")
            draw_tokens()
            return
        moves = possible_vulture_moves(vulture_pos)
        captures = [m for m in moves if m[1]]
        must_capture = any_vulture_capture()
        if must_capture:
            if not any(node == m[0] for m in captures):
                draw_tokens(highlights=[m[0] for m in captures])
                return show_info("Must capture!")
            dest, _, cap = next(m for m in captures if m[0] == node)
            occupant[cap] = None
        elif node not in [m[0] for m in moves]:
            draw_tokens(highlights=[m[0] for m in moves])
            return show_info("Invalid move.")
        occupant[vulture_pos] = None
        occupant[node] = VULTURE
        vulture_pos = node
        turn = CROW
        show_info("Crow's turn.")
        draw_tokens()
        winner = check_winner()
        if winner:
            end_game(winner)

def end_game(winner):
    msg="VULTURE wins!" if winner==VULTURE else "CROWS win!"
    show_info(msg+"  (Press 'r' to restart)")
    screen.onscreenclick(None)

def restart():
    global occupant,crow_reserve,vulture_pos,turn,drop_phase,selected_node
    for k in occupant: occupant[k]=None
    crow_reserve=TOTAL_CROWS; vulture_pos=None; turn=CROW; drop_phase=True; selected_node=None
    draw_tokens(); show_info("New game. Crow starts.")
    screen.onscreenclick(on_click)

# ---------- INIT ----------
draw_static_board()
draw_tokens()
show_info("Kaooa — Crows go first: place a crow.")
screen.onscreenclick(on_click)
screen.onkey(restart,"r")
screen.listen()
screen.mainloop()
