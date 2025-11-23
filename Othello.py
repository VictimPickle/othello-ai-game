from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Callable, Dict
import math
import random
import copy

N = 8
BLACK, WHITE, EMPTY = 1, -1, 0
PLAYERS = {BLACK: "BLACK", WHITE: "WHITE"}

DIRS = [(-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)]

def opponent(p: int) -> int:
    return -p

def on_board(r: int, c: int) -> bool:
    return 0 <= r < N and 0 <= c < N

def initial_board() -> List[List[int]]:
    b = [[EMPTY for _ in range(N)] for _ in range(N)]
    b[3][3] = WHITE; b[3][4] = BLACK
    b[4][3] = BLACK; b[4][4] = WHITE
    return b

def copy_board(b):
    return [row[:] for row in b]

def _flips_in_dir(board: List[List[int]], r: int, c: int, dr: int, dc: int, player: int) -> List[Tuple[int,int]]:
    """Return the list of opponent positions that would be flipped in one direction, else []"""
    flips = []
    r += dr; c += dc
    if not on_board(r,c) or board[r][c] != opponent(player):
        return []
    while on_board(r,c) and board[r][c] == opponent(player):
        flips.append((r,c))
        r += dr; c += dc
    if not on_board(r,c) or board[r][c] != player:
        return []
    return flips

def legal_moves(board: List[List[int]], player: int) -> List[Optional[Tuple[int,int]]]:
    """Return a list of legal moves (r,c); if none exist, the only legal move is [None] (pass)."""
    moves = []
    for r in range(N):
        for c in range(N):
            if board[r][c] != EMPTY:
                continue
            flips_any = []
            for dr,dc in DIRS:
                flips_any += _flips_in_dir(board, r, c, dr, dc, player)
            if flips_any:
                moves.append((r,c))
    if not moves:
        return [None]
    return moves

def apply_move(board: List[List[int]], move: Optional[Tuple[int,int]], player: int) -> List[List[int]]:
    """Return a NEW board after applying move (or pass=None)."""
    if move is None:
        return copy_board(board)
    r, c = move
    assert board[r][c] == EMPTY, "illegal: target not empty"
    flips_all = []
    for dr,dc in DIRS:
        flips_all += _flips_in_dir(board, r, c, dr, dc, player)
    assert flips_all, "illegal: no flips created"
    nb = copy_board(board)
    nb[r][c] = player
    for fr,fc in flips_all:
        nb[fr][fc] = player
    return nb

def count_discs(board: List[List[int]]) -> Tuple[int,int,int]:
    b = sum(cell == BLACK for row in board for cell in row)
    w = sum(cell == WHITE for row in board for cell in row)
    e = N*N - b - w
    return b, w, e

@dataclass(frozen=True)
class OthelloState:
    board: Tuple[Tuple[int, ...], ...]
    player: int

    @staticmethod
    def from_board(board: List[List[int]], player: int) -> "OthelloState":
        return OthelloState(tuple(tuple(row) for row in board), player)

    def to_board(self) -> List[List[int]]:
        return [list(row) for row in self.board]

def is_terminal(state: OthelloState) -> bool:
    b = state.to_board()
    if legal_moves(b, state.player) != [None]:
        return False
    if legal_moves(b, opponent(state.player)) != [None]:
        return False
    return True

def utility(state: OthelloState, perspective: int) -> float:
    """+1 win, -1 loss, 0 draw for perspective at terminal states."""
    b, w, _ = count_discs(state.to_board())
    if b > w:
        winner = BLACK
    elif w > b:
        winner = WHITE
    else:
        return 0.0
    return 1.0 if winner == perspective else -1.0

def print_board(board: List[List[int]]):
    header = "    " + " ".join(str(c) for c in range(N))
    print(header)
    print("   " + "--"*N)
    for r in range(N):
        row = []
        for c in range(N):
            v = board[r][c]
            row.append("●" if v == BLACK else "○" if v == WHITE else ".")
        print(f"{r:>2} | " + " ".join(row))
    b,w,e = count_discs(board)
    print(f"    (●=BLACK {b}, ○=WHITE {w}, empty {e})")

def pretty_move(m):
    return "pass" if m is None else f"({m[0]},{m[1]})"

class Agent:
    def __init__(self, name="Agent"):
        self.name = name
        self.nodes = 0

    def reset_nodes(self):
        self.nodes = 0

    def action(self, state: OthelloState) -> Optional[Tuple[int,int]]:
        raise NotImplementedError

class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        super().__init__("Random")
        if seed is not None:
            random.seed(seed)

    def action(self, state: OthelloState):
        self.nodes += 1
        b = state.to_board()
        moves = legal_moves(b, state.player)
        return random.choice(moves)

class GreedyFlipAgent(Agent):
    """Pick the move that flips the most discs immediately; pass if only pass."""
    def __init__(self):
        super().__init__("GreedyFlips")

    def action(self, state: OthelloState):
        self.nodes += 1
        b = state.to_board()
        moves = legal_moves(b, state.player)
        if moves == [None]:
            return None

        best = None
        best_flips = -1
        for m in moves:
            self.nodes += 1
            r, c = m
            flips = 0
            for dr, dc in DIRS:
                flips += len(_flips_in_dir(b, r, c, dr, dc, state.player))
            if flips > best_flips:
                best_flips, best = flips, m
        return best

def play_game(black_agent: Agent, white_agent: Agent, seed: Optional[int] = None, verbose: bool = False):
    rng = random.Random(seed)
    random.seed(seed)
    board = initial_board()
    player = BLACK
    state = OthelloState.from_board(board, player)
    history = []
    if hasattr(black_agent, "reset_nodes"): black_agent.reset_nodes()
    if hasattr(white_agent, "reset_nodes"): white_agent.reset_nodes()

    if verbose:
        print_board(board)
    while not is_terminal(state):
        agent = black_agent if state.player == BLACK else white_agent
        move = agent.action(state)
        moves = legal_moves(state.to_board(), state.player)
        assert move in moves, f"{agent.name} chose illegal move {move}; legal are {moves}"
        if move is None:
            history.append((state.player, None))
            state = OthelloState.from_board(state.to_board(), opponent(state.player))
            if verbose:
                print(f"{PLAYERS[opponent(state.player)]} passes.")
            continue
        new_board = apply_move(state.to_board(), move, state.player)
        history.append((state.player, move))
        state = OthelloState.from_board(new_board, opponent(state.player))
        if verbose:
            print(f"{PLAYERS[opponent(state.player)]} played {pretty_move(move)}")
            print_board(new_board)

    b,w,e = count_discs(state.to_board())
    result = 1 if b > w else -1 if w > b else 0
    if verbose:
        print("Final:")
        print_board(state.to_board())
        if result == 0:
            print("Draw.")
        else:
            print(f"Winner: {PLAYERS[result]}")

    try:
        print(f"Black ({black_agent.name}) nodes: {black_agent.nodes}")
    except Exception:
        pass
    try:
        print(f"White ({white_agent.name}) nodes: {white_agent.nodes}")
    except Exception:
        pass

    return result, history

CORNER_CELLS = [(0,0),(0,7),(7,0),(7,7)]

def evaluate(board: List[List[int]], perspective: int) -> float:
    position = [
                [120, -25,  15,  10,  10,  15, -25, 120],
                [-25, -40,  -5,  -5,  -5,  -5, -40, -25],
                [ 15,  -5,  10,   2,   2,  10,  -5,  15],
                [ 10,  -5,   2,   1,   1,   2,  -5,  10],
                [ 10,  -5,   2,   1,   1,   2,  -5,  10],
                [ 15,  -5,  10,   2,   2,  10,  -5,  15],
                [-25, -40,  -5,  -5,  -5,  -5, -40, -25],
                [120, -25,  15,  10,  10,  15, -25, 120],
    ]

    def ratioDiff(a, b): return (a - b) / (a + b + 1e-6)

    opp_col = opponent(perspective)
    b, w, e = count_discs(board)
    my_count  = b if perspective == BLACK else w
    opp_count = w if perspective == BLACK else b

    parity = (my_count - opp_count) / 64.0
    phase = e / 64.0

    my_moves  = legal_moves(board, perspective)
    opp_moves = legal_moves(board, opp_col)
    my_mob  = 0 if my_moves  == [None] else len(my_moves)
    opp_mob = 0 if opp_moves == [None] else len(opp_moves)
    mobility = ratioDiff(my_mob, opp_mob)

    my_corners  = sum(1 for (r,c) in CORNER_CELLS if board[r][c] == perspective)
    opp_corners = sum(1 for (r,c) in CORNER_CELLS if board[r][c] == opp_col)
    corners = ratioDiff(my_corners, opp_corners)

    def frontier_count(player):
        cnt = 0
        for r in range(N):
            for c in range(N):
                if board[r][c] != player:
                    continue
                for dr, dc in DIRS:
                    rr, cc = r + dr, c + dc
                    if on_board(rr, cc) and board[rr][cc] == EMPTY:
                        cnt += 1
                        break
        return cnt
    frontier = ratioDiff(frontier_count(opp_col), frontier_count(perspective))

    pos_score = 0
    for r in range(N):
        for c in range(N):
            if board[r][c] == perspective:
                pos_score += position[r][c]
            elif board[r][c] == opp_col:
                pos_score -= position[r][c]
    pos_score /= 1000.0

    def w(we, wl):
        return we * phase + wl * (1 - phase)

    score = (
        w(0.35, 0.10) * pos_score +
        w(0.30, 0.10) * mobility  +
        w(0.80, 0.60) * corners   +
        w(0.20, 0.10) * frontier  +
        w(0.05, 0.90) * parity
    )
    return math.tanh(score)

class MinimaxAgent(Agent):
    def __init__(self, depth: int = 4, eval_fn: Callable[[List[List[int]], int], float] = evaluate):
        self.depth = depth
        self.eval_fn = eval_fn
        self.name = f"Minimax(d={depth})"
        self.nodes = 0

    def action(self, state: OthelloState) -> Optional[Tuple[int,int]]:
        perspective = state.player
        board = state.to_board()
        moves = legal_moves(board, state.player)
        if moves == [None]:
            return None
        
        best_move, best_val = None, -math.inf
        for move in moves:
            cboard = apply_move(board, move, state.player)
            cstate = OthelloState.from_board(cboard, opponent(state.player))
            val = self._min_value(cstate, self.depth - 1, perspective)
            if val > best_val:
                best_val, best_move = val, move

        return best_move

    def _max_value(self, state: OthelloState, depth: int, perspective: int) -> float:
        self.nodes += 1
        if is_terminal(state):
            return utility(state, perspective)
        
        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)
        
        b = state.to_board()
        moves = legal_moves(b, state.player)
        
        if moves == [None]:
            passed = OthelloState.from_board(b, opponent(state.player))
            return self._min_value(passed, depth - 1, perspective)

        value = -math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            value = max(value, self._min_value(child, depth - 1, perspective))
        return value

    def _min_value(self, state: OthelloState, depth: int, perspective: int) -> float:
        self.nodes += 1
        if is_terminal(state):
            return utility(state, perspective)
        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)

        b = state.to_board()
        moves = legal_moves(b, state.player)

        if moves == [None]:
            passed = OthelloState.from_board(b, opponent(state.player))
            return self._max_value(passed, depth - 1, perspective)
            
        value = math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            value = min(value, self._max_value(child, depth - 1, perspective))

        return value

class AlphaBetaAgent(Agent):
    def __init__(self, depth: int = 5, eval_fn: Callable[[List[List[int]], int], float] = evaluate):
        self.depth = depth
        self.eval_fn = eval_fn
        self.name = f"AlphaBeta(d={depth})"
        self.nodes = 0

    def action(self, state: OthelloState) -> Optional[Tuple[int,int]]:
        perspective = state.player
        b = state.to_board()
        moves = legal_moves(b, state.player)
        if moves == [None]:
            return None

        def order_key(m):
            if m is None:
                return -1e9
            nb = apply_move(b, m, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            return self.eval_fn(child.to_board(), perspective)
        moves.sort(key=order_key, reverse=True)

        alpha, beta = -math.inf, math.inf
        best_move, best_val = None, -math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            val = self._min_value(child, self.depth - 1, perspective, alpha, beta)
            if val > best_val:
                best_val, best_move = val, move
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
        return best_move

    def _max_value(self, state: OthelloState, depth: int, perspective: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if is_terminal(state):
            return utility(state, perspective)
        
        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)

        b = state.to_board()
        moves = legal_moves(b, state.player)

        if moves == [None]:
            passed = OthelloState.from_board(b, opponent(state.player))
            return self._min_value(passed, depth - 1, perspective, alpha, beta)

        def order_key(m):
            if m is None:
                return -1e9
            nb = apply_move(b, m, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            return self.eval_fn(child.to_board(), perspective)
            
        moves.sort(key=order_key, reverse=True)

        value = -math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            value = max(value, self._min_value(child, depth - 1, perspective, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        return value

    def _min_value(self, state: OthelloState, depth: int, perspective: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if is_terminal(state):
            return utility(state, perspective)
        
        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)

        b = state.to_board()
        moves = legal_moves(b, state.player)

        if moves == [None]:
            passed = OthelloState.from_board(b, opponent(state.player))
            return self._max_value(passed, depth - 1, perspective, alpha, beta)
        
        def order_key(m):
            if m is None:
                return 1e9
            nb = apply_move(b, m, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            return self.eval_fn(child.to_board(), perspective)
        moves.sort(key=order_key, reverse=False)

        value = math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            value = min(value, self._max_value(child, depth - 1, perspective, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
            
        return value

class ExpectimaxRiskyAgent(Agent):
    def __init__(self,
                 depth: int = 4,
                 eval_fn: Callable[[List[List[int]], int], float] = evaluate,
                 blunder_rate: float = 0.20):
        self.depth = depth
        self.eval_fn = eval_fn
        self.eps = max(0.0, min(1.0, blunder_rate))
        self.name = f"ExpectimaxRisky(d={depth}, eps={self.eps:.2f})"
        self.nodes = 0

    def action(self, state: OthelloState) -> Optional[Tuple[int,int]]:
        self.nodes += 1

        board = state.to_board()
        moves = legal_moves(board, state.player)
        if moves == [None]:
            return None

        perspective = state.player
        best_move, best_val = None, -math.inf
        for move in moves:
            nb = apply_move(board, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            val = self._opponent_expectation(child, self.depth - 1, perspective)
            if val > best_val:
                best_val, best_move = val, move
        return best_move

    def _max_value(self, state: OthelloState, depth: int, perspective: int) -> float:
        if is_terminal(state):
            return utility(state, perspective)
        self.nodes += 1

        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)

        b = state.to_board()
        moves = legal_moves(b, state.player)

        if moves == [None]:
            return self._opponent_expectation(OthelloState.from_board(b, opponent(state.player)), depth - 1, perspective)

        value = -math.inf
        for move in moves:
            nb = apply_move(b, move, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            value = max(value, self._opponent_expectation(child, depth - 1, perspective))
        return value

    def _opponent_expectation(self, state: OthelloState, depth: int, perspective: int) -> float:
        if is_terminal(state):
            return utility(state, perspective)
        self.nodes += 1
        
        if depth <= 0:
            return self.eval_fn(state.to_board(), perspective)

        b = state.to_board()
        moves = legal_moves(b, state.player)

        if moves == [None]:
            return self._max_value(
                OthelloState.from_board(b, opponent(state.player)),
                depth - 1,
                perspective
            )

        vals = []
        for m in moves:
            nb = apply_move(b, m, state.player)
            child = OthelloState.from_board(nb, opponent(state.player))
            v = self._max_value(child, depth - 1, perspective)
            vals.append(v)
        vals.sort()
        vbest = vals[0]

        if len(vals) == 1 or self.eps <= 1e-12:
            return vbest

        other = vals[1:]
        k = len(other)
        if k == 0:
            return vbest
        
        exp_val = (1.0 - self.eps) * vbest + (self.eps * sum(other) / k)
        return exp_val

# Basic engine checks
b = initial_board()
print_board(b)

lm_black = legal_moves(b, BLACK)
lm_white = legal_moves(b, WHITE)
assert len([m for m in lm_black if m is not None]) == 4
assert len([m for m in lm_white if m is not None]) == 4

assert (2,3) in lm_black
b2 = apply_move(b, (2,3), BLACK)
print_board(b2)

print("✓ Basic engine checks passed.")

# GUI Setup
import sys
import subprocess

def install_and_setup_widgets():
    """Install and enable ipywidgets for Jupyter"""
    print("Checking ipywidgets installation...")
    
    try:
        import ipywidgets as widgets
        print("✓ ipywidgets is already installed")
    except ImportError:
        print("Installing ipywidgets...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets", "-q"])
        print("✓ ipywidgets installed")
    
    import platform
    if platform.system() == "Windows":
        print("\nEnabling Jupyter widget extensions for Windows...")
        try:
            subprocess.check_call([sys.executable, "-m", "jupyter", "nbextension", "enable", "--py", "widgetsnbextension", "--sys-prefix"], 
                                 stderr=subprocess.DEVNULL)
            print("✓ Extension enabled")
        except:
            print("Note: Extension enable commands may need manual execution")
            print("Run: jupyter nbextension enable --py widgetsnbextension --sys-prefix")
    
    print("\n✓ Setup complete! You may need to refresh your browser if widgets don't appear.")
    print("On Windows: If the GUI still doesn't show, try restarting Jupyter notebook.")

install_and_setup_widgets()

# GUI Implementation
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
    WIDGETS_AVAILABLE = True
except ImportError as e:
    print("ERROR: ipywidgets is not installed or not working.")
    print("Please run the setup cell above to install it.")
    print(f"Error details: {e}")
    print("\nIf you're on Windows, you may need to run these commands in terminal:")
    print("  pip install ipywidgets")
    print("  jupyter nbextension enable --py widgetsnbextension --sys-prefix")
    WIDGETS_AVAILABLE = False

if not WIDGETS_AVAILABLE:
    raise ImportError("Cannot start GUI without ipywidgets")

class OthelloGUI:
    def __init__(self):
        self.state = None
        self.agent = None
        self.human_color = None
        self.game_active = False
        self.output = widgets.Output()
        self.last_ai_move = None
        
        self.cell_size = 60  
        self.setup_menu()
    
    def setup_menu(self):
        """Setup the agent selection menu"""
        clear_output(wait=True)
        title = widgets.HTML("<h2>🎮 Othello: Human vs AI</h2>")
        
        agent_dropdown = widgets.Dropdown(
            options=[
                ('Random', 'random'),
                ('Greedy Flips', 'greedy'),
                ('Minimax (depth=3)', 'minimax_3'),
                ('Minimax (depth=4)', 'minimax_4'),
                ('Alpha-Beta (depth=3)', 'alphabeta_3'),
                ('Alpha-Beta (depth=4)', 'alphabeta_4'),
                ('Alpha-Beta (depth=5)', 'alphabeta_5'),
                ('Expectimax (depth=3, ε=0.2)', 'expectimax_3'),
                ('Expectimax (depth=4, ε=0.2)', 'expectimax_4'),
            ],
            value='alphabeta_4',
            description='AI Agent:',
            style={'description_width': '100px'}
        )
        
        color_dropdown = widgets.Dropdown(
            options=[('Black (●) - Go First', BLACK), ('White (○) - Go Second', WHITE)],
            value=BLACK,
            description='Your Color:',
            style={'description_width': '100px'}
        )
        
        start_button = widgets.Button(
            description='Start Game',
            button_style='success',
            icon='play'
        )
        
        def on_start_clicked(b):
            agent_type = agent_dropdown.value
            if agent_type == 'random':
                self.agent = RandomAgent()
            elif agent_type == 'greedy':
                self.agent = GreedyFlipAgent()
            elif agent_type == 'minimax_3':
                self.agent = MinimaxAgent(depth=3)
            elif agent_type == 'minimax_4':
                self.agent = MinimaxAgent(depth=4)
            elif agent_type == 'alphabeta_3':
                self.agent = AlphaBetaAgent(depth=3)
            elif agent_type == 'alphabeta_4':
                self.agent = AlphaBetaAgent(depth=4)
            elif agent_type == 'alphabeta_5':
                self.agent = AlphaBetaAgent(depth=5)
            elif agent_type == 'expectimax_3':
                self.agent = ExpectimaxRiskyAgent(depth=3, blunder_rate=0.2)
            elif agent_type == 'expectimax_4':
                self.agent = ExpectimaxRiskyAgent(depth=4, blunder_rate=0.2)
            
            self.human_color = color_dropdown.value
            self.start_game()
        
        start_button.on_click(on_start_clicked)
        
        menu_box = widgets.VBox([
            title,
            agent_dropdown,
            color_dropdown,
            start_button
        ])
        
        display(menu_box)
    
    def start_game(self):
        """Start a new game"""
        clear_output(wait=True)
        
        board = initial_board()
        self.state = OthelloState.from_board(board, BLACK)
        self.game_active = True
        self.last_ai_move = None
        
        self.create_game_ui()
        
        if self.state.player != self.human_color:
            self.make_ai_move()
    
    def create_game_ui(self):
        """Create the game board UI"""
        self.info_label = widgets.HTML()
        
        self.buttons = []
        rows = []
        
        display(HTML("""
        <style>
        .othello-cell button {
            border-radius: 50% !important;
            font-size: 24px !important;
            font-weight: bold !important;
            color: white !important;
            padding: 0 !important;
            transition: all 0.2s ease;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .othello-cell button:hover:not(:disabled) {
            transform: scale(1.15);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4), inset 0 2px 4px rgba(0,0,0,0.2);
            cursor: pointer;
        }
        </style>
        """))
        
        for r in range(N):
            row = []
            for c in range(N):
                btn = widgets.Button(
                    description='',
                    layout=widgets.Layout(
                        width=f'{self.cell_size}px', 
                        height=f'{self.cell_size}px',
                        margin='2px'
                    )
                )
                btn.add_class('othello-cell')
                btn.on_click(lambda b, row=r, col=c: self.on_cell_click(row, col))
                row.append(btn)
            rows.append(widgets.HBox(row, layout=widgets.Layout(margin='0')))
            self.buttons.append(row)
        
        board_box = widgets.VBox(
            rows, 
            layout=widgets.Layout(
                margin='10px 0px',
                padding='15px',
                background='#1b5e20',
                border_radius='10px'
            )
        )
        
        self.pass_button = widgets.Button(
            description='Pass',
            button_style='warning',
            icon='forward',
            layout=widgets.Layout(width='150px')
        )
        self.pass_button.on_click(lambda b: self.on_pass())
        
        new_game_button = widgets.Button(
            description='New Game',
            button_style='info',
            icon='refresh',
            layout=widgets.Layout(width='150px')
        )
        new_game_button.on_click(lambda b: self.setup_menu())
        
        self.status_output = widgets.Output()
        
        controls = widgets.HBox([self.pass_button, new_game_button])
        game_ui = widgets.VBox([
            self.info_label,
            board_box,
            controls,
            self.status_output
        ])
        
        display(game_ui)
        self.update_display()
    
    def update_display(self):
        """Update the board display"""
        board = self.state.to_board()
        legal = legal_moves(board, self.state.player) if self.game_active else []
        
        for r in range(N):
            for c in range(N):
                btn = self.buttons[r][c]
                cell = board[r][c]
                
                btn.description = ''
                btn.disabled = True
                
                is_last_move = self.last_ai_move and (r, c) == self.last_ai_move
                
                if cell == BLACK:
                    if is_last_move:
                        btn.style.button_color = '#000000'
                        btn.layout.border = '3px solid #FFA500'
                    else:
                        btn.style.button_color = '#000000'
                        btn.layout.border = '2px solid #888'
                    btn.disabled = True
                    
                elif cell == WHITE:
                    if is_last_move:
                        btn.style.button_color = '#F5F5DC'
                        btn.layout.border = '3px solid #FFA500'
                    else:
                        btn.style.button_color = '#F5F5DC'
                        btn.layout.border = '2px solid #888'
                    btn.disabled = True
                    
                elif self.game_active and self.state.player == self.human_color and (r, c) in legal:
                    btn.description = '+'
                    btn.style.button_color = '#28a745'
                    btn.layout.border = '2px solid #1e7e34'
                    btn.disabled = False
                    
                else:
                    btn.style.button_color = '#2d6a4f'
                    btn.layout.border = '2px solid #1b4332'
                    btn.disabled = True
        
        b, w, e = count_discs(board)
        info_html = f"""
        <div style='font-size: 18px; margin-bottom: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;'>
            <b>⚫ Black:</b> {b} | <b>⚪ White:</b> {w} | <b>Empty:</b> {e}<br>
        """
        
        if self.game_active:
            current_player = "⚫ Black" if self.state.player == BLACK else "⚪ White"
            if self.state.player == self.human_color:
                info_html += f"<b style='color: green;'>🎯 Your turn ({current_player})</b><br>"
                if self.last_ai_move:
                    info_html += f"<small style='color: #ff8800;'>🔶 Orange = AI's last move at ({self.last_ai_move[0]}, {self.last_ai_move[1]})</small>"
            else:
                info_html += f"<b style='color: blue;'>🤖 AI's turn ({current_player})</b>"
        else:
            if b > w:
                winner = "⚫ Black"
            elif w > b:
                winner = "⚪ White"
            else:
                winner = "🤝 Draw"
            info_html += f"<b style='color: red;'>🏁 Game Over! Winner: {winner}</b>"
        
        info_html += "</div>"
        self.info_label.value = info_html
        
        if self.game_active and self.state.player == self.human_color and legal == [None]:
            self.pass_button.disabled = False
        else:
            self.pass_button.disabled = True
    
    def on_cell_click(self, row, col):
        """Handle cell click"""
        if not self.game_active or self.state.player != self.human_color:
            return
        
        board = self.state.to_board()
        legal = legal_moves(board, self.state.player)
        
        if (row, col) not in legal:
            return
        
        new_board = apply_move(board, (row, col), self.state.player)
        self.state = OthelloState.from_board(new_board, opponent(self.state.player))
        
        self.last_ai_move = None
        
        with self.status_output:
            clear_output(wait=True)
            print(f"You played ({row}, {col})")
        
        self.update_display()
        
        if is_terminal(self.state):
            self.end_game()
            return
        
        if self.state.player != self.human_color:
            self.make_ai_move()
    
    def on_pass(self):
        """Handle pass button"""
        if not self.game_active or self.state.player != self.human_color:
            return
        
        board = self.state.to_board()
        legal = legal_moves(board, self.state.player)
        
        if legal != [None]:
            return
        
        self.state = OthelloState.from_board(board, opponent(self.state.player))
        
        self.last_ai_move = None
        
        with self.status_output:
            clear_output(wait=True)
            print("You passed.")
        
        self.update_display()
        
        if is_terminal(self.state):
            self.end_game()
            return
        
        if self.state.player != self.human_color:
            self.make_ai_move()
    
    def make_ai_move(self):
        """Make AI move"""
        import time
        
        time.sleep(0.3)
        
        board = self.state.to_board()
        legal = legal_moves(board, self.state.player)
        
        if legal == [None]:
            self.state = OthelloState.from_board(board, opponent(self.state.player))
            self.last_ai_move = None
            with self.status_output:
                clear_output(wait=True)
                print(f"AI ({self.agent.name}) passed.")
            self.update_display()
            
            if is_terminal(self.state):
                self.end_game()
            return
        
        move = self.agent.action(self.state)
        
        if move is not None:
            new_board = apply_move(board, move, self.state.player)
            self.state = OthelloState.from_board(new_board, opponent(self.state.player))
            
            self.last_ai_move = move
            
            with self.status_output:
                clear_output(wait=True)
                print(f"AI ({self.agent.name}) played ({move[0]}, {move[1]})")
        
        self.update_display()
        
        if is_terminal(self.state):
            self.end_game()
    
    def end_game(self):
        """End the game"""
        self.game_active = False
        self.update_display()
        
        b, w, e = count_discs(self.state.to_board())
        
        with self.status_output:
            clear_output(wait=True)
            if b > w:
                winner = "Black"
                if self.human_color == BLACK:
                    print(f"🎉 You won! Black: {b}, White: {w}")
                else:
                    print(f"😞 AI won! Black: {b}, White: {w}")
            elif w > b:
                winner = "White"
                if self.human_color == WHITE:
                    print(f"🎉 You won! White: {w}, Black: {b}")
                else:
                    print(f"😞 AI won! White: {w}, Black: {b}")
            else:
                print(f"🤝 Draw! Black: {b}, White: {w}")

# Start the GUI
gui = OthelloGUI()
