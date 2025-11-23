# Othello AI Game

A complete implementation of **Othello (Reversi)** with multiple AI agents using game theory algorithms.

## Features

- 🎮 **Interactive GUI** - Play against AI using Jupyter widgets
- 🤖 **Multiple AI Strategies**:
  - **Minimax** - Classic game tree search
  - **Alpha-Beta Pruning** - Optimized minimax with pruning
  - **Expectimax** - Risk-aware agent for imperfect opponents
  - **Greedy** - Immediate flip maximization
  - **Random** - Baseline random player

## Game Rules

- **Black (●)** plays first
- Place a disc to flank opponent discs in any direction
- Flanked discs flip to your color
- If no legal moves exist, you must pass
- Game ends when both players must pass
- Winner: player with more discs

## AI Algorithms

### 1. Minimax Agent
- Searches to a fixed depth
- Uses evaluation function at leaf nodes
- Guarantees optimal play against optimal opponent

### 2. Alpha-Beta Agent
- Minimax with pruning
- Significantly reduces nodes explored
- Move ordering improves pruning efficiency

### 3. Expectimax Agent
- Models imperfect opponents
- Accounts for opponent blunders (ε probability)
- Risk-aware decision making

## Evaluation Function

The AI uses a sophisticated evaluation function considering:
- **Position weights** - Strategic board positions (corners, edges)
- **Mobility** - Number of legal moves available
- **Corners** - Control of corner squares
- **Frontier** - Exposed discs adjacent to empty squares
- **Parity** - Disc count difference
- **Phase-based weighting** - Early game vs endgame strategy

## Installation

### Requirements
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

### Windows Users
```bash
pip install ipywidgets
jupyter nbextension install --py --user widgetsnbextension
jupyter nbextension enable --py --user widgetsnbextension
```

## Usage

### 1. Run Jupyter Notebook
```bash
jupyter notebook ex3_othello.ipynb
```

### 2. Play Against AI
Run the last cell to launch the interactive GUI:
- Choose your color (Black/White)
- Select AI difficulty and algorithm
- Click cells to make moves
- Orange border shows AI's last move

### 3. Run Agent vs Agent
Uncomment sections in the code to watch AI agents battle:
```python
# Example: AlphaBeta vs Random
ab_agent = AlphaBetaAgent(depth=4)
result, history = play_game(RandomAgent(), ab_agent, seed=0, verbose=True)
```

## Performance Comparison

| Algorithm | Depth | Nodes Explored | Speed |
|-----------|-------|----------------|-------|
| Minimax | 4 | ~50,000 | Slow |
| Alpha-Beta | 4 | ~11,000 | Fast |
| Expectimax | 3 | ~35,000 | Medium |

*Alpha-Beta achieves ~78% node reduction compared to Minimax*

## File Structure

```
.
├── ex3_othello.ipynb    # Main Jupyter notebook
├── README.md            # This file
├── requirements.txt     # Python dependencies
└── .gitignore          # Git ignore rules
```

## Implementation Highlights

- **Immutable state representation** using frozen dataclasses
- **Efficient move generation** with direction-based flipping
- **Move ordering** in Alpha-Beta for better pruning
- **Phase-aware evaluation** adapting strategy to game stage
- **Interactive visualization** with Jupyter widgets

## Learning Objectives

This project demonstrates:
- Game state modeling (states, actions, transitions)
- Adversarial search algorithms
- Evaluation function design
- Alpha-Beta pruning optimization
- Expectimax for stochastic opponents

## Future Enhancements

- [ ] Opening book for early game
- [ ] Endgame perfect solver
- [ ] Iterative deepening with time limits
- [ ] Transposition tables for state caching
- [ ] Monte Carlo Tree Search (MCTS) agent
- [ ] Web-based interface (Flask/Django)

## Author

Computer Science Student | AI & Game Theory Enthusiast

## License

MIT License - Feel free to use for educational purposes

## Acknowledgments

- Othello game rules from World Othello Federation
- AI algorithms based on classic game theory literature