# Kaooa Game

Kaooa is a simple strategy game played between crows and vultures on a pentagram-shaped board. The objective is for either the crows or the vultures to outmaneuver the opponent and gain control of the board.

## Game Overview

- The game is played on a pentagram-shaped grid with nodes connected by edges.
- The two players take turns controlling crows and vultures. The crows go first.
- The goal of the crows is to strategically place and move their pieces, while the vultures aim to capture the crows.
- The game ends when either the crows or the vultures have won based on the conditions outlined in the game logic.

## How to Play

1. **Crow's Turn**:
   - The game starts with the crows.
   - In the *drop phase*, crows are placed on the board by clicking on an empty node.
   - Once all crows are placed, the game moves to the *move phase* where you can click to select a crow and move it to an adjacent empty node.
   - The crows cannot move more than one step at a time.
   - The crows' goal is to move strategically and avoid being captured.

2. **Vulture's Turn**:
   - After the crows' turn, the vultures take their turn.
   - Vultures can move to adjacent empty nodes, or if they are adjacent to a crow, they can capture it.
   - The vulture’s goal is to capture enough crows to win the game.

3. **Winning the Game**:
   - The game ends when either:
     - The vultures capture 4 or more crows, or
     - The crows have no remaining reserve and cannot make any valid moves while the vultures still can.

## Game Controls

- **Click on the nodes** to place or move pieces (crows and vultures).
- **Crow Pieces**: Crows are placed in the *drop phase* and moved during the *move phase*. A crow is selected by clicking it, and it can be moved by selecting a valid destination node.
- **Vulture Pieces**: Once a vulture is placed, it can be moved similarly to the crows. If a vulture is adjacent to a crow, it can capture it.
- **Restart the Game**: Press the "r" key to restart the game after a winner is declared.

## Features

- **Interactive UI**: Click on nodes to interact with the game pieces.
- **Crow and Vulture Pieces**: Crows are black, vultures are red, and selected crows are highlighted in green.
- **Turn-based gameplay**: Each player alternates between crow and vulture turns.
- **Victory conditions**: The game determines a winner when the conditions are met (either the crows or vultures win).

## Visuals

- The board is drawn using a **pentagram** structure, with nodes placed in a star-like configuration.
- The crows and vultures are represented as colored tokens on the board.
- The selected crow is highlighted in **green** for better visibility.

## Installation

To run this game, you will need to have Python installed, along with the **turtle graphics** library.

1. Clone or download the repository.
2. Make sure Python 3.x is installed on your system.
3. Run the game by executing the `kaooa_game.py` file:

   ```bash
   python kaooa_game.py

