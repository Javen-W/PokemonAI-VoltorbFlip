# Pokémon Voltorb Flip AI: Computer Vision and Heuristic Decision-Making

This repository contains my course project for CSE803: Computer Vision, where we developed an AI to play the Pokémon HeartGold/SoulSilver minigame Voltorb Flip using convolutional neural networks (CNNs), memory mapping, and heuristic decision-making. The project involved debugging Nintendo DS (NDS) game memory addresses, writing a Lua script for the BizHawk-2.9.1 emulator to control gameplay, generating 17,000+ training screenshots with corresponding game state data, and training CNN models to predict game states and make tile selection decisions. The AI achieved a high score of 1851 coins, reaching level 7 in evaluation runs, demonstrating robust performance in a stochastic environment.

## Table of Contents
- [Pokémon Voltorb Flip AI: Computer Vision and Heuristic Decision-Making](#pokémon-voltorb-flip-ai-computer-vision-and-heuristic-decision-making)
  - [Project Overview](#project-overview)
  - [Approach](#approach)
  - [Tools and Technologies](#tools-and-technologies)
  - [Results](#results)
  - [Skills Demonstrated](#skills-demonstrated)
  - [Installation](#installation)
  - [References](#references)

## Project Overview
Voltorb Flip is a minigame in Pokémon HeartGold/SoulSilver where players flip tiles on a 5x5 grid to reveal coins (values 1–3) or bombs (value 4), aiming to collect all coin tiles without hitting a bomb. Each row and column has a fixed number of coins and bombs, displayed as hints. The goal is to maximize cumulative coins across levels, with a single bomb ending the game. I developed an AI that:
- Interfaces with the BizHawk-2.9.1 emulator via a Lua script to control gameplay and extract game states.
- Generates training data (17,000+ screenshots and CSV game states) by automatically playing randomly seeded levels.
- Uses two pretrained CNN models to predict visible (e.g., coin/bomb counts, revealed tiles) and hidden (e.g., unrevealed tile values) game states from screenshots.
- Applies heuristics to select the next best tile, avoiding bombs and maximizing coin collection.
- Evaluates performance by cumulative coins, achieving 1851 coins and reaching level 7 in a single run.

![2024-12-07_19-42-31](https://github.com/user-attachments/assets/d2184589-8d8b-48ce-9864-25e89df00fa6)

![2024-12-07_19-42-54](https://github.com/user-attachments/assets/e14b3963-7373-48c1-b950-383434b27d20)

![2024-12-07_19-43-05](https://github.com/user-attachments/assets/8ea1f568-d59b-4dc5-8305-9cb94f96d9dc)

## Approach
The project was implemented in three main components: emulator interfacing, data generation, and AI model development/evaluation.

- **Emulator Interfacing and Memory Mapping** (`eval_client.lua`):
  - Developed a Lua script for BizHawk-2.9.1 to control the NDS emulator, load savestates, and read/write game memory.
  - Debugged memory addresses (e.g., `0x2E5EC4` for tiles, `0x27C31C` for coins) to extract game states, including tile values (0=hidden, 1–3=coins, 4=bomb), coin/bomb counts, and dialogue states.
  - Implemented functions like `read_tiles()`, `read_coin_counts()`, and `select_tile()` to interact with the game, advance dialogues, and randomize seeds (`0x10F6CC`).
  - Supported two modes:
    - **Training Mode**: Automatically plays levels, revealing safe tiles (1–3) and logging states/screenshots.
    - **Evaluation Mode**: Manually selects tiles based on AI decisions, stopping on a bomb.
  - Communicated with a Python server via sockets, sending screenshots, visible/hidden states, and fitness scores (coins collected).

- **Data Generation** (`eval_client.lua`, `eval_server.py`):
  - Generated 17,000+ grayscale screenshots (256x384, cropped to 190x187) and corresponding CSV files (`visible_states.csv`, `hidden_states.csv`) for randomly seeded game levels.
  - Visible states included coin/bomb counts (10 features) and tile visibility (25 features, 0=hidden, 1–4=revealed).
  - Hidden states included actual tile values (25 features, 1–4).
  - Saved data to `training_data/screenshots/` and `training_data/*.csv`, indexed by `state_index`.
  - Used `process_screenshot()` to crop and convert PNGs to grayscale, and `process_gamestate()` to save states as CSV rows.

- **AI Model Development and Evaluation** (`visible_cnn.py`, `hidden_hybrid.py`, `hidden_lstm.py`, `eval_server.py`):
  - **Visible State Prediction** (`VoltorbFlipCNN`):
    - Designed a CNN with three convolutional layers (3→32, 32→64, 64→128, 3x3 kernels, ReLU, max-pooling 2x2) and two fully-connected layers (dynamic→256, 256→585).
    - Input: 32x32 RGB screenshots (transformed via `transforms.Resize`, `transforms.ToTensor`).
    - Output: 45 features (10 coin/bomb counts, 25 tile visibilities, 13 classes each).
    - Trained on 80% of 17,000+ samples (batch size 16, 50 epochs, Adam optimizer, BCE loss), achieving 99.57% test accuracy.
  - **Hidden State Prediction** (`HybridModel`):
    - Developed a hybrid model combining tabular data (45 visible features) and image data (grayscale screenshots).
    - Tabular branch: Two fully-connected layers (45→128, 128→64, ReLU).
    - Image branch: Modified ResNet-18 (1-channel input, 64 output features).
    - Combined branch: Fully-connected layer (64+64→100, reshaped to 25x4 for 25 tiles, 4 classes).
    - Input: 64x64 grayscale screenshots (normalized) and visible state vectors.
    - Output: 25 tile predictions (1–4, shifted to 0–3 for training).
    - Trained on 80% of data (batch size 64, 10 epochs, Adam optimizer, cross-entropy loss), achieving 99.83% test accuracy.
  - **Alternative Hidden State Prediction** (`hidden_lstm.py`):
    - Implemented an LSTM model (embedding 45→256, LSTM 128, dense 25) to predict the next tile based on visible states.
    - Trained on 90% of data (100 epochs, batch size 128, Adam optimizer, sparse categorical cross-entropy), achieving ~accuracy (not used in final evaluation).
  - **Decision-Making** (`eval_server.py`):
    - Processed screenshots and visible states through `VoltorbFlipCNN` and `HybridModel` to predict game states.
    - Generated decision scores for each tile, penalizing bombs (score=1-score), trivial tiles (score-0.25), and favoring high-value coins.
    - Sorted tiles by scores using `sort_actions()`, selecting the highest-scoring unrevealed tile.
  - **Evaluation**:
    - Ran 255 games in evaluation mode, each starting from savestate slot 2 with debug seeds (0–254).
    - Measured fitness as cumulative coins, stopping on a bomb or reaching 50,000 coins.
    - Logged visible/hidden state accuracies (e.g., 86.67–91.11% visible, 88% hidden) and decisions.

## Tools and Technologies
- **Python**: Developed CNN models, server logic, and data processing pipelines.
- **PyTorch**: Implemented `VoltorbFlipCNN` and `HybridModel` (ResNet-18), trained with Adam optimizer and BCE/cross-entropy loss.
- **Keras**: Built LSTM model for hidden state prediction.
- **NumPy/Pandas**: Handled game state data, preprocessing, and CSV generation.
- **PIL/Matplotlib**: Processed and visualized screenshots.
- **Lua**: Scripted emulator control and memory access in BizHawk-2.9.1.
- **BizHawk-2.9.1**: Emulated NDS gameplay for data generation and evaluation.
- **Socket Programming**: Enabled client-server communication between Lua script and Python server.
- **scikit-learn**: Performed train-test splits and data preprocessing.
- **Training Data**: 17,000+ grayscale screenshots and CSV states in `training_data/`.

## Results
- **Data Generation**:
  - Collected 17,000+ cropped grayscale screenshots (190x187) and CSV states (`visible_states.csv`, `hidden_states.csv`) for random game levels.
  - Structured data with 45 visible features (10 coin/bomb counts, 25 tile visibilities) and 25 hidden features (tile values).
- **Model Performance**:
  - `VoltorbFlipCNN`: Achieved 99.57% test accuracy on visible state prediction (50 epochs, BCE loss).
  - `HybridModel`: Achieved 99.83% test accuracy on hidden state prediction (10 epochs, cross-entropy loss).
  - Evaluation accuracies: 86.67–91.11% for visible states, ~88% for hidden states (per logs).
- **Game Performance**:
  - AI reached level 7, collecting 1851 coins in a single evaluation run (255 games, bomb terminates run).
  - Demonstrated robust tile selection, avoiding bombs through heuristic scoring (penalizing bombs/trivial tiles).
- **Outputs**:
  - Saved models: `weights/visible_cnn.pth`, `weights/hidden_hybrid.pth`, `weights/hidden_lstm.keras`.
  - Generated logs (`logs/eval_server.log`, `logs/screenshots/debug_*.png`) and training data (`training_data/`).

## Skills Demonstrated
- **Computer Vision and Machine Learning**:
  - Designed and trained CNNs (`VoltorbFlipCNN`, `HybridModel` with ResNet-18) for visible and hidden state prediction, achieving 99.57% and 99.83% test accuracies.
  - Processed 17,000+ grayscale screenshots for training, applying cropping, resizing, and normalization.
  - Developed heuristic decision-making to select optimal tiles, balancing coin collection and bomb avoidance.
- **Emulator Interfacing and Memory Mapping**:
  - Debugged NDS memory addresses to extract game states (tiles, coins, bombs, dialogue).
  - Wrote Lua scripts for BizHawk-2.9.1 to automate gameplay, generate data, and evaluate AI decisions.
- **Algorithm Development**:
  - Implemented socket-based client-server communication for real-time game state exchange.
  - Designed decision-making heuristics, sorting tiles by predicted scores with penalties for bombs and trivial tiles.
  - Built data generation pipelines for random game levels, producing structured CSV and image datasets.
- **Technical Proficiency**:
  - Used PyTorch for CNN training, Keras for LSTM, and NumPy/Pandas for data manipulation.
  - Applied socket programming for Lua-Python integration.
  - Tuned hyperparameters (e.g., batch sizes 16/64, learning rate 0.001, epochs 10/50) for optimal model performance.
  - Delivered well-documented code, logs, and visualizations, suitable for research and engineering roles.

## Installation

1. Clone the project repository.
   
2. Navigate to the root project directory and install the Python requirements:
   
   `pip3 install -r requirements.txt`

3. Install the required dependencies for the [BizHawk emulator](https://github.com/TASEmulators/BizHawk). The emulator itself is already included in this project.
4. Place your (legally obtained) game ROM into the `emu/` directory:

   `emu/4781 - Pokemon SoulSilver (U)(Xenophobia).nds`

## References
- [Voltorb Flip (Bulbapedia)](https://bulbapedia.bulbagarden.net/wiki/Voltorb_Flip)
- [BizHawk Emulator](https://github.com/TASEmulators/BizHawk)
- [Pokémon Data Structures (Project Pokémon)](https://projectpokemon.org/home/docs/gen-4/)
