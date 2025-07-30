# 🧠 DQN Agent for GridWorld & OpenAI Gym Environments

A PyTorch implementation of Deep Q-Learning (DQN) for various GridWorld scenarios and classic OpenAI Gym environments.
This project demonstrates how to train a DQN agent using both custom GridWorld environments and standard OpenAI Gym environments like CartPole and LunarLander.

---

## 📁 Folder Structure

```
.
├── config/                    # YAML config files for environment setup
│   └── env.yaml
├── models/                   # Saved trained models
│   ├── dqn_gridworld_basic.pth
│   └── dqn_gridworld_maze.pth
├── reinforcement/            # Core reinforcement learning modules
│   ├── agent.py
│   ├── dqn.py
│   └── utils.py
├── results/                  # Output folder for plots, training results, etc.
├── test/                     # Test script for basic functionality
│   └── test.py
├── gridworld.py              # Environment definition (4x4 GridWorld)
├── main.py                   # Entry point for training and testing
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore
```

---

## ⚙️ Installation

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

---

## 🚀 Training the Agent

To train the agent in a specific environment (e.g., basic or maze):

### Train on GridWorld Environments

```bash
python main.py --config gridworld_basic --episodes 1500 --visualize
```
```bash
python main.py --config gridworld_advanced --episodes 2000 --visualize
```
```bash
python main.py --config gridworld_maze --episodes 3000 --visualize
```
```bash
python main.py --config cliff_walking --episodes 1500 --visualize
```
```bash
python main.py --config windy_gridworld --episodes 1500 --visualize

```
### Train on OpenAI Gym Environment

```bash
python main.py --config cartpole --episodes 500
```
```bash
python main.py --config lunarlander --episodes 1000
```

- `--episodes`: Number of episodes to train
- `--visualize`: (Optional) Turn on real-time training visualization

---

## 📂 Supported Environments

The following environments are available via `--config`:

- `gridworld_basic`
- `gridworld_advanced`
- `gridworld_maze`
- `cliff_walking`
- `windy_gridworld`
- `cartpole`
- `lunarlander`

## 🧪 Testing the Agent

Run all the built-in tests to verify the setup:

```bash
python test/test.py
```

This checks:
- GridWorld environment functionality
- DQN network architecture
- Agent logic
- Training loop

---

## 📊 Visualizing Results

After training with `--visualize`, the `results/` folder will contain:
- Training scores (plot image)
- Reward history
- Saved model weights in `models/`

You can also modify the plotting functions inside `utils.py` to customize the visuals.

---

## 🛠 Configuration

Edit `config/env.yaml` to modify:
- Grid size
- Reward structure
- Training hyperparameters

---

## ✅ Requirements

Make sure you're using:
- Python 3.12.3
- PyTorch >= 2.7.1 (CPU-only is fine)

---

## 📌 Notes

- The project uses **one-hot encoding** for GridWorld states.
- Both ε-greedy strategy and target network synchronization are implemented.
- Modular design allows easy extension to other environments.