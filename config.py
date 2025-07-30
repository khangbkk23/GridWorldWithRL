# config.py - Configuration file
from reinforcement.agent import DQNAgent, train_dqn

class Config:
    """Configuration for DQN GridWorld"""
    
    # Environment settings
    ENV_WIDTH = 10
    ENV_HEIGHT = 7
    CELL_SIZE = 40
    DEFAULT_REWARD = -0.1
    
    # DQN Hyperparameters
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    
    # Training settings
    BUFFER_SIZE = 10000
    BATCH_SIZE = 32
    TARGET_UPDATE = 100
    EPISODES = 2000
    MAX_STEPS = 200
    
    # Network architecture
    HIDDEN_SIZE = 128
    
    # Logging
    LOG_INTERVAL = 100
    SAVE_MODEL = True
    MODEL_PATH = "models/dqn_gridworld.pth"