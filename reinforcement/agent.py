# reinforcement/agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import random
import yaml
import os
import logging
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = 'results'
os.makedirs(RUNS_DIR, exist_ok=True)

class DQN(nn.Module):
    """Deep Q-Network"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent"""
    def __init__(self, hyperparameter_set, render=False):
        # Load config từ file YAML
        config_path = './hyper_param.yml'  # Sử dụng file config của bạn
        with open(config_path, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        self.render = render
        self.env_id = hyperparameters['env_id']
        
        # Create environment
        if self.env_id == 'GridWorld':
            from gridworld import GridWorldEnv
            self.env = self._create_gridworld(hyperparameters.get('env_make_param', {}))
        else:
            import gym
            self.env = gym.make(self.env_id, render_mode="human" if render else None)
        
        # Hyperparameters
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.env_make_params = hyperparameters.get('env_make_param', {})
        
        # Current epsilon
        self.epsilon = self.epsilon_init
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get state and action dimensions
        if hasattr(self.env.observation_space, 'n'):
            self.state_size = self.env.observation_space.n
        else:
            self.state_size = self.env.observation_space.shape[0]
        
        if hasattr(self.env.action_space, 'n'):
            self.action_size = self.env.action_space.n
        else:
            self.action_size = self.env.action_space.shape[0]
        
        # Neural Networks
        self.q_network = DQN(self.state_size, self.action_size, self.fc1_nodes).to(self.device)
        self.target_network = DQN(self.state_size, self.action_size, self.fc1_nodes).to(self.device)
        
        # Optimizer and Loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate_a)
        self.loss_fn = nn.MSELoss()
        
        # Replay Memory
        self.replay_memory = ReplayBuffer(self.replay_memory_size)
        
        # Update target network
        self.update_target_network()
        
        # Training statistics
        self.losses = []
        self.rewards_history = []
        self.steps_done = 0
        
        # Logging
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self._setup_logging()
    
    def _create_gridworld(self, params):
        """Create GridWorld environment with parameters"""
        from gridworld import GridWorldEnv
        
        env = GridWorldEnv(
            n_width=params.get('n_width', 10),
            n_height=params.get('n_height', 7),
            u_size=params.get('u_size', 40),
            default_reward=params.get('default_reward', -0.1),
            default_type=params.get('default_type', 0),
            windy=params.get('windy', False)
        )
        
        # Set start and end positions
        env.start = tuple(params.get('start', [0, 0]))
        env.ends = [tuple(end) for end in params.get('ends', [[9, 6]])]
        
        # Set special types (obstacles)
        if 'types' in params:
            env.types = [tuple(t) for t in params['types']]
        
        # Set special rewards
        if 'rewards' in params:
            env.rewards = [tuple(r) for r in params['rewards']]
        
        env.refresh_setting()
        return env
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            filename=self.LOG_FILE,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger(__name__)
        
        # Log hyperparameters
        self.logger.info(f"Starting training with hyperparameter set: {self.hyperparameter_set}")
        self.logger.info(f"Environment: {self.env_id}")
        self.logger.info(f"State size: {self.state_size}, Action size: {self.action_size}")
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def state_to_tensor(self, state):
        """Convert state to tensor"""
        if hasattr(self.env.observation_space, 'n'):
            # Discrete state space - use one-hot encoding
            state_tensor = torch.zeros(self.state_size)
            state_tensor[state] = 1.0
        else:
            # Continuous state space
            state_tensor = torch.FloatTensor(state)
        
        return state_tensor.unsqueeze(0).to(self.device)
    
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()
    
    def act(self, state, training=True):
        """Alias for select_action"""
        return self.select_action(state, training)
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.replay_memory.push(state, action, reward, next_state, done)
    
    def remember(self, state, action, reward, next_state, done):
        """Alias for store_experience"""
        self.store_experience(state, action, reward, next_state, done)
    
    def experience_replay(self):
        """Train the network on a batch of experiences"""
        if len(self.replay_memory) < self.mini_batch_size:
            return
        
        # Sample batch
        batch = self.replay_memory.sample(self.mini_batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        if hasattr(self.env.observation_space, 'n'):
            # One-hot encoding for discrete states
            state_batch = torch.zeros(self.mini_batch_size, self.state_size)
            next_state_batch = torch.zeros(self.mini_batch_size, self.state_size)
            for i, (s, ns) in enumerate(zip(states, next_states)):
                state_batch[i, s] = 1.0
                next_state_batch[i, ns] = 1.0
        else:
            state_batch = torch.FloatTensor(states)
            next_state_batch = torch.FloatTensor(next_states)
        
        state_batch = state_batch.to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.discount_factor_g * next_q_values * ~done_batch)
        
        # Compute loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        # Update target network
        if self.steps_done % self.network_sync_rate == 0:
            self.update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps_done += 1
    
    def replay(self):
        """Alias for experience_replay"""
        self.experience_replay()
    
    def save(self, filename):
        """Save the trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'losses': self.losses,
            'rewards_history': self.rewards_history,
            'hyperparameter_set': self.hyperparameter_set
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.losses = checkpoint['losses']
        self.rewards_history = checkpoint['rewards_history']
        print(f"Model loaded from {filename}")

def state_to_features(state, env=None):
    """Convert environment state to features"""
    if isinstance(state, tuple):
        # Convert (x, y) to state index
        x, y = state
        if env is not None:
            return x * env.n_height + y
        else:
            return x * 10 + y  # Default assumption
    return state

def train_dqn(env, agent, episodes=1000, max_steps=200, print_every=100):
    """Training loop for DQN agent"""
    scores = []
    avg_scores = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"Device: {agent.device}")
    
    for episode in range(episodes):
        state = env.reset()
        state_features = state_to_features(state, env)
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state_features)
            next_state, reward, done, info = env.step(action)
            next_state_features = state_to_features(next_state, env)
            
            agent.remember(state_features, action, reward, next_state_features, done)
            state_features = next_state_features
            total_reward += reward
            steps += 1
            
            agent.replay()
            
            if done:
                break
        
        scores.append(total_reward)
        agent.rewards_history.append(total_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        if episode % print_every == 0:
            print(f"Episode {episode:4d} | "
                  f"Score: {total_reward:6.2f} | "
                  f"Avg Score: {avg_score:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps:3d}")
            
            if agent.losses:
                print(f"           | Loss: {np.mean(agent.losses[-100:]):.6f}")
        
        # Early stopping condition
        if avg_score >= agent.stop_on_reward:
            print(f"Solved in {episode} episodes! Average score: {avg_score:.2f}")
            break
    
    print("Training completed!")
    return scores, avg_scores

def test_agent(env, agent, episodes=10, render=False, max_steps=200):
    """Test the trained agent"""
    scores = []
    
    print(f"Testing agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        state_features = state_to_features(state, env)
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            if render:
                env.render()
            
            action = agent.act(state_features, training=False)
            next_state, reward, done, info = env.step(action)
            next_state_features = state_to_features(next_state, env)
            
            state_features = next_state_features
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        print(f"Test Episode {episode + 1:2d}: Score = {total_reward:6.2f}, Steps = {steps:3d}")
    
    avg_score = np.mean(scores)
    print(f"Average Test Score: {avg_score:.2f}")
    return scores

def plot_training_results(scores, avg_scores, losses=None, save_path=None):
    """Plot training results"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot scores
    axes[0].plot(scores, alpha=0.3, color='blue', label='Episode Score')
    axes[0].plot(avg_scores, color='red', linewidth=2, label='Average Score (100 episodes)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Training Scores Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot losses
    if losses and len(losses) > 0:
        if len(losses) > 100:
            smoothed_losses = []
            window = 50
            for i in range(window, len(losses)):
                smoothed_losses.append(np.mean(losses[i-window:i]))
            axes[1].plot(smoothed_losses, color='green', linewidth=1.5)
            axes[1].set_title('Training Loss (Smoothed)')
        else:
            axes[1].plot(losses, color='green', linewidth=1.5)
            axes[1].set_title('Training Loss')
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No loss data available', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Training Loss - No Data')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def visualize_grid_and_policy(agent, save_path=None):
    """Visualize GridWorld environment and learned policy"""
    if agent.env_id != 'GridWorld':
        print("Grid visualization only available for GridWorld environment")
        return
    
    env = agent.env
    n_width, n_height = env.n_width, env.n_height
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Environment Layout
    grid = np.zeros((n_height, n_width))
    
    # Mark obstacles
    for obstacle in env.types:
        x, y = obstacle[0], obstacle[1]
        if 0 <= x < n_width and 0 <= y < n_height:
            grid[n_height-1-y, x] = -1  # Obstacle
    
    # Mark special rewards
    for reward_info in env.rewards:
        x, y, reward = reward_info[0], reward_info[1], reward_info[2]
        if 0 <= x < n_width and 0 <= y < n_height:
            grid[n_height-1-y, x] = reward / 10  # Scale for visualization
    
    # Mark start and goal
    start_x, start_y = env.start
    grid[n_height-1-start_y, start_x] = 0.5  # Start
    
    for end in env.ends:
        end_x, end_y = end[0], end[1]
        grid[n_height-1-end_y, end_x] = 1  # Goal
    
    im1 = ax1.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('GridWorld Environment')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Value')
    
    # Add grid lines
    ax1.set_xticks(np.arange(-0.5, n_width, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n_height, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Plot 2: Policy Visualization
    policy_grid = np.zeros((n_height, n_width))
    arrow_grid = np.zeros((n_height, n_width, 2))
    
    # Action directions (right, up, left, down)
    action_arrows = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    for x in range(n_width):
        for y in range(n_height):
            state_idx = x * n_height + y
            
            # Skip obstacles
            if (x, y, 1) in env.types:
                policy_grid[n_height-1-y, x] = -1
                continue
            
            # Get best action for this state
            state_tensor = agent.state_to_tensor(state_idx)
            with torch.no_grad():
                q_values = agent.q_network(state_tensor)
            best_action = q_values.max(1)[1].item()
            
            policy_grid[n_height-1-y, x] = best_action
            
            # Store arrow direction
            dx, dy = action_arrows[best_action]
            arrow_grid[n_height-1-y, x] = [dx, dy]
    
    im2 = ax2.imshow(policy_grid, cmap='tab10', vmin=-1, vmax=3)
    ax2.set_title('Learned Policy (Best Actions)')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    
    # Add arrows for policy
    for i in range(n_height):
        for j in range(n_width):
            if policy_grid[i, j] >= 0:  # Not an obstacle
                dx, dy = arrow_grid[i, j]
                ax2.arrow(j, i, dx*0.3, -dy*0.3, head_width=0.1, 
                         head_length=0.1, fc='white', ec='white')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Action (0:Right, 1:Up, 2:Left, 3:Down)')
    
    # Add grid lines
    ax2.set_xticks(np.arange(-0.5, n_width, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n_height, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid visualization saved to {save_path}")
    
    plt.show()