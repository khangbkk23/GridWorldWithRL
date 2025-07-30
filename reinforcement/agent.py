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
import copy

from reinforcement.dqn import DQN, ReplayBuffer

# Import GridWorld components
from gridworld import GridWorldEnv

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = 'results'
os.makedirs(RUNS_DIR, exist_ok=True)

class DQNAgent:
    """DQN Agent"""
    def __init__(self, hyperparameter_set, render=False):
        config_path = './config/env.yaml'
        with open(config_path, 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        
        self.hyperparameter_set = hyperparameter_set
        self.render = render
        self.env_id = hyperparameters['env_id']
        
        if self.env_id == 'GridWorld':
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

        self.q_network = DQN(self.state_size, self.action_size, self.fc1_nodes).to(self.device)
        self.target_network = DQN(self.state_size, self.action_size, self.fc1_nodes).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate_a)
        self.loss_fn = nn.MSELoss()

        self.replay_memory = ReplayBuffer(self.replay_memory_size)
        self.update_target_network()
        self.losses = []
        self.rewards_history = []
        self.steps_done = 0
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self._setup_logging()
    
    def _create_gridworld(self, params):
        # Load parameters from env.yaml if not provided directly
        if isinstance(params, str):
            # params is a path to a yaml file
            params = './config/env.yaml'
            with open(params, 'r') as f:
                params = yaml.safe_load(f)

        self.n_width = params.get('n_width', 7)
        self.n_height = params.get('n_height', 7)
        self.default_reward = params.get('default_reward', -0.1)
        self.default_type = params.get('default_type', 0)
        self.windy = params.get('windy', False)

        env = GridWorldEnv(
            n_width=self.n_width,
            n_height=self.n_height,
            u_size=params.get('u_size', 40),
            default_reward=self.default_reward,
            default_type=self.default_type,
            windy=self.windy
        )
        
        if 'start' in params:
            env.start = tuple(params['start'])
        if 'ends' in params:
            env.ends = [tuple(end) for end in params['ends']]
        
        if 'types' in params:
            env.types = []
            for type_info in params['types']:
                if len(type_info) == 3:
                    env.types.append(tuple(type_info))
                else:
                    env.types.append((type_info[0], type_info[1], 1))

        if 'rewards' in params:
            env.rewards = []
            for reward_info in params['rewards']:
                env.rewards.append(tuple(reward_info))
        env.refresh_setting()
        return env
    
    def _setup_logging(self):
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
            state_tensor = torch.zeros(self.state_size)
            state_tensor[state] = 1.0
        else:
            state_tensor = torch.FloatTensor(state)
        
        return state_tensor.unsqueeze(0).to(self.device)
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.max(1)[1].item()
    
    def act(self, state, training=True):
        return self.select_action(state, training)
    
    def store_experience(self, state, action, reward, next_state, done):
        self.replay_memory.push(state, action, reward, next_state, done)
    
    def remember(self, state, action, reward, next_state, done):
        self.store_experience(state, action, reward, next_state, done)
    
    def experience_replay(self):
        if len(self.replay_memory) < self.mini_batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.mini_batch_size)
        if hasattr(self.env.observation_space, 'n'):
            state_batch = torch.zeros(self.mini_batch_size, self.state_size)
            next_state_batch = torch.zeros(self.mini_batch_size, self.state_size)
            for i, (s, ns) in enumerate(zip(states, next_states)):
                state_batch[i, int(s)] = 1.0
                next_state_batch[i, int(ns)] = 1.0
        else:
            state_batch = torch.FloatTensor(states)
            next_state_batch = torch.FloatTensor(next_states)

        state_batch = state_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        done_batch = torch.tensor(dones, dtype=torch.bool).to(self.device)
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Dự đoán Q tương lai từ target network
        with torch.no_grad():
            next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_state_batch).gather(1, next_actions).squeeze(1)
            target_q_values = reward_batch + self.discount_factor_g * next_q_values * (~done_batch)

        # Tính loss
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.losses.append(loss.item())
        if self.steps_done % self.network_sync_rate == 0:
            self.update_target_network()

        # Giảm epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        self.steps_done += 1

    
    def replay(self):
        self.experience_replay()
    
    def save(self, filename):
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
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.losses = checkpoint['losses']
        self.rewards_history = checkpoint['rewards_history']
        print(f"Model loaded from {filename}")

def train_dqn(env, agent, episodes=1000, max_steps=200, print_every=100):
    """Training loop for DQN agent"""
    scores = []
    avg_scores = []
    best_avg_score = -float("inf")
    best_model_state = None
    print(f"Starting training for {episodes} episodes...")
    print(f"Device: {agent.device}")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action) 
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            agent.replay()
            
            if done:
                break
        
        scores.append(total_reward)
        agent.rewards_history.append(total_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_model_state = copy.deepcopy(agent.q_network.state_dict())
            
        if episode % print_every == 0:
            print(f"Episode {episode:4d} | "
                  f"Score: {total_reward:6.2f} | "
                  f"Avg Score: {avg_score:6.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps:3d}")
            
            if agent.losses:
                print(f"           | Loss: {np.mean(agent.losses[-100:]):.6f}")
        
        if avg_score >= agent.stop_on_reward:
            print(f"Solved in {episode} episodes! Average score: {avg_score:.2f}")
            break
        
    if best_model_state:
        agent.q_network.load_state_dict(best_model_state)
        agent.update_target_network()
    
    print("Training completed!")
    return scores, avg_scores

def test_agent(env, agent, episodes=10, render=False, max_steps=200):
    """Test the trained agent"""
    scores = []
    
    print(f"Testing agent for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < max_steps:
            if render:
                env.render()
            action = agent.act(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
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
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    # Plot scores
    axes[0].plot(scores, alpha=0.3, color='blue', label='Episode Score')
    axes[0].plot(avg_scores, color='red', linewidth=2, label='Average Score (100 episodes)')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Training Scores Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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
    
    if os.environ.get("DISPLAY"):
        plt.show()
    else:
        plt.close()

def visualize_grid_and_policy(agent, save_path=None):
    if agent.env_id != 'GridWorld':
        print("Grid visualization only available for GridWorld environment")
        return
    
    env = agent.env
    n_width, n_height = env.n_width, env.n_height
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Environment Layout
    grid = np.zeros((n_height, n_width))
    for obstacle in env.types:
        x, y = obstacle[0], obstacle[1]
        if 0 <= x < n_width and 0 <= y < n_height:
            grid[n_height-1-y, x] = -1
    
    # Mark special rewards
    for reward_info in env.rewards:
        x, y, reward = reward_info[0], reward_info[1], reward_info[2]
        if 0 <= x < n_width and 0 <= y < n_height:
            grid[n_height-1-y, x] = reward / 10
    
    # Mark start and goal
    start_x, start_y = env.start
    grid[n_height-1-start_y, start_x] = 0.5
    
    for end in env.ends:
        end_x, end_y = end[0], end[1]
        grid[n_height-1-end_y, end_x] = 1
    
    im1 = ax1.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=1)
    ax1.set_title('GridWorld Environment')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Value')

    ax1.set_xticks(np.arange(-0.5, n_width, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n_height, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    # Plot 2: Policy Visualization
    policy_grid = np.zeros((n_height, n_width))
    arrow_grid = np.zeros((n_height, n_width, 2))
    
    # Action directions for GridWorld: 0=left, 1=right, 2=up, 3=down
    action_arrows = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    
    for x in range(n_width):
        for y in range(n_height):
            # Convert (x,y) to state index using GridWorld's method
            state_idx = env._xy_to_state(x, y)
            
            # Skip obstacles
            if env.grids.get_type(x, y) == 1:
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
    
    for i in range(n_height):
        for j in range(n_width):
            if policy_grid[i, j] >= 0:
                dx, dy = arrow_grid[i, j]
                ax2.arrow(j, i, dx*0.3, -dy*0.3, head_width=0.1, 
                         head_length=0.1, fc='white', ec='white')
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Action (0:Left, 1:Right, 2:Up, 3:Down)')
    
    ax2.set_xticks(np.arange(-0.5, n_width, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, n_height, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grid visualization saved to {save_path}")
    plt.show()
    
def create_predefined_gridworld(env_type):
    """Create predefined GridWorld environments"""
    if env_type == "RandomWalk":
        return GridWorldEnv.RandomWalk()
    elif env_type == "CliffWalk":
        return GridWorldEnv.CliffWalk()
    elif env_type == "SkullAndTreasure":
        return GridWorldEnv.SkullAndTreasure()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")