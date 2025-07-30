import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from gridworld import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent"""
    def __init__(self, state_size, action_size, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=32, target_update=100):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Update target network
        self.update_target()
        
        # Training statistics
        self.losses = []
        self.rewards_history = []
        self.steps = 0
    
    def update_target(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        
        if self.steps % self.target_update == 0:
            self.update_target()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
    
    def save(self, filename):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
    
    def load(self, filename):
        """Load model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']

def state_to_features(state, env):
    features = np.zeros(env.observation_space.n)
    features[state] = 1
    return features

def train_dqn(env, agent, episodes=1000, max_steps=200):
    """Training loop"""
    scores = []
    avg_scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_features = state_to_features(state, env)
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state_features)
            next_state, reward, done, info = env.step(action)
            next_state_features = state_to_features(next_state, env)
            
            agent.remember(state_features, action, reward, next_state_features, done)
            state_features = next_state_features
            total_reward += reward
            
            agent.replay()
            
            if done:
                break
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % 50 == 0:
            print(f"Episode {episode}")
            print(f"Average Score: {avg_scores[-1]:.2f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            print(f"Loss: {np.mean(agent.losses[-100:]):.6f}")
            print("-" * 30)
    
    return scores, avg_scores

def test_agent(env, agent, episodes=10, render=False):
    """Test trained agent"""
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        state_features = state_to_features(state, env)
        total_reward = 0
        steps = 0
        
        while True:
            if render:
                env.render()
            
            action = agent.act(state_features, training=False)
            next_state, reward, done, info = env.step(action)
            next_state_features = state_to_features(next_state, env)
            
            state_features = next_state_features
            total_reward += reward
            steps += 1
            
            if done or steps > 200:
                break
        
        scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}, Steps = {steps}")
    
    print(f"Average Test Score: {np.mean(scores):.2f}")
    return scores

def plot_training_results(scores, avg_scores, losses):
    """Plot training results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot scores
    ax1.plot(scores, alpha=0.3, label='Episode Score')
    ax1.plot(avg_scores, label='Average Score (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Plot losses
    if losses:
        ax2.plot(losses)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create environment
    env = GridWorldEnv(n_width=10, n_height=7, default_reward=-0.1)
    env.ends = [(9, 6)]  # Goal at top-right
    env.start = (0, 0)   # Start at bottom-left
    # Add some obstacles
    env.types = [(3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 1)]
    # Add some rewards/penalties
    env.rewards = [(9, 6, 10), (2, 5, -5), (7, 2, -5)]
    env.refresh_setting()
    
    # Create DQN agent
    state_size = env.observation_space.n  # One-hot encoding size
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    print("Starting training...")
    scores, avg_scores = train_dqn(env, agent, episodes=2000)
    
    # Save trained model
    agent.save("dqn_gridworld.pth")
    
    # Plot results
    plot_training_results(scores, avg_scores, agent.losses)
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_scores = test_agent(env, agent, episodes=5, render=True)
    
    print("Training completed!")