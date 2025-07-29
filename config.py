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

# utils.py - Utility functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gridworld import GridWorldEnv

def create_custom_environment(env_type="default"):
    """Create different types of GridWorld environments"""
    
    if env_type == "simple":
        env = GridWorldEnv(n_width=5, n_height=5, default_reward=-0.1)
        env.start = (0, 0)
        env.ends = [(4, 4)]
        env.rewards = [(4, 4, 10)]
        
    elif env_type == "maze":
        env = GridWorldEnv(n_width=10, n_height=10, default_reward=-0.1)
        env.start = (0, 0)
        env.ends = [(9, 9)]
        # Create maze-like obstacles
        obstacles = [
            (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
            (4, 1), (4, 2), (4, 3), (4, 4),
            (6, 3), (6, 4), (6, 5), (6, 6), (6, 7),
            (8, 1), (8, 2), (8, 3), (8, 4), (8, 5)
        ]
        env.types = [(x, y, 1) for x, y in obstacles]
        env.rewards = [(9, 9, 20), (5, 5, -10), (7, 2, -10)]
        
    elif env_type == "cliff":
        return GridWorldEnv.CliffWalk()
        
    elif env_type == "skull_treasure":
        return GridWorldEnv.SkullAndTreasure()
        
    else:  # default
        env = GridWorldEnv(n_width=10, n_height=7, default_reward=-0.1)
        env.start = (0, 0)
        env.ends = [(9, 6)]
        env.types = [(3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 1)]
        env.rewards = [(9, 6, 10), (2, 5, -5), (7, 2, -5)]
    
    env.refresh_setting()
    return env

def visualize_policy(env, agent, title="Learned Policy"):
    """Visualize the learned policy as arrows on the grid"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create policy matrix
    policy_matrix = np.zeros((env.n_height, env.n_width))
    value_matrix = np.zeros((env.n_height, env.n_width))
    
    # Action to arrow mapping
    action_arrows = {0: '←', 1: '→', 2: '↑', 3: '↓'}
    
    for y in range(env.n_height):
        for x in range(env.n_width):
            state = env._xy_to_state(x, y)
            state_features = np.zeros(env.observation_space.n)
            state_features[state] = 1
            
            # Get action from agent
            action = agent.act(state_features, training=False)
            policy_matrix[env.n_height-1-y, x] = action
            
            # Get Q-values for visualization
            import torch
            state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(agent.device)
            q_values = agent.q_network(state_tensor)
            value_matrix[env.n_height-1-y, x] = torch.max(q_values).item()
    
    # Create heatmap of values
    sns.heatmap(value_matrix, annot=False, cmap='RdYlBu_r', ax=ax, cbar=True)
    
    # Add arrows for policy
    for y in range(env.n_height):
        for x in range(env.n_width):
            if not env._is_end_state(x, env.n_height-1-y) and env.grids.get_type(x, env.n_height-1-y) != 1:
                action = int(policy_matrix[y, x])
                ax.text(x+0.5, y+0.5, action_arrows[action], 
                       ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Mark start and end positions
    start_y = env.n_height - 1 - env.start[1]
    ax.add_patch(plt.Rectangle((env.start[0], start_y), 1, 1, 
                              fill=False, edgecolor='green', lw=3))
    ax.text(env.start[0]+0.1, start_y+0.1, 'S', fontsize=12, color='green', fontweight='bold')
    
    for end in env.ends:
        end_y = env.n_height - 1 - end[1]
        ax.add_patch(plt.Rectangle((end[0], end_y), 1, 1, 
                                  fill=False, edgecolor='red', lw=3))
        ax.text(end[0]+0.1, end_y+0.1, 'G', fontsize=12, color='red', fontweight='bold')
    
    # Mark obstacles
    for x in range(env.n_width):
        for y in range(env.n_height):
            if env.grids.get_type(x, y) == 1:
                obstacle_y = env.n_height - 1 - y
                ax.add_patch(plt.Rectangle((x, obstacle_y), 1, 1, 
                                          fill=True, facecolor='black', alpha=0.7))
    
    ax.set_title(title)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.tight_layout()
    plt.show()

def visualize_q_values(env, agent, state_x, state_y):
    """Visualize Q-values for a specific state"""
    state = env._xy_to_state(state_x, state_y)
    state_features = np.zeros(env.observation_space.n)
    state_features[state] = 1
    
    import torch
    state_tensor = torch.FloatTensor(state_features).unsqueeze(0).to(agent.device)
    q_values = agent.q_network(state_tensor).cpu().data.numpy()[0]
    
    actions = ['Left', 'Right', 'Up', 'Down']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(actions, q_values)
    plt.title(f'Q-Values for State ({state_x}, {state_y})')
    plt.ylabel('Q-Value')
    
    # Highlight the best action
    best_action = np.argmax(q_values)
    bars[best_action].set_color('red')
    
    # Add value labels on bars
    for i, v in enumerate(q_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def compare_algorithms(env, agents_dict, episodes=100):
    """Compare performance of different agents"""
    results = {}
    
    for name, agent in agents_dict.items():
        scores = []
        for episode in range(episodes):
            state = env.reset()
            state_features = np.zeros(env.observation_space.n)
            state_features[state] = 1
            total_reward = 0
            steps = 0
            
            while steps < 200:
                action = agent.act(state_features, training=False)
                next_state, reward, done, _ = env.step(action)
                next_state_features = np.zeros(env.observation_space.n)
                next_state_features[next_state] = 1
                
                state_features = next_state_features
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            scores.append(total_reward)
        
        results[name] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    names = list(results.keys())
    means = [results[name]['mean'] for name in names]
    stds = [results[name]['std'] for name in names]
    
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.title('Algorithm Performance Comparison')
    plt.ylabel('Average Score')
    plt.xticks(rotation=45)
    
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.5, f'{mean:.2f}±{std:.2f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_exploration(agent):
    """Analyze agent's exploration behavior"""
    epsilon_history = []
    # This would need to be collected during training
    # For now, we'll create a theoretical decay curve
    
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    for step in range(10000):
        epsilon_history.append(epsilon)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_history)
    plt.title('Epsilon Decay During Training')
    plt.xlabel('Training Step')
    plt.ylabel('Epsilon Value')
    plt.grid(True)
    plt.show()

def create_training_report(scores, avg_scores, losses, agent, env):
    """Create a comprehensive training report"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training scores
    ax1.plot(scores, alpha=0.3, label='Episode Score')
    ax1.plot(avg_scores, label='Average Score (100 episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True)
    
    # Loss curve
    if losses:
        ax2.plot(losses)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True)
    
    # Score distribution
    ax3.hist(scores[-500:], bins=30, alpha=0.7)
    ax3.set_xlabel('Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Score Distribution (Last 500 Episodes)')
    ax3.grid(True)
    
    # Learning curve smoothed
    window_size = 50
    smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')
    ax4.plot(smoothed_scores)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Smoothed Score')
    ax4.set_title(f'Learning Curve (Moving Average {window_size})')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=== Training Summary ===")
    print(f"Total Episodes: {len(scores)}")
    print(f"Final Average Score: {avg_scores[-1]:.2f}")
    print(f"Best Episode Score: {max(scores):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Total Training Steps: {agent.steps}")
    print(f"Average Loss: {np.mean(losses[-1000:]):.6f}" if losses else "No loss data")

# Example usage functions
def quick_test():
    """Quick test of the DQN system"""
    
    env = create_custom_environment("simple")
    agent = DQNAgent(env.observation_space.n, env.action_space.n)
    
    print("Running quick test...")
    scores, avg_scores = train_dqn(env, agent, episodes=500)
    
    visualize_policy(env, agent, "Quick Test Policy")
    create_training_report(scores, avg_scores, agent.losses, agent, env)

if __name__ == "__main__":
    # Run a quick test
    quick_test()