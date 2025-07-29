# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from gridworld import GridWorldEnv
from reinforcement.agent import DQNAgent, train_dqn, test_agent, plot_training_results

def main():
    # Tạo thư mục để lưu kết quả
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Tạo environment
    print("Creating GridWorld environment...")
    env = GridWorldEnv(n_width=8, n_height=6, default_reward=-0.1)
    
    # Thiết lập environment
    env.start = (0, 0)        # Điểm bắt đầu
    env.ends = [(7, 5)]       # Điểm kết thúc (goal)
    
    # Thêm một số chướng ngại vật
    env.types = [
        (2, 2, 1), (2, 3, 1), (2, 4, 1),  # Tường dọc
        (4, 1, 1), (4, 2, 1), (4, 3, 1),  # Tường dọc khác
        (6, 3, 1), (6, 4, 1)               # Chướng ngại vật
    ]
    
    # Thêm rewards/penalties
    env.rewards = [
        (7, 5, 100),    # Reward lớn ở goal
        (3, 4, -10),    # Penalty 
        (5, 2, -10),    # Penalty khác
        (6, 1, 5)       # Small reward
    ]
    
    env.refresh_setting()
    
    print(f"Environment created: {env.n_width}x{env.n_height}")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    
    # Tạo DQN agent
    print("Creating DQN agent...")
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        batch_size=32
    )
    
    # Training
    print("Starting training...")
    episodes = 1500
    scores, avg_scores = train_dqn(env, agent, episodes=episodes, max_steps=100)
    
    print("Training completed!")
    
    # Lưu model
    model_path = 'models/dqn_gridworld_trained.pth'
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Vẽ kết quả training
    plot_training_results(scores, avg_scores, agent.losses)
    
    # Test agent đã train
    print("\nTesting trained agent...")
    test_scores = test_agent(env, agent, episodes=5, render=False)
    
    print(f"\nTraining Summary:")
    print(f"Episodes: {episodes}")
    print(f"Final average score: {avg_scores[-1]:.2f}")
    print(f"Best episode score: {max(scores):.2f}")
    print(f"Test average score: {np.mean(test_scores):.2f}")

if __name__ == "__main__":
    main()