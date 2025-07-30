# main.py - DQN Training with YAML Config

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from reinforcement.agent import DQNAgent, train_dqn, test_agent, plot_training_results, visualize_grid_and_policy

def print_environment_info(agent):
    print("=" * 60)
    print("ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"Configuration: {agent.hyperparameter_set}")
    print(f"Environment ID: {agent.env_id}")
    
    if agent.env_id == 'GridWorld':
        env = agent.env
        print(f"Grid Size: {env.n_width} x {env.n_height}")
        print(f"Start Position: {env.start}")
        print(f"Goal Position(s): {env.ends}")
        print(f"Default Reward: {agent.env_make_params.get('default_reward', -0.1)}")
        
        if hasattr(env, 'types') and env.types:
            print(f"Obstacles: {len(env.types)} positions")
        
        if hasattr(env, 'rewards') and env.rewards:
            print(f"Special Rewards: {len(env.rewards)} positions")
    
    print(f"State Space Size: {agent.state_size}")
    print(f"Action Space Size: {agent.action_size}")
    print(f"Device: {agent.device}")
    print("=" * 60)

def print_training_config(agent):
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Replay Memory Size: {agent.replay_memory_size}")
    print(f"Batch Size: {agent.mini_batch_size}")
    print(f"Learning Rate: {agent.learning_rate_a}")
    print(f"Discount Factor (γ): {agent.discount_factor_g}")
    print(f"Initial Epsilon: {agent.epsilon_init}")
    print(f"Epsilon Decay: {agent.epsilon_decay}")
    print(f"Min Epsilon: {agent.epsilon_min}")
    print(f"Target Network Update: Every {agent.network_sync_rate} steps")
    print(f"Hidden Layer Size: {agent.fc1_nodes}")
    print(f"Stop on Reward: {agent.stop_on_reward}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='DQN Training with YAML Config')
    parser.add_argument('--config', type=str, default='gridworld_basic',
                       help='Configuration name from YAML file')
    parser.add_argument('--episodes', type=int, default=1500,
                       help='Number of training episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during testing')
    parser.add_argument('--test-only', type=str, default=None,
                       help='Path to saved model for testing only')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize grid and policy after training')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Create agent from config
    print(f"Creating DQN agent with config: {args.config}")
    try:
        agent = DQNAgent(args.config, render=args.render)
    except Exception as e:
        print(f"Error creating agent: {e}")
        print("Available configs in hyper_param.yml:")
        print("- gridworld_basic")
        print("- gridworld_advanced") 
        print("- gridworld_maze")
        print("- cliff_walking")
        print("- cartpole")
        print("- lunarlander")
        return
    
    # Print information
    print_environment_info(agent)
    print_training_config(agent)

    if args.test_only:
        print(f"Loading model from {args.test_only}")
        agent.load(args.test_only)
        print("Testing loaded model...")
        test_scores = test_agent(agent.env, agent, episodes=10, render=args.render)
        print(f"Average test score: {np.mean(test_scores):.2f}")
        
        if args.visualize and agent.env_id == 'GridWorld':
            visualize_grid_and_policy(agent, f'results/{args.config}_policy_test.png')
        return
    
    print(f"Starting training for {args.episodes} episodes...")
    start_time = datetime.now()
    
    max_steps = 200
    if agent.env_id == 'GridWorld':
        env = agent.env
        grid_size = env.n_width * env.n_height
        max_steps = min(300, max(100, grid_size * 2))
    
    scores, avg_scores = train_dqn(
        agent.env, agent, 
        episodes=args.episodes, 
        max_steps=max_steps,
        print_every=max(50, args.episodes // 20)
    )
    
    end_time = datetime.now()
    training_time = end_time - start_time

    print("=" * 60)
    print("TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Training Time: {training_time}")
    print(f"Total Episodes: {len(scores)}")
    print(f"Final Average Score: {avg_scores[-1]:.2f}")
    print(f"Best Episode Score: {max(scores):.2f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Total Training Steps: {agent.steps_done}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'models/dqn_{args.config}_{timestamp}.pth'
    agent.save(model_filename)
    plot_save_path = f'results/{args.config}_training_{timestamp}.png'
    plot_training_results(scores, avg_scores, agent.losses, plot_save_path)
    
    # Test the trained agent
    print("\n" + "=" * 60)
    print("TESTING TRAINED AGENT")
    print("=" * 60)
    test_scores = test_agent(agent.env, agent, episodes=10, render=args.render)
    
    print(f"\nFinal Results:")
    print(f"Training Average Score: {avg_scores[-1]:.2f}")
    print(f"Test Average Score: {np.mean(test_scores):.2f}")
    print(f"Model saved to: {model_filename}")
    
    # Visualize grid and policy (for GridWorld only)
    if args.visualize and agent.env_id == 'GridWorld':
        print("\nVisualizing grid and learned policy...")
        policy_save_path = f'results/{args.config}_policy_{timestamp}.png'
        visualize_grid_and_policy(agent, policy_save_path)

if __name__ == "__main__":
    main()