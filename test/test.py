# test.py

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Bước 1: Test GridWorld Environment
def test_step1():
    print("=== BƯỚC 1: Test GridWorld Environment ===")
    try:
        from gridworld import GridWorldEnv
        
        # Tạo environment nhỏ
        env = GridWorldEnv(n_width=4, n_height=4, default_reward=-0.1)
        env.start = (0, 0)
        env.ends = [(3, 3)]
        env.rewards = [(3, 3, 10)]
        env.refresh_setting()
        
        print(f"✓ Environment created successfully!")
        print(f"  - Size: {env.n_width}x{env.n_height}")
        print(f"  - State space: {env.observation_space.n}")
        print(f"  - Action space: {env.action_space.n}")
        
        # Test reset và step
        state = env.reset()
        print(f"  - Initial state: {state}")
        
        # Test một vài steps
        for i in range(5):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            print(f"  - Step {i+1}: action={action}, next_state={next_state}, reward={reward}")
            if done:
                print(f"  - Episode ended at step {i+1}")
                break
        
        print("✓ GridWorld test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"✗ GridWorld test FAILED: {e}\n")
        return False

# Bước 2: Test PyTorch và DQN Network
def test_step2():
    print("=== BƯỚC 2: Test PyTorch và DQN Network ===")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Test PyTorch
        print(f"✓ PyTorch version: {torch.__version__}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ Device: {device}")
        
        # Test DQN Network
        class SimpleDQN(nn.Module):
            def __init__(self, state_size, action_size):
                super(SimpleDQN, self).__init__()
                self.fc1 = nn.Linear(state_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, action_size)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return self.fc3(x)
        
        # Test network
        net = SimpleDQN(16, 4)  # 4x4 grid = 16 states, 4 actions
        test_input = torch.randn(1, 16)
        output = net(test_input)
        
        print(f"✓ DQN Network created successfully!")
        print(f"  - Input shape: {test_input.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output values: {output.detach().numpy().flatten()}")
        
        print("✓ PyTorch test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch test FAILED: {e}\n")
        return False

# Bước 3: Test DQN Agent cơ bản
def test_step3():
    print("=== BƯỚC 3: Test DQN Agent cơ bản ===")
    try:
        # Import cần thiết
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import random
        from collections import deque
        
        # Simple DQN Agent
        class SimpleDQNAgent:
            def __init__(self, state_size, action_size):
                self.state_size = state_size
                self.action_size = action_size
                self.epsilon = 1.0
                
                # Simple network
                self.q_network = nn.Sequential(
                    nn.Linear(state_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_size)
                )
                
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.01)
            
            def act(self, state):
                if random.random() <= self.epsilon:
                    return random.randrange(self.action_size)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        
        # Test agent
        agent = SimpleDQNAgent(16, 4)
        
        # Test action selection
        test_state = np.zeros(16)
        test_state[0] = 1  # One-hot encoding for state 0
        
        action = agent.act(test_state)
        print(f"✓ Agent created successfully!")
        print(f"  - State size: {agent.state_size}")
        print(f"  - Action size: {agent.action_size}")
        print(f"  - Test action: {action}")
        print(f"  - Epsilon: {agent.epsilon}")
        
        print("✓ DQN Agent test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"✗ DQN Agent test FAILED: {e}\n")
        return False

# Bước 4: Test training loop đơn giản
def test_step4():
    print("=== BƯỚC 4: Test Training Loop đơn giản ===")
    try:
        from gridworld import GridWorldEnv
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import random
        
        # Simple agent (copy từ step 3)
        class SimpleDQNAgent:
            def __init__(self, state_size, action_size):
                self.state_size = state_size
                self.action_size = action_size
                self.epsilon = 1.0
                self.epsilon_decay = 0.99
                self.epsilon_min = 0.1
                
                self.q_network = nn.Sequential(
                    nn.Linear(state_size, 32),
                    nn.ReLU(),
                    nn.Linear(32, action_size)
                )
                
                self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.01)
                self.memory = []
            
            def act(self, state):
                if random.random() <= self.epsilon:
                    return random.randrange(self.action_size)
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
            
            def remember(self, state, action, reward, next_state, done):
                self.memory.append((state, action, reward, next_state, done))
                if len(self.memory) > 1000:
                    self.memory.pop(0)
            
            def replay(self):
                if len(self.memory) < 32:
                    return
                
                # Đơn giản hóa: chỉ train trên experience cuối
                state, action, reward, next_state, done = self.memory[-1]
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                target = reward
                if not done:
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    target += 0.99 * self.q_network(next_state_tensor).max()
                
                current_q = self.q_network(state_tensor)[0][action]
                loss = nn.MSELoss()(current_q, target)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
        
        # Tạo environment và agent
        env = GridWorldEnv(n_width=3, n_height=3, default_reward=-0.1)
        env.start = (0, 0)
        env.ends = [(2, 2)]
        env.rewards = [(2, 2, 10)]
        env.refresh_setting()
        
        agent = SimpleDQNAgent(9, 4)  # 3x3 = 9 states
        
        # Training loop ngắn
        scores = []
        for episode in range(50):
            state = env.reset()
            state_features = np.zeros(9)
            state_features[state] = 1
            
            episode_reward = 0
            for step in range(20):
                action = agent.act(state_features)
                next_state, reward, done, _ = env.step(action)
                
                next_state_features = np.zeros(9)
                next_state_features[next_state] = 1
                
                agent.remember(state_features, action, reward, next_state_features, done)
                agent.replay()
                
                state_features = next_state_features
                episode_reward += reward
                
                if done:
                    break
            
            scores.append(episode_reward)
            
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                print(f"  Episode {episode}: avg_score={avg_score:.2f}, epsilon={agent.epsilon:.3f}")
        
        print(f"✓ Training completed!")
        print(f"  - Episodes: 50")
        print(f"  - Final average score: {np.mean(scores[-10:]):.2f}")
        print(f"  - Final epsilon: {agent.epsilon:.3f}")
        
        print("✓ Training Loop test PASSED!\n")
        return True
        
    except Exception as e:
        print(f"✗ Training Loop test FAILED: {e}\n")
        return False

# Chạy tất cả tests
def run_all_tests():
    print("CHẠY TẤT CẢ TESTS DQN GRIDWORLD\n")
    
    tests = [
        ("GridWorld Environment", test_step1),
        ("PyTorch & DQN Network", test_step2),
        ("DQN Agent", test_step3),
        ("Training Loop", test_step4)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("=" * 50)
    print("KẾT QUẢ TỔNG HỢP:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status} - {test_name}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n TẤT CẢ TESTS ĐỀU PASS! Bạn có thể chạy DQN đầy đủ!")
        print("\n BƯỚC TIẾP THEO:")
        print("1. Chạy: python main.py")
        print("2. Hoặc import và sử dụng DQN agent từ artifacts")
    else:
        print("\n  MỘT SỐ TESTS FAIL. Hãy kiểm tra lại:")
        print("1. Cài đặt đủ thư viện")
        print("2. Kiểm tra file gridworld.py")
        print("3. Kiểm tra version Python/PyTorch")

if __name__ == "__main__":
    run_all_tests()