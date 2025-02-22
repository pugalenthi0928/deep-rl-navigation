import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the GridWorld environment
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.max_steps = 50
        self.reset()

    def reset(self):
        # Agent starts at top-left (0,0), goal at bottom-right (size-1, size-1)
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        # State as 2-channel image: channel 0 for agent, channel 1 for goal
        state = torch.zeros((2, self.size, self.size))
        state[0, self.agent_pos[0], self.agent_pos[1]] = 1.0  # Agent position
        state[1, self.goal_pos[0], self.goal_pos[1]] = 1.0  # Goal position
        return state

    def step(self, action):
        # Action: 0=up, 1=right, 2=down, 3=left
        r, c = self.agent_pos
        if action == 0 and r > 0:
            r -= 1
        elif action == 1 and c < self.size - 1:
            c += 1
        elif action == 2 and r < self.size - 1:
            r += 1
        elif action == 3 and c > 0:
            c -= 1
        self.agent_pos = [r, c]
        self.steps += 1
        # Compute reward
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Reached goal
        else:
            reward = -1.0  # Time penalty
        done = (self.agent_pos == self.goal_pos) or (self.steps >= self.max_steps)
        return self._get_state(), reward, done

# Define the DQN (Deep Q-Network) using convolutional layers for vision input
class DQN(nn.Module):
    def __init__(self, grid_size=5):
        super().__init__()
        # Two convolutional layers to extract features from the 2-channel grid image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3),  # Conv layer (output 16 channels, 3x3 kernel)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2),  # Conv layer (output 32 channels)
            nn.ReLU()
        )
        # Compute size of conv output (for grid_size=5, output feature map will be 2x2x32)
        conv_output_dim = 32 * 2 * 2
        # Fully connected layers to produce Q-values for 4 actions
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 possible actions
        )

    def forward(self, state):
        # state: tensor of shape [batch, 2, grid_size, grid_size]
        features = self.conv_layers(state)  # Extract conv features
        features = features.view(features.size(0), -1)  # Flatten
        q_values = self.fc_layers(features)  # Output Q-values for each action
        return q_values

# Initialize environment and Q-network
env = GridWorld(size=5)
q_network = DQN(grid_size=5)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)

# Training parameters
episodes = 300
gamma = 0.9  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.1
epsilon_decay = 0.995

# Training loop (Q-learning)
for ep in range(episodes):
    state = env.reset()
    done = False
    while not done:
        # Choose action (ε-greedy policy)
        if random.random() < epsilon:
            action = random.randint(0, 3)  # Explore
        else:
            with torch.no_grad():
                q_vals = q_network(state.unsqueeze(0))  # Get Q-values for state
                action = int(torch.argmax(q_vals[0]).item())  # Choose best action
        # Take action in the environment
        next_state, reward, done = env.step(action)
        # Compute target Q-value
        with torch.no_grad():
            max_next_q = 0.0
            if not done:
                max_next_q = torch.max(q_network(next_state.unsqueeze(0))[0]).item()
        target_q = reward + gamma * max_next_q
        # Update Q-network
        q_vals = q_network(state.unsqueeze(0))
        current_q = q_vals[0, action]
        loss = (current_q - target_q) ** 2  # MSE loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Move to next state
        state = next_state
    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Test the trained agent (greedy policy) from the start
state = env.reset()
done = False
path = [tuple(env.agent_pos)]  # Record the agent's path
while not done:
    with torch.no_grad():
        q_vals = q_network(state.unsqueeze(0))
        action = int(torch.argmax(q_vals[0]).item())
    state, reward, done = env.step(action)
    path.append(tuple(env.agent_pos))

print("Agent path:", path)
