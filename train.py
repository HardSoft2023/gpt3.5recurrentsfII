import retro
import cv2
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import torch
from collections import namedtuple

# Define the device to use (either 'cpu' or 'cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
LOG_INTERVAL = 10

# Define the transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Define the replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the preprocessing function
def preprocess(state):
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (84, 84))
    state = np.ascontiguousarray(state, dtype=np.float32) / 255
    state = np.expand_dims(state, axis=0)
    return state

# Define the DQN model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Define the select action function
def select_action(model, state, eps_threshold):
    sample = random.random()
    if sample > eps_threshold:
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            q_values = model(state)
            action = q_values.max(1)[1].unsqueeze(0).item()
    else:
        action = random.randrange(env.action_space.n)
    
    # Convert action value to button values
    buttons = [False] * len(env.buttons)
    buttons[action] = True
    
    return action, buttons

# Define the optimization function
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s, device=device, dtype=torch.float).unsqueeze(0)
                                       for s in batch.next_state if s is not None])
    state_batch = torch.tensor(np.array(batch.state), device=device, dtype=torch.float)
    action_batch = torch.tensor(np.array(batch.action), device=device, dtype=torch.long).reshape(-1)  # Reshape action_batch
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Initialize the environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

# Initialize the model and optimizer
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())

# Initialize the replay memory and variables
memory = ReplayMemory(10000)
state = env.reset()
state = preprocess(state)
eps_threshold = EPS_START
steps_done = 0
episode_rewards = []
num_episodes = 1000  # or any other number of episodes you want to train for

# Train the model over the episodes
best_reward = float('-inf')
for i_episode in range(num_episodes):
    episode_reward = 0
    state = env.reset()
    state = preprocess(state)

    for t in count():
        # Select an action using an epsilon-greedy policy
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            np.exp(-1. * steps_done / EPS_DECAY)
        # Select an action using an epsilon-greedy policy
        action, buttons = select_action(policy_net, state, eps_threshold)

        # Perform the action
        next_state, reward, done, info = env.step(buttons)
        reward = torch.tensor([reward], device=device)

        # Preprocess the next state and store the transition in memory
        if not done:
            next_state = preprocess(next_state)
        else:
            next_state = None
        memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state
        episode_reward += reward.item()
        steps_done += 1

        # Optimize the model
        optimize_model()
        if steps_done % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if done:
            episode_rewards.append(episode_reward)
            break

    # Save the best model every 100 steps
    if episode_reward > best_reward and i_episode % 100 == 0:
        best_reward = episode_reward
        torch.save(policy_net.state_dict(), 'best_model.pth')

    # Print progress
    if i_episode % LOG_INTERVAL == 0:
        print('Episode {}\tAverage reward: {:.2f}'.format(
            i_episode, np.mean(episode_rewards[-LOG_INTERVAL:])))

# Save the final model
torch.save(policy_net.state_dict(), 'final_model.pth')

# Close the environment
env.close()