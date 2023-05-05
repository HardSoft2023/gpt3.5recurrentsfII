import retro
import cv2
import torch
import numpy as np
from train import DQN, preprocess

# Define the device to use (either 'cpu' or 'cuda')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = DQN().to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

# Initialize the environment
env = retro.make(game='StreetFighterIISpecialChampionEdition-Genesis')

# Play one episode using the trained model
state = env.reset()
state = preprocess(state)
episode_reward = 0

while True:
    # Select an action using the trained model
    with torch.no_grad():
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q_values = model(state)
        action = q_values.max(1)[1].unsqueeze(0).item()

    # Convert action value to button values
    buttons = [False] * len(env.buttons)
    buttons[action] = True

    # Perform the action
    next_state, reward, done, info = env.step(buttons)
    episode_reward += reward

    # Preprocess the next state
    next_state = preprocess(next_state)

    # Move to the next state
    state = next_state

    # Render the game screen and print the action taken by the model
    env.render()
    print(f"Action taken: {action}")

    if done:
        print(f"Episode finished with reward {episode_reward}")
        break

# Close the environment
env.close()