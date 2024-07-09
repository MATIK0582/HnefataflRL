import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # discount factor
memory_size = 10000
batch_size = 64
eps_start = 1.0
eps_end = 0.01
eps_decay = 200
target_update = 10


# Q-Network
# class DQN(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(DQN, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, action_dim)
#
#     def forward(self, x):
#         x = x.reshape(-1, 6*7*7)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened feature map
        self.flattened_size = 64 * 7 * 7  # 64 channels from conv2 with 7x7 feature map size

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 600)  # 600 output neurons for Q-values


    def forward(self, x):
        # Pass through convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from the conv layers
        x = x.view(-1, self.flattened_size)

        # Pass through fully connected layers with ReLU activations
        x = F.relu(self.fc1(x))

        # Output layer (Q-values)
        x = self.fc2(x)

        return x


# Helper function to select an action with epsilon-greedy policy
def select_action(model, state, eps_threshold, output_size, all_possible_moves, legal_moves):
    if random.random() > eps_threshold:
        with torch.no_grad():
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze()

            legal_indices = [index for index, move in all_possible_moves if move in legal_moves]

            max_q_index = q_values[legal_indices].argmax().item()
            best_legal_index = legal_indices[max_q_index]

            return torch.tensor([[best_legal_index]])
    else:
        # Wybierz losowy ruch spośród legalnych
        random_move = random.choice(legal_moves)

        # Znajdź indeks tego ruchu w all_possible_moves
        random_move_index = next(index for index, move in all_possible_moves if move == random_move)

        return torch.tensor([[random_move_index]])
