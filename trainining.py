import torch
import torch.nn as nn
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
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# # Initialize environment, network, and optimizer
# # env = gym.make('CartPole-v1')
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
#
# policy_net = DQN(state_dim, action_dim)
# target_net = DQN(state_dim, action_dim)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()  # Set the target net to eval mode
#
# optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
# memory = deque(maxlen=memory_size)


# Helper function to select an action with epsilon-greedy policy
def select_action(model, state, eps_threshold, output_size, all_possible_moves, legal_moves):
    if random.random() > eps_threshold:
        with torch.no_grad():

            # TODO:
            # TODO: Czy poprawnie V przekazane .to(device)
            # TODO:
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device)).squeeze()

            legal_indices = [index for index, move in all_possible_moves if move in legal_moves]

            max_q_index = q_values[legal_indices].argmax().item()
            best_legal_index = legal_indices[max_q_index]

            return torch.tensor([[best_legal_index]], dtype=torch.long)
    else:
        # Wybierz losową akcję z pełnego zakresu output_size
        return random.choice(legal_moves)

# # Training loop
# steps_done = 0
# for episode in range(200):
#     state = env.reset()
#     total_reward = 0
#     done = False
#
#     while not done:
#         eps_threshold = eps_end + (eps_start - eps_end) * \
#                         np.exp(-1. * steps_done / eps_decay)
#         action = select_action(state, eps_threshold)
#         next_state, reward, done, _ = env.step(action.item())
#         memory.append((state, action, reward, next_state, done))
#
#         state = next_state
#         total_reward += reward
#         steps_done += 1
#
#         if len(memory) > batch_size:
#             transitions = random.sample(memory, batch_size)
#             batch = list(zip(*transitions))
#             states, actions, rewards, next_states, dones = batch
#
#             state_batch = torch.FloatTensor(states)
#             action_batch = torch.cat(actions)
#             reward_batch = torch.FloatTensor(rewards)
#             next_state_batch = torch.FloatTensor(next_states)
#             done_batch = torch.FloatTensor(dones)
#
#             current_q_values = policy_net(state_batch).gather(1, action_batch)
#             next_q_values = target_net(next_state_batch).max(1)[0].detach()
#             expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)
#
#             loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#     if episode % target_update == 0:
#         target_net.load_state_dict(policy_net.state_dict())
#
#     print(f"Episode {episode}, Total reward: {total_reward}, Epsilon: {eps_threshold}")
#
# # Close the environment
# env.close()
