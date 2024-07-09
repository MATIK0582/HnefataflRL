import copy
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

from Hnefatafl import Hnefatafl
from trainining import DQN, select_action
from utils import generate_pawn_moves, get_reward

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

NUMBER_OF_POSSIBLE_PAWN_MOVES_AT_7X7 = 587
STATE_SIZE = 6 * 7 * 7 # 294
OUTPUT_SIZE = NUMBER_OF_POSSIBLE_PAWN_MOVES_AT_7X7
ALL_POSSIBLE_MOVES = generate_pawn_moves()

# Hyperparameters
learning_rate = 0.001
gamma = 0.99  # discount factor
memory_size = 10000
batch_size = 64
eps_start = 1.0
eps_end = 0.01
eps_decay = 200
target_update = 10

state_dim = STATE_SIZE
action_dim = OUTPUT_SIZE

policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)

steps_done = 0
for episode in range(3000):
    game = Hnefatafl()
    state = copy.deepcopy(game.get_game_state())
    total_reward = 0

    print(episode)
    while not game.is_game_over():

        eps_threshold = eps_end + (eps_start - eps_end) * \
                        np.exp(-1. * steps_done / eps_decay)
        # TODO: CHANGE ME
        # eps_threshold = 1

        # print(ALL_POSSIBLE_MOVES)
        # print(game.get_all_legal_moves())

        action = select_action(policy_net, state, eps_threshold, OUTPUT_SIZE, ALL_POSSIBLE_MOVES, game.get_all_legal_moves())
        # move = input(f"{'Escaping' if game.escapee_turn else 'Attacking'} player move (e.g. e2 e4): ")

        old_board = copy.deepcopy(game.board)
        old_king_pos = copy.deepcopy(game.king_position)
        old_player_turn = copy.deepcopy(game.escapee_turn)

        try:
            start, end = game.parse_move(ALL_POSSIBLE_MOVES[action.item()][1])
            if game.validate_move(start, end):
                game.make_move(start, end)
                game.is_game_over()
                game.switch_player()

        ##########################################
        ##########################################

                reward = get_reward(old_board, game.board, old_king_pos, game.king_position, old_player_turn, game.game_result)
                memory.append((state, action, reward, game.get_game_state()))
                state = copy.deepcopy(game.get_game_state())
                total_reward += reward
                steps_done += 1

                if len(memory) > batch_size:
                    transitions = random.sample(memory, batch_size)
                    batch = list(zip(*transitions))
                    states, actions, rewards, next_states = batch

                    state_batch = torch.FloatTensor(np.array(states)).to(device)
                    action_batch = torch.cat(actions).to(device)
                    reward_batch = torch.FloatTensor(np.array(rewards)).to(device)
                    next_state_batch = torch.FloatTensor(np.array(next_states)).to(device)

                    current_q_values = policy_net(state_batch).gather(1, action_batch)
                    next_q_values = target_net(next_state_batch).max(1)[0].detach()
                    expected_q_values = reward_batch + gamma * next_q_values

                    loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        ##########################################
        ##########################################

            else:
                print(f"Invalid move. Try again. \n")
        except Exception as e:
            print(f"Error: {e}. Try again.")
    game.print_board()

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

torch.save(policy_net.state_dict(), 'model_state_3000.pth')