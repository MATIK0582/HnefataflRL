import copy
import torch

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


state_dim = STATE_SIZE
action_dim = OUTPUT_SIZE

policy_net = DQN(state_dim, action_dim).to(device)
policy_net.load_state_dict(torch.load('model_state.pth'))
policy_net.eval()


steps_done = 0
for episode in range(200):
    game = Hnefatafl()
    state = copy.deepcopy(game.get_game_state())

    print(episode)
    while not game.is_game_over():

        eps_threshold = 0
        if game.escapee_turn:
            eps_threshold = 1

        action = select_action(policy_net, state, eps_threshold, OUTPUT_SIZE, ALL_POSSIBLE_MOVES, game.get_all_legal_moves())


        try:
            start, end = game.parse_move(ALL_POSSIBLE_MOVES[action.item()][1])
            if game.validate_move(start, end):
                game.make_move(start, end)
                game.is_game_over()
                game.switch_player()
            else:
                print(f"Invalid move. Try again. \n")
        except Exception as e:
            print(f"Error: {e}. Try again.")
    game.print_board()
