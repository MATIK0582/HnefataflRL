from constants import BOARD_CORNER_CHAR, BOARD_EMPTY_CHAR, BOARD_SIZE, MIDDLE_SQUARE
import numpy as np
from utils import count_pieces


class Hnefatafl:
    def __init__(self):
        self.size = 7
        self.board = [[BOARD_EMPTY_CHAR for _ in range(self.size)] for _ in range(self.size)]
        self.setup_board()
        self.escapee_turn = False
        self.stalemate = False
        self.king_position = (MIDDLE_SQUARE, MIDDLE_SQUARE)
        self.game_result = 'ongoing'

    def setup_board(self):
        self.board[0][0] = BOARD_CORNER_CHAR
        self.board[0][BOARD_SIZE - 1] = BOARD_CORNER_CHAR
        self.board[BOARD_SIZE - 1][0] = BOARD_CORNER_CHAR
        self.board[BOARD_SIZE - 1][BOARD_SIZE - 1] = BOARD_CORNER_CHAR

        for i in range(MIDDLE_SQUARE - 1, MIDDLE_SQUARE + 2):
            for j in range(MIDDLE_SQUARE - 1, MIDDLE_SQUARE + 2):
                self.board[i][j] = self.board[i][MIDDLE_SQUARE] = 'D'
        self.board[MIDDLE_SQUARE][MIDDLE_SQUARE] = 'K'

        for i in range(2, self.size - 2):
            self.board[0][i] = self.board[self.size - 1][i] = 'A'
            self.board[i][0] = self.board[i][self.size - 1] = 'A'

        self.board[1][MIDDLE_SQUARE] = 'A'
        self.board[BOARD_SIZE - 2][MIDDLE_SQUARE] = 'A'
        self.board[MIDDLE_SQUARE][1] = 'A'
        self.board[MIDDLE_SQUARE][BOARD_SIZE - 2] = 'A'

    def print_board(self):
        print("   " + " ".join(chr(ord('a') + i) for i in range(self.size)))
        for i in range(self.size):
            print(f"{self.size - i:2} " + " ".join(self.board[i]) + f" {self.size - i}")
        print("   " + " ".join(chr(ord('a') + i) for i in range(self.size)))

    def parse_move(self, move):
        start, end = move.split()
        start = (self.size - int(start[1:]), ord(start[0]) - ord('a'))
        end = (self.size - int(end[1:]), ord(end[0]) - ord('a'))
        return start, end

    def validate_move(self, start, end, verbose=False):
        start_i, start_j = start
        end_i, end_j = end

        # Are coordinates different from each other
        if start_i == end_i and start_j == end_j:
            if verbose:
                print('End coordinates must be different from start ones')
            return False

        # Is player moving a pawn
        if self.board[start_i][start_j] not in ('A', 'D', 'K'):
            if verbose:
                print('Only pawns are movable pieces')
            return False

        # Is player moving his pawns:
        if self.escapee_turn and self.board[start_i][start_j] not in ('D', 'K'):
            if verbose:
                print('You are defending. Move D or K pawn')
            return False
        if not self.escapee_turn and self.board[start_i][start_j] not in 'A':
            if verbose:
                print('You are attacking. Move A pawns')
            return False

        # Check if the destination is empty or allowable for king
        if self.board[end_i][end_j] != BOARD_EMPTY_CHAR:
            if self.board[start_i][start_j] != 'K':
                if self.board[end_i][end_j] != BOARD_CORNER_CHAR:
                    if verbose:
                        print('Move into empty place only')
                    return False
            else:  # Figure is 'K'
                if self.board[end_i][end_j] in ('A', 'D'):
                    if verbose:
                        print('King cannot move to a space occupied by A or D')
                    return False

        # # Check if the destination is empty
        # if self.board[end_i][end_j] != BOARD_EMPTY_CHAR:
        #     if self.board[start_i][start_j] != 'K' and self.board[end_i][end_j] != BOARD_CORNER_CHAR:
        #         if verbose:
        #             print('Move into empty place only')
        #         return False
        #

        # Is not-king pawn trying to go to corner
        if self.board[start_i][start_j] in ('A', 'D'):
            if self.board[end_i][end_j] == BOARD_CORNER_CHAR:
                if verbose:
                    print('Only king pawn can enter corner')
                return False

        # Is move in a straight line
        if start_i != end_i and start_j != end_j:
            if verbose:
                print('Move must be in straight line')
            return False

        # Are there no obstacles on the move path
        for i in range(min(start_i, end_i) + 1, max(start_i, end_i)):
            if self.board[i][start_j] != BOARD_EMPTY_CHAR:
                if verbose:
                    print('Pawn can\'t jump over pieces')
                return False
        for j in range(min(start_j, end_j) + 1, max(start_j, end_j)):
            if self.board[start_i][j] != BOARD_EMPTY_CHAR:
                if verbose:
                    print('Pawn can\'t jump over pieces')
                return False

        # If no error was returned
        return True

    def make_move(self, start, end):
        start_i, start_j = start
        end_i, end_j = end

        piece = self.board[start_i][start_j]
        self.board[start_i][start_j] = BOARD_EMPTY_CHAR
        self.board[end_i][end_j] = piece

        if piece == 'K':
            self.king_position = (end_i, end_j)

        self.capture_pieces(end_i, end_j)

    def capture_pieces(self, end_i, end_j):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            opponent_i, opponent_j = end_i + di, end_j + dj
            if 0 <= opponent_i < self.size and 0 <= opponent_j < self.size:
                opponent = 'A' if self.escapee_turn else 'D'
                player_pieces = ('D', 'K', BOARD_CORNER_CHAR) if self.escapee_turn else ('A', BOARD_CORNER_CHAR)
                if self.board[opponent_i][opponent_j] in opponent:
                    adjacent_i, adjacent_j = opponent_i + di, opponent_j + dj
                    if 0 <= adjacent_i < self.size and 0 <= adjacent_j < self.size and self.board[adjacent_i][
                        adjacent_j] in player_pieces:
                        self.board[opponent_i][opponent_j] = BOARD_EMPTY_CHAR

    def switch_player(self):
        self.escapee_turn = not self.escapee_turn

    def is_king_captured(self):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        capture_conditions = 0

        king_i, king_j = self.king_position

        for di, dj in directions:
            neighbor_i, neighbor_j = king_i + di, king_j + dj
            if 0 <= neighbor_i < self.size and 0 <= neighbor_j < self.size:
                if self.board[neighbor_i][neighbor_j] == 'A':
                    capture_conditions += 1
                elif (neighbor_i in [0, self.size - 1] or neighbor_j in [0, self.size - 1]) and \
                        self.board[neighbor_i][neighbor_j] == BOARD_CORNER_CHAR:
                    capture_conditions += 1
            else:
                return False  # Jeśli pole jest poza planszą, to król nie jest złapany

        return capture_conditions >= 4

    def is_game_over(self):
        king_i, king_j = self.king_position
        is_over = False

        # Check if no moves are avaliavble
        if len(self.get_all_legal_moves()) == 0 and count_pieces(self.board, 'A') == 0:
            self.game_result = 'escaped'
            print("Defender wins! King escaped.")
            return True

        elif len(self.get_all_legal_moves()) == 0 and self.is_king_captured():
            self.game_result = 'captured'
            print("Attacker wins! King captured.")
            return True

        elif len(self.get_all_legal_moves()) == 0:
            self.game_result = 'stalemate'
            print("Stalemate")
            return True

        # Check if king escaped
        if self.board[0][0] == 'K' or self.board[0][self.size - 1] == 'K' or \
                self.board[self.size - 1][0] == 'K' or self.board[self.size - 1][self.size - 1] == 'K':
            self.game_result = 'escaped'
            print("Defender wins! King escaped.")
            return True

        # Check if king was captured
        # Keep playing if king is has free square next to him or defender
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_i, neighbor_j = king_i + di, king_j + dj
            if 0 <= neighbor_i < self.size and 0 <= neighbor_j < self.size:
                if self.board[neighbor_i][neighbor_j] == BOARD_EMPTY_CHAR or self.board[neighbor_i][neighbor_j] == 'D':
                    is_over = False

        # Check for attackers presence0
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 'A':
                    is_over = False

        # Else king was captured
        if self.is_king_captured():
            self.game_result = 'captured'
            print("Attacker wins! King captured.")
            return True

        return is_over

    def play(self):
        while not self.is_game_over():
            self.print_board()

            move = input(f"{'Escaping' if self.escapee_turn else 'Attacking'} player move (e.g. e2 e4): ")

            try:
                start, end = self.parse_move(move)
                if self.validate_move(start, end):
                    self.make_move(start, end)
                    self.switch_player()
                else:
                    print(f"Invalid move. Try again. \n")
            except Exception as e:
                print(f"Error: {e}. Try again.")
        self.print_board()

    def get_all_legal_moves(self):
        legal_moves = []
        player_pieces = ['A'] if not self.escapee_turn else ['D', 'K']

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] in player_pieces:
                    start_pos = f"{chr(ord('a') + j)}{self.size - i}"

                    # Generate vertical moves
                    for ni in range(self.size):
                        if ni != i:
                            if self.validate_move((i, j), (ni, j), verbose=False):
                                end_pos = f"{chr(ord('a') + j)}{self.size - ni}"
                                legal_moves.append(start_pos + ' ' + end_pos)

                    # Generate horizontal moves
                    for nj in range(self.size):
                        if nj != j:
                            if self.validate_move((i, j), (i, nj), verbose=False):
                                end_pos = f"{chr(ord('a') + nj)}{self.size - i}"
                                legal_moves.append(start_pos + ' ' + end_pos)

        return legal_moves

    def get_game_state(self):
        # Create a tensor of shape (6, 7, 7)
        state = np.zeros((6, self.size, self.size))

        # Layer 0: escapee turn indicator
        state[0] = np.full((self.size, self.size), 1 if self.escapee_turn else -1)

        # Fill in other layers based on board content
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 'A':
                    state[1][i][j] = 1
                elif self.board[i][j] == 'D':
                    state[2][i][j] = 1
                elif self.board[i][j] == 'K':
                    state[3][i][j] = 1
                elif self.board[i][j] == BOARD_EMPTY_CHAR:
                    state[4][i][j] = 1
                elif self.board[i][j] == BOARD_CORNER_CHAR:
                    state[5][i][j] = 1

        return state


game = Hnefatafl()
# game.play()
