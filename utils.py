from collections import deque

import numpy

from constants import BOARD_CORNER_CHAR, BOARD_EMPTY_CHAR, BOARD_SIZE, MIDDLE_SQUARE


def generate_pawn_moves():
    columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    rows = ['1', '2', '3', '4', '5', '6', '7']

    moves = []

    index = 0
    for col in columns:
        for row in rows:
            start_pos = col + row
            # Ruchy pionowe
            for r in rows:
                if r != row:
                    end_pos = col + r
                    moves.append((index, f"{start_pos} {end_pos}"))
                    index += 1
            # Ruchy poziome
            for c in columns:
                if c != col:
                    end_pos = c + row
                    moves.append((index, f"{start_pos} {end_pos}"))
                    index += 1
    return moves


def count_pieces(board, piece):
    count = 0
    for row in board:
        for cell in row:
            if cell == piece:
                count += 1
    return count


def is_within_board(i, j, board_size=BOARD_SIZE):
    return 0 <= i < board_size and 0 <= j < board_size


def is_king_sandwiched(board, king_position):
    king_i, king_j = king_position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # góra, dół, lewo, prawo
    osaczony = False
    bardzo_osaczony = False

    # Sprawdzanie, czy król jest osaczony z dwóch stron
    for d1, d2 in [(0, 1), (2, 3)]:  # sprawdzamy pary: góra-dół i lewo-prawo
        di1, dj1 = directions[d1]
        di2, dj2 = directions[d2]

        neighbor1_i, neighbor1_j = king_i + di1, king_j + dj1
        neighbor2_i, neighbor2_j = king_i + di2, king_j + dj2

        if is_within_board(neighbor1_i, neighbor1_j) and is_within_board(neighbor2_i, neighbor2_j):
            if (board[neighbor1_i][neighbor1_j] == 'A' and
                    (board[neighbor2_i][neighbor2_j] == 'A' or board[neighbor2_i][
                        neighbor2_j] == BOARD_CORNER_CHAR)):
                osaczony = True
            if (board[neighbor2_i][neighbor2_j] == 'A' and
                    (board[neighbor1_i][neighbor1_j] == 'A' or board[neighbor1_i][
                        neighbor1_j] == BOARD_CORNER_CHAR)):
                osaczony = True

    # Sprawdzanie, czy król jest osaczony z trzech stron
    sides_blocked = 0
    for di, dj in directions:
        neighbor_i, neighbor_j = king_i + di, king_j + dj
        if is_within_board(neighbor_i, neighbor_j):
            if board[neighbor_i][neighbor_j] == 'A' or board[neighbor_i][neighbor_j] == BOARD_CORNER_CHAR:
                sides_blocked += 1

    if sides_blocked >= 3:
        bardzo_osaczony = True

    return osaczony, bardzo_osaczony


def king_moves_to_reach_corner(board, king_position, size=BOARD_SIZE):
    king_i, king_j = king_position
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # góra, dół, lewo, prawo
    corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]

    queue = deque([(king_i, king_j, 0)])  # Kolejka przechowuje również licznik ruchów
    visited = set()
    visited.add((king_i, king_j))

    # print(f"Start position of the king: {king_i}, {king_j}")
    # print(f"Corners: {corners}")

    while queue:
        current_i, current_j, moves = queue.popleft()

        # print(f"Current position: ({current_i}, {current_j}), Moves: {moves}")

        if (current_i, current_j) in corners:
            # print(f"King reached corner ({current_i}, {current_j}) in {moves} moves")
            return moves  # Król dotarł do narożnika

        for di, dj in directions:
            next_i, next_j = current_i + di, current_j + dj
            # print(f"Checking direction: ({di}, {dj})")

            while is_within_board(next_i, next_j) and board[next_i][next_j] == '·':
                # print(f"Next position: ({next_i}, {next_j})")
                if (next_i, next_j) not in visited:
                    queue.append((next_i, next_j, moves + 1))
                    visited.add((next_i, next_j))
                    # print(f"Adding to queue: ({next_i}, {next_j}), Moves: {moves + 1}")
                next_i += di
                next_j += dj

            if is_within_board(next_i, next_j) and (next_i, next_j) in corners:
                # print( f"King can move directly to corner ({next_i}, {next_j}) from ({current_i}, {current_j})
                # in {moves + 1} moves")
                return moves + 1

    print("King cannot reach any corner")
    return numpy.inf  # Król nie może dotrzeć do żadnego narożnika


def get_reward(initial_board, current_board, initial_king_position, current_king_position, escapee_turn, game_result):
    # Wagi dla różnych elementów nagrody
    COUNT_WEIGHT = 1.0
    SANDWICH_WEIGHT = 2.0
    TRIPLE_SANDWICH_WEIGHT = 3.0
    CORNER_WEIGHT = 2.5
    WIN_WEIGHT = 100.0

    reward = 0.0

    # Sprawdź stan sandwicha i ruchy króla do narożnika przed i po ruchu
    prev_sandwich, prev_triple_sandwich = is_king_sandwiched(initial_board, initial_king_position)
    prev_king_moves_to_corner = king_moves_to_reach_corner(initial_board, initial_king_position)

    next_sandwich, next_triple_sandwich = is_king_sandwiched(current_board, current_king_position)
    next_king_moves_to_corner = king_moves_to_reach_corner(current_board, current_king_position)

    # Wyświetl wartości diagnostyczne
    # print(f"Previous sandwich: {prev_sandwich}, Previous triple sandwich: {prev_triple_sandwich}")
    # print(f"Next sandwich: {next_sandwich}, Next triple sandwich: {next_triple_sandwich}")
    # print(f"Previous king moves to corner: {prev_king_moves_to_corner}")
    # print(f"Next king moves to corner: {next_king_moves_to_corner}")

    if escapee_turn:
        prev_attacker_pawns = count_pieces(initial_board, 'A')
        next_attacker_pawns = count_pieces(current_board, 'A')
        did_win = 1 if game_result == 'escaped' else 0

        # Wyświetl wartości diagnostyczne
        # print(f"Escapee turn: Previous attacker pawns: {prev_attacker_pawns}, Next attacker pawns: {next_attacker_pawns}")
        # print(f"Escapee turn: Did win: {did_win}")

        # Obliczanie nagrody dla uciekającego
        reward = (
            # +reward
            COUNT_WEIGHT * (1 if next_attacker_pawns < prev_attacker_pawns else 0) +
            SANDWICH_WEIGHT * (1 if prev_sandwich and not next_sandwich else 0) +
            TRIPLE_SANDWICH_WEIGHT * (1 if prev_triple_sandwich and not next_triple_sandwich else 0) +
            CORNER_WEIGHT * (1 if next_king_moves_to_corner < prev_king_moves_to_corner else 0) +
            WIN_WEIGHT * did_win

            # -reward
            - SANDWICH_WEIGHT * (1 if not prev_sandwich and next_sandwich else 0)
            - TRIPLE_SANDWICH_WEIGHT * (1 if not prev_triple_sandwich and next_triple_sandwich else 0)
            - CORNER_WEIGHT * (1 if next_king_moves_to_corner > prev_king_moves_to_corner else 0)
        )

        # Wyświetl nagrodę
        print(f"Escapee turn: Reward: {reward}")

    else:  # Atakujący
        prev_defender_pawns = count_pieces(initial_board, 'D')
        next_defender_pawns = count_pieces(current_board, 'D')
        did_win = 1 if game_result == 'captured' else 0

        # Wyświetl wartości diagnostyczne
        # print(f"Attacker turn: Previous defender pawns: {prev_defender_pawns}, Next defender pawns: {next_defender_pawns}")
        # print(f"Attacker turn: Did win: {did_win}")

        # Obliczanie nagrody dla atakującego
        reward = (
            # + reward
            COUNT_WEIGHT * (1 if next_defender_pawns < prev_defender_pawns else 0) +
            SANDWICH_WEIGHT * (1 if not prev_sandwich and next_sandwich else 0) +
            TRIPLE_SANDWICH_WEIGHT * (1 if not prev_triple_sandwich and next_triple_sandwich else 0) +
            CORNER_WEIGHT * (1 if next_king_moves_to_corner > prev_king_moves_to_corner else 0) +
            WIN_WEIGHT * did_win

            # -reward
            - SANDWICH_WEIGHT * (1 if prev_sandwich and not next_sandwich else 0)
            - TRIPLE_SANDWICH_WEIGHT * (1 if prev_triple_sandwich and not next_triple_sandwich else 0)
            - CORNER_WEIGHT * (1 if next_king_moves_to_corner < prev_king_moves_to_corner else 0)

        )

        # Wyświetl nagrodę
        print(f"Attacker turn: Reward: {reward}")

    return reward
    # return 0


