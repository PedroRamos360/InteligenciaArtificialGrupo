import copy

from conecta4 import COLS, EMPTY, PLAYER_1, PLAYER_2, ROWS, Conecta4


def minimax(game, depth, alpha, beta, maximizing_player, eval_fn):
    result = game.get_game_result()
    if result == PLAYER_1:
        return float("inf") if maximizing_player else float("-inf")
    elif result == PLAYER_2:
        return float("-inf") if maximizing_player else float("inf")
    elif result == "draw":
        return 0
    elif depth == 0:
        return eval_fn(game)

    valid_moves = game.get_valid_moves()
    if maximizing_player:
        max_eval = float("-inf")
        for move in valid_moves:
            next_game = copy.deepcopy(game)
            next_game.make_move(move)
            if next_game.current_player == game.current_player:
                next_game.switch_player()
            eval = minimax(next_game, depth - 1, alpha, beta, False, eval_fn)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for move in valid_moves:
            next_game = copy.deepcopy(game)
            next_game.make_move(move)
            if next_game.current_player == game.current_player:
                next_game.switch_player()
            eval = minimax(next_game, depth - 1, alpha, beta, True, eval_fn)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def evaluate_board(game, weights=(1.0, 2.0, 0.5)):
    player = game.current_player
    opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2
    w1, w2, w3 = weights

    def count_open_lines(p):
        count = 0
        for row in range(ROWS):
            for col in range(COLS - 3):
                line = [game.board[row][col + i] for i in range(4)]
                if opponent not in line:
                    if line.count(p) > 0:
                        count += 1

        for col in range(COLS):
            for row in range(ROWS - 3):
                line = [game.board[row + i][col] for i in range(4)]
                if opponent not in line:
                    if line.count(p) > 0:
                        count += 1

        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                line = [game.board[row + i][col + i] for i in range(4)]
                if opponent not in line:
                    if line.count(p) > 0:
                        count += 1

        for row in range(3, ROWS):
            for col in range(COLS - 3):
                line = [game.board[row - i][col + i] for i in range(4)]
                if opponent not in line:
                    if line.count(p) > 0:
                        count += 1

        return count

    def count_triples(p):
        count = 0
        for row in range(ROWS):
            for col in range(COLS - 2):
                line = [game.board[row][col + i] for i in range(3)]
                if line.count(p) == 3 and EMPTY in line:
                    count += 1

        for col in range(COLS):
            for row in range(ROWS - 2):
                line = [game.board[row + i][col] for i in range(3)]
                if line.count(p) == 3 and EMPTY in line:
                    count += 1

        for row in range(ROWS - 2):
            for col in range(COLS - 2):
                line = [game.board[row + i][col + i] for i in range(3)]
                if line.count(p) == 3 and EMPTY in line:
                    count += 1

        for row in range(2, ROWS):
            for col in range(COLS - 2):
                line = [game.board[row - i][col + i] for i in range(3)]
                if line.count(p) == 3 and EMPTY in line:
                    count += 1

        return count

    def control_center(p):
        center_cols = [2]
        return sum(
            game.board[row][col] == p for row in range(ROWS) for col in center_cols
        )

    score_player = (
        w1 * count_open_lines(player)
        + w2 * count_triples(player)
        + w3 * control_center(player)
    )

    score_opponent = (
        w1 * count_open_lines(opponent)
        + w2 * count_triples(opponent)
        + w3 * control_center(opponent)
    )

    return score_player - score_opponent


def best_move(game, depth, eval_fn):
    best_score = float("-inf")
    best_col = None
    for move in game.get_valid_moves():
        next_game = copy.deepcopy(game)
        next_game.make_move(move)
        if next_game.current_player == game.current_player:
            next_game.switch_player()
        score = minimax(
            next_game, depth - 1, float("-inf"), float("inf"), False, eval_fn
        )
        if score > best_score:
            best_score = score
            best_col = move
    if best_col is None:
        return game.get_valid_moves()[0]
    return best_col


if __name__ == "__main__":
    game = Conecta4()
    while True:
        game.print_board()

        if game.current_player == PLAYER_1:
            move = best_move(game, 4, evaluate_board)
            print(f"Agente escolheu a coluna: {move}")
        else:
            move = int(input("Jogador, escolha a coluna: "))

        if not game.make_move(move):
            print("Movimento inválido!")
            continue

        result = game.get_game_result()
        if result:
            game.print_board()
            print(
                "Resultado:",
                "Empate!" if result == "draw" else f"Jogador {result} venceu!",
            )
            break

        game.switch_player()
