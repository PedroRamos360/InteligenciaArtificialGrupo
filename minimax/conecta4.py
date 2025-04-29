ROWS = 5
COLS = 5

PLAYER_1 = 1
PLAYER_2 = 2
EMPTY = 0


class Conecta4:
    def __init__(self):
        self.board = [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]
        self.current_player = PLAYER_1

    def print_board(self):
        for row in self.board:
            print(" ".join(str(cell) if cell != EMPTY else "." for cell in row))
        print("0 1 2 3 4\n")

    def get_valid_moves(self):
        return [col for col in range(COLS) if self.board[0][col] == EMPTY]

    def make_move(self, col):
        if col not in self.get_valid_moves():
            return False

        for row in reversed(range(ROWS)):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = self.current_player
                break
        return True

    def switch_player(self):
        self.current_player = PLAYER_1 if self.current_player == PLAYER_2 else PLAYER_2

    def is_winning_move(self, player):
        for row in range(ROWS):
            for col in range(COLS - 3):
                if all(self.board[row][col + i] == player for i in range(4)):
                    return True
        for col in range(COLS):
            for row in range(ROWS - 3):
                if all(self.board[row + i][col] == player for i in range(4)):
                    return True
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                if all(self.board[row - i][col + i] == player for i in range(4)):
                    return True
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    return True
        return False

    def is_full(self):
        return all(self.board[0][col] != EMPTY for col in range(COLS))

    def get_game_result(self):
        if self.is_winning_move(PLAYER_1):
            return PLAYER_1
        elif self.is_winning_move(PLAYER_2):
            return PLAYER_2
        elif self.is_full():
            return "draw"
        else:
            return None


if __name__ == "__main__":
    game = Conecta4()
    while True:
        game.print_board()
        print(f"Jogador {game.current_player}, escolha a coluna: ", end="")
        col = int(input())
        if not game.make_move(col):
            print("Movimento inválido. Tente novamente.")
            continue

        result = game.get_game_result()
        if result == PLAYER_1 or result == PLAYER_2:
            game.print_board()
            print(f"Jogador {result} venceu!")
            break
        elif result == "draw":
            game.print_board()
            print("Empate!")
            break

        game.switch_player()
