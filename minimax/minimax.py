import copy
from conecta4 import (
    COLS,
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    ROWS,
    Conecta4,
)  # Supondo que estejam disponíveis
from tree_visualizer import (
    TreeNode,
)  # Supondo que tree_visualizer.py está no mesmo diretório

# Constantes para os papéis dos jogadores neste contexto do Minimax
# Assumindo que PLAYER_1 é sempre a IA/Jogador Maximizador para a chamada raiz de get_best_move_and_tree
AI_PLAYER_TOKEN = PLAYER_1
OPPONENT_PLAYER_TOKEN = PLAYER_2


def minimax_recursive(
    game: Conecta4,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing_player_turn: bool,
    eval_fn,
    move_made_to_reach_this_state: int | None = None,
):
    """
    Função Minimax recursiva com poda Alfa-Beta que também constrói uma árvore de decisão.

    Retorna:
        (score, TreeNode)
    """

    # Criar o TreeNode para o estado atual do jogo
    node = TreeNode(
        player_type="MAX" if is_maximizing_player_turn else "MIN",
        depth=depth,
        alpha=alpha,  # Alpha/beta *antes* de explorar os filhos deste nó
        beta=beta,
        move_info=(
            f"Col {move_made_to_reach_this_state}"
            if move_made_to_reach_this_state is not None
            else "Estado Atual"
        ),
    )

    game_result = game.get_game_result()

    # Verificações de estado terminal
    if game_result is not None:
        node.is_terminal = True
        if game_result == AI_PLAYER_TOKEN:  # IA (ex: PLAYER_1) venceu
            node.score = float("inf")
            node.notes = f"Vitória P{AI_PLAYER_TOKEN}"
        elif game_result == OPPONENT_PLAYER_TOKEN:  # Oponente (ex: PLAYER_2) venceu
            node.score = float("-inf")
            node.notes = f"Vitória P{OPPONENT_PLAYER_TOKEN}"
        elif game_result == "draw":
            node.score = 0
            node.notes = "Empate"

        # Atualiza alfa/beta do nó com seu score se terminal, para fins de inspeção.
        if node.score is not None:
            if is_maximizing_player_turn:
                node.alpha = max(
                    node.alpha if node.alpha is not None else float("-inf"), node.score
                )
            else:
                node.beta = min(
                    node.beta if node.beta is not None else float("inf"), node.score
                )
        return node.score, node

    if depth == 0:
        node.is_terminal = True
        node.score = eval_fn(
            game
        )  # A heurística deve ser do ponto de vista de game.current_player
        node.notes = "Prof. Max."
        if is_maximizing_player_turn:
            node.alpha = max(
                node.alpha if node.alpha is not None else float("-inf"), node.score
            )
        else:
            node.beta = min(
                node.beta if node.beta is not None else float("inf"), node.score
            )
        return node.score, node

    valid_moves = game.get_valid_moves()
    if (
        not valid_moves
    ):  # Deveria ser pego por "draw" antes se o tabuleiro estiver cheio
        node.is_terminal = True
        node.score = 0
        node.notes = "Sem movimentos"
        if is_maximizing_player_turn:
            node.alpha = max(
                node.alpha if node.alpha is not None else float("-inf"), node.score
            )
        else:
            node.beta = min(
                node.beta if node.beta is not None else float("inf"), node.score
            )
        return node.score, node

    if is_maximizing_player_turn:
        max_eval = float("-inf")
        for i, move_col in enumerate(valid_moves):
            next_game = copy.deepcopy(game)
            next_game.make_move(move_col)
            next_game.switch_player()  # Prepara para o turno do oponente

            eval_child, child_node = minimax_recursive(
                next_game, depth - 1, alpha, beta, False, eval_fn, move_col
            )
            node.add_child(child_node, edge_label=f"Col {move_col}")
            max_eval = max(max_eval, eval_child)
            alpha = max(alpha, eval_child)

            if beta <= alpha:
                node.notes = "Poda β"
                child_node.notes = (
                    child_node.notes + " (Corte β)" if child_node.notes else "(Corte β)"
                )
                node.was_cutoff_node = True
                for unvisited_move_col in valid_moves[i + 1 :]:
                    pruned_child = TreeNode(
                        player_type="MIN",
                        depth=depth - 1,
                        move_info=f"Col {unvisited_move_col}",
                        is_pruned=True,
                        notes="Não explorado (Poda β)",
                    )
                    node.add_child(pruned_child, edge_label=f"Col {unvisited_move_col}")
                break
        node.score = max_eval
        node.alpha = alpha  # Alpha final para este nó MAX
        return max_eval, node
    else:  # Turno do jogador Minimizador
        min_eval = float("inf")
        for i, move_col in enumerate(valid_moves):
            next_game = copy.deepcopy(game)
            next_game.make_move(move_col)
            next_game.switch_player()  # Prepara para o turno do Maximizador

            eval_child, child_node = minimax_recursive(
                next_game, depth - 1, alpha, beta, True, eval_fn, move_col
            )
            node.add_child(child_node, edge_label=f"Col {move_col}")
            min_eval = min(min_eval, eval_child)
            beta = min(beta, eval_child)

            if beta <= alpha:
                node.notes = "Poda α"
                child_node.notes = (
                    child_node.notes + " (Corte α)" if child_node.notes else "(Corte α)"
                )
                node.was_cutoff_node = True
                for unvisited_move_col in valid_moves[i + 1 :]:
                    pruned_child = TreeNode(
                        player_type="MAX",
                        depth=depth - 1,
                        move_info=f"Col {unvisited_move_col}",
                        is_pruned=True,
                        notes="Não explorado (Poda α)",
                    )
                    node.add_child(pruned_child, edge_label=f"Col {unvisited_move_col}")
                break
        node.score = min_eval
        node.beta = beta  # Beta final para este nó MIN
        return min_eval, node


def get_best_move_and_tree(game: Conecta4, depth: int, eval_fn):
    """
    Encontra o melhor movimento para o jogador atual usando Minimax e retorna o movimento e a árvore de decisão.
    Assume que game.current_player é a IA tomando a decisão.
    """
    ai_player_at_root = game.current_player
    is_ai_maximizing_at_root = ai_player_at_root == AI_PLAYER_TOKEN

    # Nó raiz para esta decisão. Seus filhos serão os resultados após cada possível 1º lance da IA.
    overall_root_node = TreeNode(
        player_type="MAX" if is_ai_maximizing_at_root else "MIN",
        depth=depth,  # Profundidade inicial da busca
        alpha=float("-inf"),
        beta=float("inf"),
        notes=f"IA (P{ai_player_at_root}) Decidindo...",
        board_state_repr="Tabuleiro Atual",  # Opcional: representação do tabuleiro
    )

    valid_moves = game.get_valid_moves()
    if not valid_moves:
        overall_root_node.notes += "\nSem movimentos válidos!"
        return None, overall_root_node

    best_move_col = valid_moves[0]  # Fallback inicial

    # Alpha e Beta para as escolhas do nó raiz (IA)
    # Estes são passados para as chamadas recursivas.
    current_root_alpha = float("-inf")
    current_root_beta = float("inf")

    if is_ai_maximizing_at_root:
        best_eval_for_ai = float("-inf")
        for move_col in valid_moves:
            next_game = copy.deepcopy(game)
            next_game.make_move(move_col)  # IA faz este movimento
            next_game.switch_player()  # Agora é o turno do oponente

            # Oponente tentará minimizar o score da IA (is_maximizing_player_turn = False)
            eval_of_ai_move, child_node = minimax_recursive(
                next_game,
                depth - 1,
                current_root_alpha,
                current_root_beta,
                False,  # Turno do oponente (MIN)
                eval_fn,
                move_col,  # Este é o movimento que a IA fez para chegar ao estado do child_node
            )
            overall_root_node.add_child(
                child_node, edge_label=f"IA joga Col {move_col}"
            )

            if eval_of_ai_move > best_eval_for_ai:
                best_eval_for_ai = eval_of_ai_move
                best_move_col = move_col

            current_root_alpha = max(current_root_alpha, eval_of_ai_move)
            # A poda alfa-beta para os filhos diretos da raiz é tratada pelo `current_root_alpha` e `current_root_beta`
            # sendo passados e atualizados. Se um ramo inicial for muito ruim, os `alpha` e `beta`
            # passados para os ramos subsequentes já estarão mais restritos.
            # A condição `if beta <= alpha` dentro de `minimax_recursive` fará a poda.

        overall_root_node.score = best_eval_for_ai
        overall_root_node.alpha = current_root_alpha
    else:  # IA está Minimizando (caso raro, se AI_PLAYER_TOKEN != game.current_player e aun assim é a "IA")
        best_eval_for_ai = float("inf")
        for move_col in valid_moves:
            next_game = copy.deepcopy(game)
            next_game.make_move(move_col)
            next_game.switch_player()

            # Oponente tentará maximizar o score da IA (is_maximizing_player_turn = True)
            eval_of_ai_move, child_node = minimax_recursive(
                next_game,
                depth - 1,
                current_root_alpha,
                current_root_beta,
                True,  # Turno do oponente (MAX)
                eval_fn,
                move_col,
            )
            overall_root_node.add_child(
                child_node, edge_label=f"IA joga Col {move_col}"
            )

            if eval_of_ai_move < best_eval_for_ai:
                best_eval_for_ai = eval_of_ai_move
                best_move_col = move_col
            current_root_beta = min(current_root_beta, eval_of_ai_move)

        overall_root_node.score = best_eval_for_ai
        overall_root_node.beta = current_root_beta

    return best_move_col, overall_root_node


# --- Função de Avaliação (adaptada do código do usuário) ---
def evaluate_board(
    game: Conecta4, weights=(1.0, 2.0, 0.5)
):  # (Peso linhas abertas, Peso trios, Peso controle centro)
    player = game.current_player  # Jogador cujo turno é neste estado
    opponent = PLAYER_1 if player == PLAYER_2 else PLAYER_2
    w1_open_lines, w2_threes, w3_center = weights

    # Função para contar "linhas abertas" (janelas de 4 onde o oponente não bloqueia e o jogador tem peças)
    def count_player_potential_lines(p_player, p_opponent):
        count = 0
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                line = [game.board[r][c + i] for i in range(4)]
                if p_opponent not in line and line.count(p_player) > 0:
                    count += 1
        # Vertical
        for c_idx in range(COLS):
            for r_idx in range(ROWS - 3):
                line = [game.board[r_idx + i][c_idx] for i in range(4)]
                if p_opponent not in line and line.count(p_player) > 0:
                    count += 1
        # Diagonal Positiva
        for r_idx in range(ROWS - 3):
            for c_idx in range(COLS - 3):
                line = [game.board[r_idx + i][c_idx + i] for i in range(4)]
                if p_opponent not in line and line.count(p_player) > 0:
                    count += 1
        # Diagonal Negativa
        for r_idx in range(3, ROWS):
            for c_idx in range(COLS - 3):
                line = [game.board[r_idx - i][c_idx + i] for i in range(4)]
                if p_opponent not in line and line.count(p_player) > 0:
                    count += 1
        return count

    # Função para contar N peças do jogador em uma janela de 4 com os demais espaços vazios
    def count_N_in_potential_line(p_player, num_pieces_player):
        count = 0
        # Horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                window = [game.board[r][c + i] for i in range(4)]
                if window.count(p_player) == num_pieces_player and window.count(
                    EMPTY
                ) == (4 - num_pieces_player):
                    count += 1
        # Vertical
        for c_idx in range(COLS):
            for r_idx in range(ROWS - 3):
                window = [game.board[r_idx + i][c_idx] for i in range(4)]
                if window.count(p_player) == num_pieces_player and window.count(
                    EMPTY
                ) == (4 - num_pieces_player):
                    count += 1
        # Diagonal Positiva
        for r_idx in range(ROWS - 3):
            for c_idx in range(COLS - 3):
                window = [game.board[r_idx + i][c_idx + i] for i in range(4)]
                if window.count(p_player) == num_pieces_player and window.count(
                    EMPTY
                ) == (4 - num_pieces_player):
                    count += 1
        # Diagonal Negativa
        for r_idx in range(3, ROWS):
            for c_idx in range(COLS - 3):
                window = [game.board[r_idx - i][c_idx + i] for i in range(4)]
                if window.count(p_player) == num_pieces_player and window.count(
                    EMPTY
                ) == (4 - num_pieces_player):
                    count += 1
        return count

    score_open_lines_player = count_player_potential_lines(player, opponent)
    score_open_lines_opponent = count_player_potential_lines(opponent, player)

    score_threes_player = count_N_in_potential_line(
        player, 3
    )  # 3 peças do jogador e 1 vazio na janela de 4
    score_threes_opponent = count_N_in_potential_line(opponent, 3)

    # Controle do centro (coluna do meio)
    center_col_index = COLS // 2
    center_control_player = sum(
        game.board[row][center_col_index] == player for row in range(ROWS)
    )
    center_control_opponent = sum(
        game.board[row][center_col_index] == opponent for row in range(ROWS)
    )

    score_player_perspective = (
        w1_open_lines * score_open_lines_player
        + w2_threes * score_threes_player
        + w3_center * center_control_player
    )

    score_opponent_perspective = (
        w1_open_lines * score_open_lines_opponent
        + w2_threes * score_threes_opponent
        + w3_center * center_control_opponent
    )
    # Retorna a pontuação da perspectiva do 'player' atual do 'game'
    return score_player_perspective - score_opponent_perspective


# --- Bloco de execução principal (exemplo) ---
if __name__ == "__main__":
    # Supondo que tree_visualizer.py e sua função visualize_decision_tree estão disponíveis
    from tree_visualizer import visualize_decision_tree

    game = Conecta4()
    game_turn_counter = 0

    AI_MAIN_PLAYER_ID = PLAYER_1  # IA é o Jogador 1
    HUMAN_PLAYER_ID = PLAYER_2  # Humano é o Jogador 2

    while True:
        game.print_board()
        game_turn_counter += 1
        current_player_for_move = game.current_player

        if current_player_for_move == AI_MAIN_PLAYER_ID:
            print(
                f"\nTurno {game_turn_counter}: IA (Jogador {AI_MAIN_PLAYER_ID}) está pensando..."
            )

            TreeNode._node_counter = (
                0  # Reseta o contador de nós para IDs consistentes por imagem
            )

            # Profundidade da IA. Aumentar profundidade aumenta muito o tempo de cálculo.
            ia_depth = 3  # Comece com 3 ou 4 para testes.

            chosen_column, decision_tree_root = get_best_move_and_tree(
                game, depth=ia_depth, eval_fn=evaluate_board
            )

            if chosen_column is not None:
                print(
                    f"IA (Jogador {AI_MAIN_PLAYER_ID}) escolheu a coluna: {chosen_column}"
                )
                if not game.make_move(chosen_column):
                    print(
                        "Erro: IA fez um movimento inválido!"
                    )  # Não deveria acontecer
                    break
            else:
                print("IA não conseguiu encontrar um movimento.")
                break

            if decision_tree_root:
                tree_filename = f"decision_tree_turn_{game_turn_counter}_P{AI_MAIN_PLAYER_ID}_depth{ia_depth}.png"
                if (
                    chosen_column is not None
                ):  # Adiciona a escolha final ao nó raiz para clareza
                    decision_tree_root.notes += (
                        f"\nMelhor Coluna Escolhida: {chosen_column}"
                    )
                visualize_decision_tree(
                    decision_tree_root, tree_filename, view_in_window=True
                )
        else:  # Turno do Jogador Humano
            print(f"\nTurno {game_turn_counter}: Sua vez (Jogador {HUMAN_PLAYER_ID})")
            valid_human_moves = game.get_valid_moves()
            if not valid_human_moves:
                print("Não há movimentos válidos para você.")
                break

            chosen_column = -1
            while True:
                try:
                    move_str = input(
                        f"Jogador {HUMAN_PLAYER_ID}, escolha a coluna ({valid_human_moves}): "
                    )
                    chosen_column = int(move_str)
                    if chosen_column in valid_human_moves:
                        break
                    else:
                        print(f"Coluna inválida. Escolha entre: {valid_human_moves}")
                except ValueError:
                    print("Entrada inválida. Por favor, insira um número de coluna.")

            if not game.make_move(chosen_column):
                print(
                    "Movimento inválido! Tente novamente."
                )  # Não deveria acontecer com validação
                game_turn_counter -= 1
                continue

        game_result = game.get_game_result()
        if game_result:
            game.print_board()
            if game_result == "draw":
                print("Resultado: Empate!")
            else:
                print(f"Resultado: Jogador {game_result} venceu!")
            break

        game.switch_player()
