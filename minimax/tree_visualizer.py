import pydotplus
import os
import platform
import subprocess


class TreeNode:
    _node_counter = 0

    @classmethod
    def _get_simple_node_id(cls):
        cls._node_counter += 1
        return f"N{cls._node_counter}"

    def __init__(
        self,
        board_state_repr=None,
        score=None,
        alpha=None,
        beta=None,
        player_type=None,
        move_info="",
        is_pruned=False,
        is_terminal=False,
        depth=None,
        notes="",
        children=None,
        was_cutoff_node=False,
    ):
        self.id = self._get_simple_node_id()
        self.board_state_repr = board_state_repr if board_state_repr else self.id
        self.score = score
        self.alpha = alpha
        self.beta = beta
        self.player_type = player_type
        self.move_info = move_info
        self.is_pruned = is_pruned
        self.was_cutoff_node = was_cutoff_node
        self.is_terminal = is_terminal
        self.depth = depth
        self.notes = notes
        self.children = children if children is not None else []
        self.parent_edge_label = ""

    def add_child(self, child_node, edge_label=""):
        child_node.parent_edge_label = edge_label
        self.children.append(child_node)

    def __repr__(self):
        return (
            f"Node({self.id}, Mv:{self.move_info}, Scr:{self.score}, α:{self.alpha}, "
            f"β:{self.beta}, Pruned:{self.is_pruned}, Cutoff:{self.was_cutoff_node})"
        )


def _add_nodes_edges(graph, tree_node):
    if tree_node is None:
        return

    label_parts = [f"ID: {tree_node.id}"]
    if tree_node.move_info and tree_node.depth is not None and tree_node.depth > 0:
        label_parts.append(f"Mv: {tree_node.move_info}")
    if tree_node.player_type:
        label_parts.append(f"Tipo: {tree_node.player_type}")
    if tree_node.score is not None:
        score_val = (
            f"{tree_node.score:.1f}"
            if isinstance(tree_node.score, float)
            and tree_node.score not in [float("inf"), float("-inf")]
            else str(tree_node.score)
        )
        if tree_node.score == float("inf"):
            score_val = "+∞"
        if tree_node.score == float("-inf"):
            score_val = "-∞"
        label_parts.append(f"Score: {score_val}")
    if tree_node.alpha is not None:
        alpha_val = (
            f"{tree_node.alpha:.1f}"
            if isinstance(tree_node.alpha, float)
            and tree_node.alpha not in [float("inf"), float("-inf")]
            else str(tree_node.alpha)
        )
        if tree_node.alpha == float("inf"):
            alpha_val = "+∞"
        if tree_node.alpha == float("-inf"):
            alpha_val = "-∞"
        label_parts.append(f"α: {alpha_val}")
    if tree_node.beta is not None:
        beta_val = (
            f"{tree_node.beta:.1f}"
            if isinstance(tree_node.beta, float)
            and tree_node.beta not in [float("inf"), float("-inf")]
            else str(tree_node.beta)
        )
        if tree_node.beta == float("inf"):
            beta_val = "+∞"
        if tree_node.beta == float("-inf"):
            beta_val = "-∞"
        label_parts.append(f"β: {beta_val}")
    if tree_node.depth is not None:
        label_parts.append(f"Prof: {tree_node.depth}")
    if tree_node.notes:
        label_parts.append(tree_node.notes)

    node_label = "\n".join(label_parts)
    node_shape = "box"
    fillcolor = "lightgray"
    fontcolor = "black"

    if tree_node.player_type == "MAX":
        fillcolor = "lightblue"
    elif tree_node.player_type == "MIN":
        fillcolor = "lightpink"

    if tree_node.is_terminal:
        node_shape = "ellipse"
        fillcolor = "khaki"
        if "Vitória P1" in tree_node.notes or "Vitória P2" in tree_node.notes:
            fillcolor = (
                "mediumseagreen"
                if (
                    ("P1" in tree_node.notes and tree_node.score == float("inf"))
                    or ("P2" in tree_node.notes and tree_node.score == float("inf"))
                )
                else "salmon"
            )
        if "Empate" in tree_node.notes:
            fillcolor = "lightgoldenrodyellow"

    if tree_node.is_pruned:
        node_label += "\n(RAMO PODADO)"
        fillcolor = "gray"
        fontcolor = "white"
    elif tree_node.was_cutoff_node:
        node_label += "\n(NÓ DE CORTE)"

    node_pydot = pydotplus.Node(
        tree_node.id,
        label=node_label,
        style="filled",
        shape=node_shape,
        fillcolor=fillcolor,
        fontcolor=fontcolor,
    )
    graph.add_node(node_pydot)

    for child in tree_node.children:
        edge_color = "black"
        edge_style = "solid"
        if child.is_pruned:
            edge_style = "dashed"
            edge_color = "gray"
        edge_label_str = (
            child.parent_edge_label if child.parent_edge_label else child.move_info
        )
        edge = pydotplus.Edge(
            tree_node.id,
            child.id,
            label=edge_label_str,
            color=edge_color,
            style=edge_style,
        )
        graph.add_edge(edge)
        _add_nodes_edges(graph, child)


def visualize_decision_tree(
    root_decision_node, output_filename="decision_tree.png", view_in_window=False
):
    if not root_decision_node:
        print("Nó raiz não fornecido para visualização.")
        return

    TreeNode._node_counter = 0

    graph = pydotplus.Dot("DecisionTree", graph_type="digraph", rankdir="TB")
    graph.set_node_defaults(fontname="Arial", fontsize="10")
    graph.set_edge_defaults(fontname="Arial", fontsize="8")

    _add_nodes_edges(graph, root_decision_node)

    try:
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        graph.write_png(output_filename)
        print(f"Árvore de decisão salva em {output_filename}")

        if view_in_window:
            try:
                current_system = platform.system()
                if current_system == "Windows":
                    subprocess.Popen(
                        ["start", "", os.path.abspath(output_filename)], shell=True
                    )
                elif current_system == "Darwin":
                    subprocess.Popen(["open", os.path.abspath(output_filename)])
                else:
                    subprocess.Popen(["xdg-open", os.path.abspath(output_filename)])
                print(f"Tentando abrir {output_filename} no visualizador padrão...")
            except FileNotFoundError:
                print(
                    f"Comando para abrir arquivo não encontrado. Não foi possível abrir {output_filename} automaticamente."
                )
            except Exception as e_open:
                print(f"Não foi possível abrir o arquivo automaticamente: {e_open}")
                print(
                    f"Você pode abrir {os.path.abspath(output_filename)} manualmente."
                )

    except pydotplus.graphviz.InvocationException as e_graphviz:
        print(
            f"Erro crítico: Falha ao invocar o Graphviz (o executável 'dot' não foi encontrado ou falhou)."
        )
        print(f"Detalhes: {e_graphviz}")
        print(
            "Verifique se o Graphviz está instalado CORRETAMENTE e se o diretório 'bin' do Graphviz está no PATH do sistema."
        )
        print(
            "  - Windows: Baixe de graphviz.org e adicione ao PATH (ex: C:\\Program Files\\Graphviz\\bin)."
        )
        print("  - macOS (via Homebrew): 'brew install graphviz'")
        print("  - Linux (via apt): 'sudo apt-get install graphviz'")
        print("Sem o Graphviz, a imagem da árvore não pode ser gerada.")
    except Exception as e:
        print(f"Erro ao gerar ou visualizar o gráfico: {e}")
