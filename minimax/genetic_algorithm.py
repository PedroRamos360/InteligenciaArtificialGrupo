import random
from conecta4 import Conecta4, PLAYER_1, PLAYER_2
from minimax import (
    get_best_move_and_tree,
    evaluate_board,
)

POPULATION_SIZE = 10
N_GENERATIONS = 10
MUTATION_RATE = 0.3
MUTATION_STRENGTH = 1
TOURNAMENT_SIZE = 5
MINIMAX_DEPTH = 4


def create_random_chromosome():
    return (random.uniform(0, 5), random.uniform(0, 5), random.uniform(0, 5))


def initialize_population(size):
    return [create_random_chromosome() for _ in range(size)]


def play_one_game(weights1, weights2, depth=MINIMAX_DEPTH):
    game = Conecta4()

    def eval_fn_player1(g):
        score = evaluate_board(g, weights=weights1, player1=PLAYER_1, player2=PLAYER_2)
        return score

    def eval_fn_player2(g):
        score = evaluate_board(g, weights=weights2, player1=PLAYER_1, player2=PLAYER_2)
        return score

    while True:
        current_player_for_move = game.current_player
        eval_fn_for_current_player = None

        if current_player_for_move == PLAYER_1:
            eval_fn_for_current_player = eval_fn_player1
        else:
            eval_fn_for_current_player = eval_fn_player2

        best_move, _ = get_best_move_and_tree(
            game, depth=depth, eval_fn=eval_fn_for_current_player
        )

        if best_move is None:
            return "draw"

        game.make_move(best_move)
        result = game.get_game_result()

        if result is not None:
            return result

        game.switch_player()


def calculate_fitness(chromosome_weights, population, tournament_size=TOURNAMENT_SIZE):
    wins = 0
    draws = 0
    opponents_indices = random.sample(
        range(len(population)), min(tournament_size, len(population) - 1)
    )

    for i in opponents_indices:
        opponent_weights = population[i]
        if opponent_weights == chromosome_weights:
            continue

        result1 = play_one_game(chromosome_weights, opponent_weights)
        if result1 == PLAYER_1:
            wins += 1
        elif result1 == "draw":
            draws += 0.5

        result2 = play_one_game(opponent_weights, chromosome_weights)
        if result2 == PLAYER_2:
            wins += 1
        elif result2 == "draw":
            draws += 0.5

    return wins + draws


def selection(population_with_fitness):
    tournament = random.sample(
        population_with_fitness, k=min(5, len(population_with_fitness))
    )
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0], tournament[1][0]


def crossover(parent1_weights, parent2_weights):
    point = random.randint(1, len(parent1_weights) - 1)
    child1_weights = parent1_weights[:point] + parent2_weights[point:]
    child2_weights = parent2_weights[:point] + parent1_weights[point:]
    return tuple(child1_weights), tuple(child2_weights)


def mutate(chromosome_weights, rate, strength):
    mutated_weights = list(chromosome_weights)
    for i in range(len(mutated_weights)):
        if random.random() < rate:
            change = random.uniform(-strength, strength)
            mutated_weights[i] += change
            mutated_weights[i] = max(0, mutated_weights[i])
    return tuple(mutated_weights)


def run_genetic_algorithm():
    population = initialize_population(POPULATION_SIZE)
    best_overall_chromosome = None
    best_overall_fitness = -1

    for generation in range(N_GENERATIONS):
        print(f"\n--- Generation {generation + 1}/{N_GENERATIONS} ---")

        population_with_fitness = []
        print("Calculating fitness...")
        for i, chrom_weights in enumerate(population):
            print(f"  Evaluating chromosome {i+1}/{POPULATION_SIZE}: {chrom_weights}")
            fitness = calculate_fitness(chrom_weights, population)
            population_with_fitness.append((chrom_weights, fitness))
            print(f"    Fitness: {fitness}")

        population_with_fitness.sort(key=lambda x: x[1], reverse=True)

        current_best_chromosome, current_best_fitness = population_with_fitness[0]
        print(
            f"Best in Gen {generation + 1}: Weights {current_best_chromosome}, Fitness: {current_best_fitness}"
        )

        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_chromosome = current_best_chromosome
            print(
                f"*** New Overall Best Found: {best_overall_chromosome}, Fitness: {best_overall_fitness} ***"
            )

        next_generation = []

        elitism_count = 1
        for i in range(min(elitism_count, len(population_with_fitness))):
            next_generation.append(population_with_fitness[i][0])

        while len(next_generation) < POPULATION_SIZE:
            parent1, parent2 = selection(population_with_fitness)
            offspring1, offspring2 = crossover(parent1, parent2)

            offspring1 = mutate(offspring1, MUTATION_RATE, MUTATION_STRENGTH)
            if len(next_generation) < POPULATION_SIZE:
                next_generation.append(offspring1)

            if len(next_generation) < POPULATION_SIZE:
                offspring2 = mutate(offspring2, MUTATION_RATE, MUTATION_STRENGTH)
                next_generation.append(offspring2)

        population = next_generation
        population = population[:POPULATION_SIZE]

    print("\n--- Genetic Algorithm Finished ---")
    print(f"Best weights found: {best_overall_chromosome}")
    print(f"Best fitness: {best_overall_fitness}")
    return best_overall_chromosome


if __name__ == "__main__":
    print("Starting Genetic Algorithm to find optimal Connect 4 AI weights...")
    optimal_weights = run_genetic_algorithm()
    print(f"\nRecommended weights for evaluate_board: {optimal_weights}")
