import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

# Hiperparâmetros
num_episodes = 50_000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.05

# Ambiente
env = gym.make("Blackjack-v1", sab=True)  # sab=True = mais consistente com RL clássico

# Q-table: (player_sum, dealer_card, usable_ace) → action-value
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# Política epsilon-greedy
def policy(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # exploração
    return np.argmax(Q[state])  # exploração

# Treinamento
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False

    while not done:
        action = policy(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Atualiza Q-value
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + discount_factor * Q[next_state][best_next_action]
        td_delta = td_target - Q[state][action]
        Q[state][action] += learning_rate * td_delta

        state = next_state

    # Decaimento do epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

print("Treinamento completo!")

# Testando o agente treinado
def test_agent(episodes=1000):
    wins = 0
    draws = 0
    losses = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        if reward > 0:
            wins += 1
        elif reward == 0:
            draws += 1
        else:
            losses += 1

    print(f"Vitórias: {wins}, Empates: {draws}, Derrotas: {losses}")

test_agent()
