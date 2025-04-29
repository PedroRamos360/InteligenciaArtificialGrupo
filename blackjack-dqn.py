import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Hyperparâmetros
EPISODES = 10000#50000
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 1000

env = gym.make("Blackjack-v1", sab=True)

# Processamento de estado: converte pra vetor [player_sum, dealer_card, usable_ace]
def preprocess(state):
    return torch.tensor([state[0], state[1], int(state[2])], dtype=torch.float32)

# Rede Neural simples
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 ações: hit ou stick
        )

    def forward(self, x):
        return self.fc(x)

# Instâncias
policy_net = DQN()
target_net = DQN()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = EPSILON_START
steps = 0

# Função pra escolher ação (epsilon-greedy)
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1])
    with torch.no_grad():
        return policy_net(preprocess(state)).argmax().item()

# Função de treino
def train():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack([preprocess(s) for s in states])
    next_states = torch.stack([preprocess(s) for s in next_states])
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Q(s, a)
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

    # Q(s', a') usando target network
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        targets = rewards + GAMMA * max_next_q * (1 - dones)

    loss = nn.MSELoss()(q_values, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Loop de treino
for episode in range(EPISODES):
    state, _ = env.reset()
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))
        state = next_state
        train()
        steps += 1

        # Atualiza target network periodicamente
        if steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % 1000 == 0:
        print(f"Episode {episode}, epsilon {epsilon:.4f}")

# Testando
def test_agent(n=1000):
    wins, draws, losses = 0, 0, 0
    for _ in range(n):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(preprocess(state)).argmax().item()
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
