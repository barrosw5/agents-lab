"""
Agentes Autónomos 2026/27 — Reinforcement Learning
Exercício 1, alíneas a) a f): ambiente, random walk e resultados de referência
"""

import random
import time
import statistics
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Grelha 10x10, estados de 1 a 100 (esquerda->direita, cima->baixo).
# Paredes só à volta da grelha.
# ---------------------------------------------------------------

ESTADO_INICIAL = 1
ESTADO_OBJETIVO = 100
MAX_PASSOS = 1000
N_TESTES = 30
ACOES = ["up", "down", "left", "right"]
DESLOCAMENTO = {"up": -10, "down": +10, "left": -1, "right": +1}


# auxiliar
def possible_actions(estado):
    acoes = []
    if estado > 10:  # não está na 1ª linha
        acoes.append("up")
    if estado <= 90:  # não está na última linha
        acoes.append("down")
    if estado % 10 != 1:  # não está na coluna da esquerda
        acoes.append("left")
    if estado % 10 != 0:  # não está na coluna da direita
        acoes.append("right")
    return acoes


# a) Funcao de transicao s' = f(s, a)
def state_transition(estado, acao):
    if acao in possible_actions(estado):
        return estado + DESLOCAMENTO[acao]
    return estado


# b) Funcao de recompensa
def reward(estado):
    if estado == ESTADO_OBJETIVO:
        return 100
    else:
        return 0


# c) Escolha aleatoria de acao
def next_action_random(estado):
    return random.choice(possible_actions(estado))


# d) correr um episodio
def run_episode():
    position = ESTADO_INICIAL
    rew = 0
    num = 0

    start = time.perf_counter()
    while position != ESTADO_OBJETIVO and num < MAX_PASSOS:
        position = state_transition(position, next_action_random(position))
        num += 1
        rew += reward(position)
    end = time.perf_counter()
    tempo_levado = end - start
    return num, rew, tempo_levado


if __name__ == "__main__":
    # e) 30 episódios, cada um com a sua seed (seeds 0 a 29)
    rewards = []  # recompensa média por passo
    nsteps = []  # passos até ao objetivo (1000 se não chegou)
    times = []  # tempo de execução (s)

    for seed in range(N_TESTES):
        random.seed(seed)
        eps_nsteps, eps_reward, eps_execution_time = run_episode()
        rewards.append(eps_reward / eps_nsteps)
        nsteps.append(eps_nsteps)
        times.append(eps_execution_time)

    data = [rewards, nsteps, times]
    titles = ["Mean reward per step", "Number of steps", "Execution time (s)"]

    for nome, valores in zip(titles, data):
        print(
            f"{nome}: média = {statistics.mean(valores):.5f}, "
            f"desvio padrão = {statistics.stdev(valores):.5f}"
        )

    chegaram = sum(1 for n in nsteps if n < MAX_PASSOS)
    print(f"Episódios que chegaram ao objetivo: {chegaram}/{N_TESTES}")

    # f) boxplots verticais, um por medida
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, var, title in zip(axes, data, titles):
        ax.boxplot(var)
        ax.set_title(title)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    plt.show()
