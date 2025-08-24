import numpy as np
import matplotlib.pyplot as plt
from minigrid.wrappers import FullyObsWrapper
from simple_env import SimpleEnv
import random

def get_state(env, obs):
    """
    Combines agent position (x, y) and direction into a single state tuple.
    This is necessary because the environment's observation is a dictionary,
    but our Q-tables need a hashable state representation.
    """
    x, y = env.unwrapped.agent_pos
    direction = obs["direction"]
    return (y, x, direction)

def epsilon_greedy(Q, state, allowed_actions, epsilon):
    """
    Implements the Epsilon-Greedy policy for action selection.
    
    With probability `epsilon`, it chooses a random action (exploration).
    Otherwise, it chooses the action with the highest Q-value for the current state (exploitation).
    """
    if random.random() < epsilon:
        # Exploration: choose a random action
        return random.choice(allowed_actions)
    else:
        # Exploitation: choose the best action based on Q-values
        q_vals = [Q.get((state, a), 0.0) for a in allowed_actions]
        best_action = allowed_actions[int(np.argmax(q_vals))]
        return best_action

def compute_decay_params(episodes, epsilon_start=1.0, epsilon_end=0.05):
    """
    Computes decay parameters for epsilon and lambda.
    Epsilon decay is exponential, ensuring epsilon decreases smoothly over episodes.
    Lambda is a constant decay factor for eligibility traces.
    """
    epsilon_decay = (epsilon_end / epsilon_start) ** (1 / episodes)
    lambda_val = 1 - (1 / episodes)
    return epsilon_decay, lambda_val

def sarsa_lambda(env, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05):
    """
    SARSA(λ) is an on-policy, temporal-difference control algorithm that
    uses eligibility traces to update Q-values. It updates after every step.
    
    - On-policy: It uses the same policy (epsilon-greedy) to choose both the
      current action and the next action.
    - Eligibility traces: It updates not just the last state-action pair,
      but all recently visited pairs, scaled by a decay factor (λ).
    """
    Q = {}  # The Q-value table: Q[(state, action)]
    rewards_per_episode = []
    epsilon = epsilon_start
    epsilon_decay, lamb = compute_decay_params(episodes, epsilon_start, epsilon_end)

    for ep in range(episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        allowed_actions = [0, 1, 2]
        action = epsilon_greedy(Q, state, allowed_actions, epsilon)
        
        E = {}  # Eligibility trace table: E[(state, action)]
        total_reward = 0
        done = False
        alpha = 1.0 / np.sqrt(ep + 1) # Learning rate decay

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state(env, next_obs)
            next_action = epsilon_greedy(Q, next_state, allowed_actions, epsilon)

            total_reward += reward
            
            # Initialize Q-values and eligibility traces for new state-action pairs
            Q.setdefault((state, action), 0.0)
            Q.setdefault((next_state, next_action), 0.0)
            E.setdefault((state, action), 0.0)

            # Calculate the TD-error (delta)
            # This is the on-policy update target: reward + gamma * Q(next_state, next_action)
            delta = reward + gamma * Q[(next_state, next_action)] - Q[(state, action)]
            
            # Increment the eligibility trace for the current state-action pair
            E[(state, action)] += 1

            # Update all state-action pairs based on their eligibility trace
            for (s, a) in list(E.keys()):
                Q.setdefault((s, a), 0.0)
                Q[(s, a)] += alpha * delta * E[(s, a)]
                # Decay the eligibility trace
                E[(s, a)] *= gamma * lamb

            # Move to the next state and action
            state = next_state
            action = next_action

        rewards_per_episode.append(total_reward)
        # Decay epsilon for the next episode
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        print(f"[SARSA(λ)] Episode {ep + 1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}")

    return Q, rewards_per_episode

def q_learning(env, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05):
    """
    Q-learning is an off-policy, temporal-difference control algorithm.
    
    - Off-policy: It uses a greedy approach (max Q-value) to compute the target
      for the Q-update, while the agent's behavior (action selection) remains
      epsilon-greedy. This allows it to learn the optimal policy regardless
      of the exploration policy.
    """
    Q = {}
    rewards_per_episode = []
    epsilon = epsilon_start
    epsilon_decay, _ = compute_decay_params(episodes, epsilon_start, epsilon_end)

    for ep in range(episodes):
        obs, _ = env.reset()
        state = get_state(env, obs)
        total_reward = 0
        done = False
        alpha = 1.0 / np.sqrt(ep + 1) # Learning rate decay

        while not done:
            allowed_actions = [0, 1, 2]
            # Ensure Q-values exist for all allowed actions in the current state
            for a in allowed_actions:
                Q.setdefault((state, a), 0.0)

            action = epsilon_greedy(Q, state, allowed_actions, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state(env, next_obs)

            # Ensure Q-values exist for all allowed actions in the next state
            for a in allowed_actions:
                Q.setdefault((next_state, a), 0.0)

            # Find the maximum Q-value for the next state (the greedy choice)
            max_next_q = max([Q[(next_state, a)] for a in allowed_actions])
            
            # Update Q-value using the Bellman Optimality Equation
            # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max_a' Q(s',a') - Q(s,a)]
            Q[(state, action)] += alpha * (reward + gamma * max_next_q - Q[(state, action)])

            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        print(f"[Q-learning] Episode {ep + 1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}")

    return Q, rewards_per_episode

def monte_carlo_control(env, episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05):
    """
    Monte Carlo Control learns the optimal policy by averaging returns from
    complete episodes. It works by generating full episodes and then updating
    Q-values by working backward from the end of the episode.
    """
    Q = {}
    returns = {} # Dictionary to store all returns for each state-action pair
    rewards_per_episode = []
    epsilon = epsilon_start
    epsilon_decay, _ = compute_decay_params(episodes, epsilon_start, epsilon_end)

    for ep in range(episodes):
        obs, _ = env.reset()
        allowed_actions = [0, 1, 2]
        state = get_state(env, obs)

        episode = []
        done = False
        total_reward = 0

        # Run one full episode
        while not done:
            action = epsilon_greedy(Q, state, allowed_actions, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = get_state(env, next_obs)

            episode.append((state, action, reward))
            total_reward += reward
            state = next_state

        G = 0 # Initialize the return
        visited = set() # To ensure we only update for the first visit of each state-action pair

        # Iterate backward through the episode to calculate returns
        for t in reversed(range(len(episode))):
            state_t, action_t, reward_t = episode[t]
            # Calculate the discounted return G
            G = gamma * G + reward_t

            # Only update for the first time we visit this state-action pair in this episode
            if (state_t, action_t) not in visited:
                visited.add((state_t, action_t))
                # Add the new return to our list of returns for this pair
                returns.setdefault((state_t, action_t), []).append(G)
                # Update the Q-value to the average of all returns seen so far
                Q[(state_t, action_t)] = np.mean(returns[(state_t, action_t)])

        rewards_per_episode.append(total_reward)
        # Decay epsilon for the next episode
        epsilon = max(epsilon * epsilon_decay, epsilon_end)

        print(f"[Monte Carlo] Episode {ep + 1}, Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    return Q, rewards_per_episode

def plot_rewards(sarsa_rewards, q_learning_rewards, mc_rewards):
    """
    Plots the total rewards per episode for each algorithm to visually
    compare their learning performance.
    """
    plt.figure(figsize=(18, 4))

    plt.subplot(1, 3, 1)
    plt.plot(sarsa_rewards, color='purple')
    plt.title("SARSA(λ)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(q_learning_rewards, color='orange')
    plt.title("Q-learning")
    plt.xlabel("Episode")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(mc_rewards, color='green')
    plt.title("Monte Carlo")
    plt.xlabel("Episode")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to set up the environment and run the three RL algorithms.
    """
    episodes = 100
    # Use FullyObsWrapper to get the full state observation (position, direction)
    env = FullyObsWrapper(SimpleEnv(grid_size=6, render_mode=None))

    # Run each algorithm and store the results
    Q_sarsa, sarsa_rewards = sarsa_lambda(env, episodes=episodes)
    Q_q, q_rewards = q_learning(env, episodes=episodes)
    Q_mc, mc_rewards = monte_carlo_control(env, episodes=episodes)

    # Plot the results
    plot_rewards(sarsa_rewards, q_rewards, mc_rewards)

if __name__ == "__main__":
    main()
