# Autonomous-Driving-Deep-Reinforcement-Learning-

My project goal is to train a vehicle to drive autonomously by navigating lanes, avoiding obstacles, and controlling its speed appropriately.

To build the necessary skills, I began by learning the fundamentals of Python. Once I had a solid foundation, I studied the principles of Reinforcement Learning, including Markov Decision Processes (MDPs), planning via dynamic programming, and both value and policy iteration. To put this theory into practice, I successfully implemented value and policy iteration in the Frozen Lake environment.

Next, I explored model-free algorithms. I learned about the critical trade-off between exploration and exploitation and implemented the epsilon-greedy strategy to manage it. For policy evaluation, I studied Monte Carlo and Temporal Difference (TD) methods. I then combined these concepts to implement on-policy control algorithms like Monte Carlo Control and SARSA(λ), using eligibility traces for better efficiency. I also implemented the off-policy algorithm, Q-learning. I applied all of these model-free techniques in the Minigrid environment to solidify my understanding.

However, I recognized that these tabular methods are not suitable for problems with vast or continuous state spaces. This led me to function approximation, where neural networks can be used to estimate the value function or action-value (Q) function.

To build a foundational understanding of this new topic, I started by implementing a neural network to solve the classic XOR problem.
