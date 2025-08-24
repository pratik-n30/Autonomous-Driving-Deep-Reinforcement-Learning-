# Autonomous-Driving-Deep-Reinforcement-Learning-

The Goal of this project is to train an vehicle to drive on its own, navigate lanes, avoid obstacles, and control its speed accordingly.

As part of my learning process, I first learnt the fundamentals of python. After gaining enough experience with Python I studied about the basics of Reinforcement Learning, Markov Decision Process, Planning by dynamic programming, value iteration and policy iteration. To implement the knowledge I learnt, I implemented Value iteration and Policy iteration in frozen lake environment.

Then I explored the algorithms for model free prediction, I realised that exploration and exploitation both are equally important and we need to balance it in an efficient way so I used the epsilon greedy algorithm. For policy evaluation, I learnt monte carlo method and temporal difference method. Combining everything is Monte Carlo Control and Sarsa Lambda algorithm for on policy learning. I also used the concept of eligibility traces for better efficiency. I also learnt Q learning for off policy.

I implemented the knowledge gained so far on the minigrid environment.

But the next problem is that, sometimes the state space could be vast and continuous. In that case we can use Neural networks and other methods to predict the value function and action value function for Q learning. 

So to correctly understand Neural Networks, I implemented it in solving the XOR.
