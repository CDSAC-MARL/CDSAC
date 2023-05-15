# Scalable Primal-Dual Actor-Critic Method for Safe Multi-Agent Reinforcement Learning with General Utilities (2023)
Authors:

## Introduction
There are three experiments in this repository. The experiments are conducted in a decentralized multi-agent setting
where each agent maintains a separate set of Q-function and policy network. Notably, each agent only has access to
the observation/action/Q-function of other agents that are within its kappa-hop neighborhood. We train the agents
so that they satisfy some safety constraints as defined by occupancy measures.

**A detailed description of the environments and training results can be found in
this PDF: [./ICML23_rebuttal.pdf](./ICML23_rebuttal.pdf).

We highlight the key characteristics of each experiment here:
1. **Binary message passing**: toy example where agents form a 1D linear interation graph and needs to pass a state from
right to left. The environment code is ./envs/linemsg.py and the corresponding training code is main_synthetic.py.
![Synthetic experiment on N agents passing a message from right to left. Reward depends on the agent's own action
and the next agent's state.](./readme_images/synthetic.png)

2. **Pistonball**: a modification of [the original Pistonball environment](https://pettingzoo.farama.org/environments/butterfly/pistonball/)
to highlight the decentralized MARL with safety constraint setting. In this _physics-based_ cooperative game, each agent
is a piston that can adjust its vertical position in order to move a ball to the left. Each agent only has access
to the piston heights in its kappa-hop neighborhood and can only access the ball's information if the ball
enters the neighborhood. We add an entropy constraint based on the occupancy measures. The environment code is
in ./envs/pistonball and the corresponding training code is main_piston.py.

3. **Wireless Communication**: the agents form a _2D grid_ and there is an access point in the middle of every four agents.
The goal is for each agent to transmit a package before the package deadline to a nearby access point.
If an access point receives more than one package at a single time step, all packages will fail to be transmitted.
As there are more agents than access points, the agents must learn to make sacrifices in order to maximize
the overall number of successful package transmissions. We enforce an _apprenticeship learning_ constraint
by encouraging more deterministic actions from the agents in order to improve their coordination.
The environment code is in ./envs/wireless_comm.py and the corresponding training code is main_wireless.py.

## To run:

1. Install required packages.
   ```bash
   pip install wandb
   pip install SuperSuit==3.6.0
   pip install pettingzoo==1.22.0
   pip install torch==1.13.1
   pip install opencv-python
   ```

2. Start training. See the beginning of '__main__' for a list of arguments it takes.

   ```bash
   python3 main_piston.py
   ```
3. To sync the results via wandb:

   ```bash
   python3 main_piston.py  --track
   ```
4. To visualize the outputs:

   ```bash
   python3 main_piston.py  --track --debug
   ```

## Network Architecture
![Network architecture. Every agent trains a different set of weights and can only communicate
with their neighbors in a decentralized manner.](./readme_images/alg_flow.png)

## Experiment results
### Binary message passing
![How the magnitude of the RHS of the constraint affects each agent's likelihood of violating the
constraint.](./readme_images/constraint_rhs_value_constraint.png)
How the magnitude of the RHS of the constraint affects each agent's likelihood of violating the
constraint.

### Pistonball
![Visualization of three different stages in Pistonball when
executing the learned policy.](./readme_images/Figure2-piston-kappa=3.gif)
In this figure, agents' positions are initialized randomly at the beginning. To facilitate the
ball’s leftward movement, agent 1 must move upwards, while agent 2 should move downwards. This is
confirmed by a high upward probability of agent 1 and a high downward probability of agent 2.
Later in the process, agents 1 − 4 have created a slope for the ball to move leftward rapidly.
After the ball passes, we can see that the upward probabilities of agents 1 − 4 are very close to one,
meaning that they move upwards to eliminate the possibility of the ball moving back to the right. However, agents 8 − 10 still obstruct the
ball’s path, as they have not detected the arrival of the ball due to the limited communication range
and move mostly randomly to satisfy the entropy constraint. Finally, we observe that
when the ball approaches, the downward probabilities of agents 9 − 10 become one, and the upward
probabilities of agents 5 − 8 also increase to one.

![Illustration of the benefit of having a relatively-larger communication range (kappa = 2). The agents on
the right make a sacrifice by intentionally raised the ball all the way up to provide more flexibility
for agents on their left..](./readme_images/Figure3-piston-kappa=2.gif)
In this figure, we observe that agents 1 − 3 decide to move the ball all the way up. Despite incurring a time penalty
for themselves, this provides more flexibility for agent 4, allowing more time to take random actions in order to
satisfy the safety constraint. This happens because during training, the proposed algorithm performs policy update
by directly incorporating the Q-functions of all agents in its $\kappa$-hop neighborhood. This encourages the agents
to take actions that might be suboptimal for the agent's individual reward, but beneficial for the overall objective.

## Wireless Communication
![Wireless Communication when eta = 100 and rhs = 0.3.](./readme_images/Figure5-wireless.gif)
In the Wireless Communication environment, explicitly encouraging a deterministic policy allows other agents to better
predict the agent's next moves and thus avoid collisions. In this figure, we see that due to the added apprenticeship
learning constraint, the agents learn to take more deterministic actions so that their actions are more predictable
for their neighbors. Remarkably, the agents are able to collaboratively identify a plan so that each access point
is only used by one agent in order to avoid collision. Being able to achieve this requires much collaboration,
coordination, and logical reasoning among the agents.

