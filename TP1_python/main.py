from gridworld import GridWorld1
import gridrender as gui
import numpy as np
import matplotlib.pyplot as plt
import time
from utils import v_from_q
from dynamic_programming import value_iteration, policy_iteration

################################################################################
# Dynamic programming
################################################################################

value_iteration()
policy_iteration()




env = GridWorld1

################################################################################
# Work to do: Q4
################################################################################
# here the v-function and q-function to be used for question 4
v_q4 = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.67106071, -0.99447514, 0.00000000, -0.82847001, -0.87691855,
        -0.93358351, -0.99447514]

### Compute mu0
mu0 = np.array([env.reset() for i in range(5000)])
unique, counts = np.unique(mu0, return_counts=True)
mu0 = counts/np.sum(counts)

iterates = []
# Policy right when possible and up otherwise
pol = [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3]

val = [0]*len(pol)
counter = [0]*len(pol)
Tmax = 300

for i in range(2000):
    term = False
    tot_r = 0
    state_0 = state = env.reset()
    counter[state] += 1
    t = 1
    while t < Tmax and not term:
        action = pol[state]
        nexts, reward, term = env.step(state,action)
        tot_r += reward * env.gamma**(t-1)
        state = nexts
        t += 1

    val[state_0] += tot_r
    
    iterates.append(np.sum(np.nan_to_num(mu0 * np.array(val)/(np.array(counter)))))

val = np.array(val)/(np.array(counter))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.array(iterates) - np.sum(mu0 * np.array(v_q4)))
ax.set_xlabel('Iterations')
ax.set_ylabel('Jn - Jpi')
plt.savefig('monte_carlo.pdf')

################################################################################
# Work to do: Q5
################################################################################
v_opt = [0.87691855, 0.92820033, 0.98817903, 0.00000000, 0.82369294, 0.92820033, 0.00000000, 0.77818504, 0.82369294,
         0.87691855, 0.82847001]


q = np.zeros((len(pol), 4))
counts = np.zeros((len(pol), 4))
polq5 = [[0]*len(env.state_actions[i]) for i in range(len(pol))]

eps = 0.2
qs = []
rewards = [0]

for i in range(1000):
    t = 0
    term = False
    state_0 = state = env.reset()
    rew = 0
    while t < Tmax and not term:
        rd = np.random.random()
        action = np.argmax(q[state,:]) if rd > eps else np.random.choice(env.state_actions[state])
        
        # Choose the first action available since max was 0
        while not action in env.state_actions[state]:
            action = (action + 1)%max(env.state_actions[state])
          
        counts[state, action] += 1
        nexts, reward, term = env.step(state,action)
        rew += reward
        delta = reward + env.gamma * np.max(q[nexts,:]) - q[state, action]
        
        q[state, action] += delta/counts[state, action]**0.51
        state = nexts
        t += 1
    rewards.append(rew + rewards[-1])
    qs.append(q.copy())


#gui.render_q(env, q)
### Plot the infinite norm of v^* - v_pi
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.max(np.abs(np.array([v_from_q(11,env.state_actions, 
                                         np.argmax(q, axis=1), 
                                         q) for q in qs]) - np.array(v_opt)), 
                axis=1))
ax.set_xlabel('Iterations')
ax.set_ylabel('Max norm of the gap to optimal value function')
plt.savefig('q_learning.pdf')

### Plot the mean reward over the episodes
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rewards[:100])
ax.set_xlabel('Episode')
ax.set_ylabel('Cumulated reward')
plt.savefig('rewards.pdf')