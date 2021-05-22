# Based on this https://pythonprogramming.net/q-learning-reinforcement-learning-python-tutorial/
# and this https://www.youtube.com/watch?v=yMk_XtIEzH8

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
env.reset()  # necessary

print(env.observation_space.high)  # you might not know these
print(env.observation_space.low)  # but we do

# to make the state space smaller
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

def get_descrete_state(state):
    discrete = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete.astype(int))

# create Q-Table - 20*20*3 of size
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_reward = []
agg_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important is the future reward with respect to the current one
EPISODES = 2000

epsilon = 0.5 # exploration rate
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2 
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

SHOW_EVERY = 500 # every 2000 episodes, show where are we

for episode in range(EPISODES):
    episode_reward = 0
    
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    
    done = False
    discrete_state = get_descrete_state(env.reset())
    
    while not done:
        
        # exploration/exploitation
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)
            
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        
        # the values are going to be position and velocity
        new_discrete_state = get_descrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            # Bellman equation
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            # update the table for the state after the action is taken
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print(f'Done on episode {episode}')
            q_table[discrete_state + (action, )] = 0  # no punishment if you win

        # update the state for the next loop
        discrete_state = new_discrete_state
        
    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
            
    ep_reward.append(episode_reward)
    
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_reward[-SHOW_EVERY:]) / len(ep_reward[-SHOW_EVERY:])
        agg_ep_rewards['ep'].append(episode)
        agg_ep_rewards['avg'].append(average_reward)
        agg_ep_rewards['max'].append(max(ep_reward[-SHOW_EVERY:]))
        agg_ep_rewards['min'].append(min(ep_reward[-SHOW_EVERY:]))
        print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
        

env.close()

plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['avg'], label='avg')
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['max'], label='max')
plt.plot(agg_ep_rewards['ep'], agg_ep_rewards['min'], label='min')
plt.legend()
plt.show()
