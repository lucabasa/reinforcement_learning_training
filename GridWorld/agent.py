# Freely inspired by https://github.com/MJeremy2017/reinforcement-learning-implementation/tree/master/GridWorld
# More details here https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff

from GridWorld.board import State

import numpy as np

class Agent:
    def __init__(self, *, state=None, lr=0.2, exp_rate=0.3, verbose=False):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        if state is None:
            state = State()
        self.State = state
        self.isEnd = self.State.isEnd
        self.lr = lr
        self.exp_rate = exp_rate
        self.orig_state = state
        self.verbose = verbose
        
        # initial state reward
        self.state_values = {}
        for i in range(state.rows):  
            for j in range(state.cols):
                self.state_values[(i, j)] = 0  # init_reward[i, j]
    
    
    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""
        
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                # if the action is deterministic
                nxt_reward = self.state_values[self.State.nxtPosition(a)]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        if action == '':
            action = np.random.choice(self.actions)
        return action
    
    
    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        st_dict = self.State.__dict__.copy()
        st_dict['state'] = position
        return State(**st_dict)    
    
    
    def reset(self):
        self.states = []
        self.State = self.orig_state
        self.isEnd = self.State.isEnd
        
        
    def _backpropagate(self):
        # back propagate
        reward = self.State.giveReward()
        # explicitly assign end state to reward values
        self.state_values[self.State.state] = reward
        if self.verbose:
            print("Game End Reward", reward)
        for s in reversed(self.states):
            reward = self.state_values[s] + self.lr * (reward - self.state_values[s])
            self.state_values[s] = round(reward, 3)
        self.reset()
        
    
    def _find_solution(self):
        action = self.chooseAction()
        # append trace
        self.states.append(self.State.nxtPosition(action))
        if self.verbose:
            print("current position {} action {}".format(self.State.state, action))
        # by taking the action, it reaches the next state
        self.State = self.takeAction(action)
        # mark is end
        self.State.isEndFunc()
        if self.verbose:
            print("nxt state", self.State.state)
            print("---------------------")
        self.isEnd = self.State.isEnd
    
    
    def train(self, rounds=10):
        i = 0
        while i < rounds:
            # to the end of game back propagate reward
            if self.State.isEnd:
                self._backpropagate()
                i += 1
            else:
                self._find_solution()
                
        self.reset()
    
    
    def play(self, max_steps=100):
        # We don't need it to explore anymore
        self.exp_rate_bak = self.exp_rate
        self.exp_rate = 0
        self.verbose_bak = self.verbose
        self.verbose = True
        i = 0
        while i < max_steps:
            if self.State.isEnd:
                print(f'Solution found in {i} steps')
                self.exp_rate = self.exp_rate_bak
                self.verbose = self.verbose_bak
                self.reset()
                break
            else:
                self._find_solution()
                i += 1
    
    
    def showValues(self):
        for i in range(0, self.State.rows):
            print('----------------------------------')
            out = '| '
            for j in range(0, self.State.cols):
                out += str(self.state_values[(i, j)]) + ' | '
            print(out)
        print('----------------------------------')
        


class Agent_Q(Agent):
    def __init__(self, *, state=None, decay_gamma=0.9, lr=0.2, exp_rate=0.3, verbose=False):
        super().__init__()
        if state is None:
            state = State(deterministic=False)
        self.State = state
        self.decay_gamma = decay_gamma
        
        # initial Q values
        self.Q_values = {}
        for i in range(self.State.rows):
            for j in range(self.State.cols):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0  # Q value is a dict of dict
    
    
    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = 0
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
            # print("current pos: {}, greedy aciton: {}".format(self.State.state, action))
        if action == '':
            action = np.random.choice(self.actions)
        return action
    
    
    def _backpropagate(self):
        # back propagate
        reward = self.State.giveReward()
        for a in self.actions:
            self.Q_values[self.State.state][a] = reward
        if self.verbose:
            print("Game End Reward", reward)
        for s in reversed(self.states):
            current_q_value = self.Q_values[s[0]][s[1]]
            reward = current_q_value + self.lr * (self.decay_gamma * reward - current_q_value)
            self.Q_values[s[0]][s[1]] = round(reward, 3)
        self.reset()
        
    
    def _find_solution(self):
        action = self.chooseAction()
        # append trace
        self.states.append([(self.State.state), action])
        if self.verbose:
            print("current position {} action {}".format(self.State.state, action))
        # by taking the action, it reaches the next state
        self.State = self.takeAction(action)
        # mark is end
        self.State.isEndFunc()
        if self.verbose:
            print("nxt state", self.State.state)
            print("---------------------")
        self.isEnd = self.State.isEnd
        
    
    def showValues(self):
        for pos in self.Q_values.keys():
            print(pos, self.Q_values[pos])
