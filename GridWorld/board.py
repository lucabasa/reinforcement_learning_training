# Freely inspired by https://github.com/MJeremy2017/reinforcement-learning-implementation/tree/master/GridWorld
# More details here https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff


import numpy as np


class State:
    def __init__(self, rows=3, cols=4, 
                 state=None, # default ignored in the init
                 win_state=None, lose_state=None, # default ignored in the init
                 forbidden_blocks=None,
                 deterministic=True, **kwargs):
        
        if win_state is None:
            win_state = (0, cols-1)
        if lose_state is None:
            lose_state = (1, cols-1)
        if state is None:
            state = (rows-1, 0)
        if forbidden_blocks is None:
            forbidden_blocks = []
        
        self.rows = rows
        self.cols = cols
        self.win_state = win_state
        self.lose_state = lose_state
        self.state = state
        self.board = np.zeros([rows, cols])
        
        if not isinstance(forbidden_blocks, list):
            forbidden_blocks = [forbidden_blocks]
        if self.win_state in forbidden_blocks:
            forbidden_blocks = [b for b in forbidden_blocks if b != self.win_state]
        if self.lose_state in forbidden_blocks:
            forbidden_blocks = [b for b in forbidden_blocks if b != self.lose_state]
        if self.state in forbidden_blocks:
            forbidden_blocks = [b for b in forbidden_blocks if b != self.state] 
        for block in forbidden_blocks:
            self.board[block] = -1  # Forbidden block 
        self.forbidden_blocks = forbidden_blocks 
        
        self.isEnd = False
        self.deterministic = deterministic
        
        
    def giveReward(self):
        if self.state == self.win_state:
            return 1
        elif self.state == self.lose_state:
            return -1
        else:
            return 0
    
    
    def isEndFunc(self):
        if (self.state == self.win_state) or (self.state == self.lose_state):
            self.isEnd = True
    
    
    def nxtPosition(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.deterministic:
            if action == "up":
                nxtState = (self.state[0]-1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0]+1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1]-1)
            else:
                nxtState = (self.state[0], self.state[1]+1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= self.rows-1):  # board boundaries
                if (nxtState[1] >= 0) and (nxtState[1] <= self.cols-1): 
                    if nxtState not in self.forbidden_blocks:  # forbidden block
                        return nxtState
            return self.state
    
    
    def showBoard(self):
        self.board[self.state] = 1
        for i in range(0, self.rows):
            print('-----'*(self.cols-1)+'--')
            out = '| '
            for j in range(0, self.cols):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----'*(self.cols-1)+'--')



