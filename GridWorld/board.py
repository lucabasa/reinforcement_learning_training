# Freely inspired by https://github.com/MJeremy2017/reinforcement-learning-implementation/tree/master/GridWorld
# More details here https://towardsdatascience.com/reinforcement-learning-implement-grid-world-from-scratch-c5963765ebff


import numpy as np


class State:
    def __init__(self, BOARD_ROWS=3, BOARD_COLS=4, 
                 state=(0, 0), # default ignored in the init
                 WIN_STATE=(0, 3), LOSE_STATE=(1, 3), # default ignored in the init
                 FORBIDDEN_BLOCKS=(1, 1),
                 DETERMINISTIC=True):
        
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.rows = BOARD_ROWS
        self.cols = BOARD_COLS
        self.win_state = (0, BOARD_COLS-1)
        self.lose_state = (1, BOARD_COLS-1)
        self.state = (BOARD_ROWS-1, 0)
        
        if not isinstance(FORBIDDEN_BLOCKS, list):
            FORBIDDEN_BLOCKS = [FORBIDDEN_BLOCKS]
        if self.win_state in FORBIDDEN_BLOCKS:
            FORBIDDEN_BLOCKS = [b for b in FORBIDDEN_BLOCKS if b != self.win_state]
        if self.lose_state in FORBIDDEN_BLOCKS:
            FORBIDDEN_BLOCKS = [b for b in FORBIDDEN_BLOCKS if b != self.lose_state]
        if self.state in FORBIDDEN_BLOCKS:
            FORBIDDEN_BLOCKS = [b for b in FORBIDDEN_BLOCKS if b != self.state] 
        for block in FORBIDDEN_BLOCKS:
            self.board[block] = -1  # Forbidden block 
        self.forbidden_blocks = FORBIDDEN_BLOCKS 
        
        self.isEnd = False
        self.determine = DETERMINISTIC
        
        
    
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
        if self.determine:
            if action == "up":
                nxtState = (self.state[0]-1, self.state[1])
            elif action == "down":
                nxtState = (self.state[0]+1, self.state[1])
            elif action == "left":
                nxtState = (self.state[0], self.state[1]-1)
            else:
                nxtState = (self.state[0], self.state[1]+1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= 2):  # board boundaries
                if (nxtState[1] >= 0) and (nxtState[1] <= 3): 
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



