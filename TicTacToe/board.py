# Freely inspired by https://github.com/MJeremy2017/reinforcement-learning-implementation/blob/master/TicTacToe/ticTacToe.py
# More details at https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542

import numpy as np


class State:
    def __init__(self, p1, p2, rows=3, cols=3, to_win=3):
        
        self.board = np.zeros((rows, cols))
        self.rows = rows
        self.cols = cols
        
        self.p1 = p1
        self.p2 = p2
        self.to_win = to_win
        self.isEnd = False
        self.boardHash = None
        # init p1 plays first
        self.playerSymbol = 1
        
        
    # get unique hash of current board state
    def getHash(self):
        self.boardHash = str(self.board.reshape(self.cols * self.rows))
        return self.boardHash
    
    
    def availablePositions(self):
        positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i, j] == 0:
                    positions.append((i, j))  # need to be tuple
        return positions

    
    def updateState(self, position):
        self.board[position] = self.playerSymbol
        # switch to another player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1
        
        
    def winner(self):
        # row
        for i in range(self.rows):
            if sum(self.board[i, :]) == self.to_win:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -self.to_win:
                self.isEnd = True
                return -1
        # col
        for i in range(self.cols):
            if sum(self.board[:, i]) == self.to_win:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -self.to_win:
                self.isEnd = True
                return -1
        # diagonal  FIXME: only works for 3x3
        diag_sum1 = sum([self.board[i, i] for i in range(self.cols)])
        diag_sum2 = sum([self.board[i, self.cols - i - 1] for i in range(self.cols)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == self.to_win:
            self.isEnd = True
            if diag_sum1 == self.to_win or diag_sum2 == self.to_win:
                return 1
            else:
                return -1

        # tie
        # no available positions
        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0
        # not end
        self.isEnd = False
        return None
    
    # only when game ends
    def giveReward(self):
        result = self.winner()
        # backpropagate reward, uses Players methods
        if result == 1: 
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        else:
            self.p1.feedReward(0.1)  # a tie is still bad for the p1
            self.p2.feedReward(0.5)
            
            
    def showBoard(self):
        # p1: x  p2: o
        for i in range(0, self.rows):
            print('-------------')
            out = '| '
            for j in range(0, self.cols):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
        
    
    # board reset
    def reset(self):
        self.board = np.zeros((self.rows, self.cols))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1
        
        
    def _player_action(self, player):
        positions = self.availablePositions()
        pl_action = player.chooseAction(positions=positions, 
                                        current_board=self.board, 
                                        symbol=self.playerSymbol)
        # take action and upate board state
        self.updateState(pl_action)
        board_hash = self.getHash()
        player.addState(board_hash)
        
    
    def _check_win(self, human=False):
        # check board status if it is end
        win = self.winner()
        if win is not None:
            # ended with player either win or draw
            if human:
                if win == 1:
                    print(self.p1.name, "wins!")
                else:
                    print("tie!")
            else:
                self.giveReward()
                self.p1.reset()
                self.p2.reset()
            self.reset()
            return 1
        else:
            return 0

        
    def play(self, rounds=100):
        for i in range(rounds):
            if i % 1000 == 0:
                print("Rounds {}".format(i))
            while not self.isEnd:
                # Player 1
                self._player_action(self.p1)
                if self._check_win() > 0:
                    break
                # Player 2
                self._player_action(self.p2)
                if self._check_win() > 0:
                    break

    
    # play against human
    def play_human(self):
        while not self.isEnd:
            # Player 1
            self._player_action(self.p1)
            self.showBoard()
            # check board status if it is end
            if self._check_win(human=True) > 0:
                break
            # Player 2
            self._player_action(self.p2)
            self.showBoard()
            if self._check_win(human=True) > 0:
                break