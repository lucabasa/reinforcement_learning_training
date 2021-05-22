import numpy as np


class Blob:
    def __init__(self, SIZE):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        self.SIZE = SIZE
    
    
    def __str__(self):  # to print(Blob)
        return f"{self.x}, {self.y}"
    
    
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
    
    
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        4 diagonal moves
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)


    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.SIZE - 1:
            self.x = self.SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.SIZE - 1:
            self.y = self.SIZE - 1
            
