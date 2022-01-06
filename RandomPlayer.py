import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import copy


class Random_Player:

    def __init__(self, player,r,c):
        self.player = player
        self.r = r
        self.c = c

    def set_state(self, state):
        self.state = state

    def is_terminal_state(self,next_state, action):
        if self.is_winning_state(next_state, action):
            return True, "win"

        for y in range(5):
            if(next_state[0][y] == 0):
                return False, ".."

        return True, "draw"

    def out_of_bounds(self,x,y):
        return not(x>=0 and x<self.r and y>=0 and y<self.c)

    def is_winning_state(self,next_state, action):
        y = action
        x = 0
    
        for i in range(self.r):
            if next_state[i][y] != 0:
                break
            x+=1

        directions =  [ [1,1], [1,-1], [0,1], [1,0] ]
        for d in directions:
            for i in range(4):
        
                count = 0
                for j in range(i):
                    x_dash = x + (j+1)*d[0]
                    y_dash = y + (j+1)*d[1]
                    
                    if( self.out_of_bounds(x_dash,y_dash) or next_state[x_dash][y_dash] != self.player):
                        break
                    count+=1
                
                for j in range(3-i):
                    x_dash = x - (j+1)*d[0]
                    y_dash = y - (j+1)*d[1]
        
                    if( self.out_of_bounds(x_dash,y_dash) or next_state[x_dash][y_dash] != self.player):
                        break
                    count+=1

                if(count == 3):
                    return True
        
        return False


    def random_action(self):
        n = 0
        probabilites = np.ones(self.c, dtype = float)

        for i in range(self.c):
            if self.state[0][i] == 0:
                n+=1

        for i in range(self.c):
            if self.state[0][i] != 0:
                probabilites[i] = 0
            else:
                probabilites[i] /= n

        actions = np.ones(self.c)
        for i in range(self.c):
            actions[i] = i

        action = random.choices(actions, probabilites, k=1)
        return np.int64(action[0])


    def take_action(self):
        
        action = self.random_action()
        for x in range(self.r):
            if self.state[self.r-1-x][action] == 0:
                self.state[self.r-1-x][action] = self.player
                break

        isTS, result = self.is_terminal_state(self.state,action)
        return self.state, isTS, result, action
