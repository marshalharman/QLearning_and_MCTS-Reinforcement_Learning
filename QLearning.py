import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import copy
from RandomPlayer import *

class Q_Learning:

    def __init__(self, player, alpha, discount_factor, epsilon, r, c):
        self.previous_state = None
        self.previous_action = None
        self.player = player
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.game_status = "running"
        self.r = r
        self.c = c
        self.total_rewards = 0

    
    def set_Qvalues(self, q_values):
        self.q_values = q_values

    def set_state(self, state):
        self.state = state



    def is_terminal_state(self,next_state, action):

        if self.is_winning_state(next_state, action):
            return True, "win"

        for y in range(self.c):
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


    def epislon_greedy_policy(self, next_q):
        n = 0
        greedy_action_value = -math.inf
        greedy_action_indx = -1
        
        actions_list = []
        greedy_actions = []

        for i in range(self.c):
            if next_q[i] != -math.inf:
                n += 1
                actions_list.append(i)

        greedy_action_value = max(next_q)

        actions = np.ones(self.c)

        for i in range(self.c):
            actions[i] = i
            if greedy_action_value == next_q[i]:
                greedy_actions.append(i)

        e = random.uniform(0,1)
        
        if e < self.epsilon:
            action =  random.choices(actions_list, k=1)
            action = np.int64(action[0])

        else:
            action =  random.choices(greedy_actions, k=1)
            action = np.int64(action[0])
            
        return action



    def mirror_state_action(self, curr_state, action):
        str_mirror_state_action = ""
        for m in range(self.r):
            for n in range(self.c):
                str_mirror_state_action += str(curr_state[m][self.c-1-n])
        str_mirror_state_action += str( self.c - 1 - action)

        return str_mirror_state_action



    def take_action(self):      
        

        ## Find q_values of state,actions from current state

        str_curr_state = ""
        for m in range(self.r):
            for n in range(self.c):
                str_curr_state += str(self.state[m][n])
            
        max_q_sDash_a = -math.inf

        curr_q_vals = []
        for i in range(self.c):
            if self.state[0][i] == 0:
                str_curr_state_action = str_curr_state + str(i)

                
                if str_curr_state_action not in self.q_values:
                    str_mirror_state_action = self.mirror_state_action(self.state,i)
                    if  str_mirror_state_action not in self.q_values:
                        self.q_values[str_mirror_state_action] = 0
                    
                    self.q_values[str_curr_state_action] = self.q_values[str_mirror_state_action] 


                curr_q_vals.append(self.q_values[str_curr_state_action])
                if max_q_sDash_a < self.q_values[str_curr_state_action]:
                    max_q_sDash_a = self.q_values[str_curr_state_action]
            else:
                curr_q_vals.append(-math.inf)





        ## update q values of prev move
        str_prev_state_action = ""

        if self.previous_state != None:
            for m in range(self.r):
                for n in range(self.c):
                    str_prev_state_action += str(self.previous_state[m][n])
            
            str_prev_state_action += str(self.previous_action)

            R = -1
            if self.game_status != "running":

                next_q = np.zeros(self.c) 
                if self.game_status == "loss":
                    R = -50
                elif self.game_status == "draw":
                    R = -10
                else:
                    R = 50

                q_s_a = self.q_values[str_prev_state_action]
                q_s_a += self.alpha*(R - q_s_a )
                self.total_rewards += R
                self.q_values[str_prev_state_action] = q_s_a
                return self.state, True 


            q_s_a = self.q_values[str_prev_state_action]
            q_s_a += self.alpha*(R + self.discount_factor*max_q_sDash_a - q_s_a )
            self.q_values[str_prev_state_action] = q_s_a
            str_mirror_prev_state_action = self.mirror_state_action(self.previous_state,self.previous_action)
            self.q_values[str_mirror_prev_state_action] = q_s_a
            self.total_rewards += R


        temp = []
        for x in range(self.r):
            row = []
            for y in range(self.c):
                row.append(self.state[x][y])
            temp.append(row)

        self.previous_state = temp



        action = self.epislon_greedy_policy(curr_q_vals)
        self.previous_action =  action


        # Updating q values of final state for win/draw
        R = -1
        next_state = copy.deepcopy(self.state)
        for x in range(self.r):
            if( next_state[self.r-1-x][action] == 0):
                next_state[self.r-1-x][action] = self.player
                break

        isTS,result = self.is_terminal_state(next_state,action)
        
        str_curr_state_action = str_curr_state + str(action)

        if isTS:
            if self.is_winning_state(next_state,action):
                R = 50
                
            else:
                R = -10

            # str_curr_state_action = str_curr_state + str(action)
            curr_q_s_a = self.q_values[str_curr_state_action]
            curr_q_s_a += self.alpha*(R - curr_q_s_a )
            self.q_values[str_curr_state_action] = curr_q_s_a
            str_mirror_state_action = self.mirror_state_action(self.state, action)
            self.q_values[str_mirror_state_action] = curr_q_s_a
            self.total_rewards += R

        self.state = next_state


        return next_state, isTS, result, action, self.q_values[str_curr_state_action]
