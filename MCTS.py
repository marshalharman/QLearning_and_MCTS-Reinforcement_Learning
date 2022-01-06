import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import copy

from RandomPlayer import *

class Node:

    def __init__(self, state):
        self.N = 0
        self.win = 0
        self.children = []
        self.state = copy.deepcopy(state)
        
    def update(self, win):
        if self.win == True:
            self.win+=1
        self.N += 1

    def get_ratio(self):
        if( self.N == 0):
            return 0

        return 1.0*self.win/self.N


class MCTS:

    def __init__(self, play_outs, player, C=2,r=6,c=5):
        self.player = player
        self.play_outs = play_outs
        self.parents = []
        self.C = C
        self.r = r
        self.c = c
    

    def set_state(self,state):
        self.state = state      

    def selection(self, root, depth):

        depth += 1
        self.parents.append(root)

        if( len(root.children) == 0 ):
            return root, depth - 1

        next_node = root.children[0]
        probabilites = np.ones(len(root.children), dtype = float)

        max_val = -math.inf
        
        zero_states = []
        for i in range(len(root.children)):
            if root.children[i].N == 0:
                zero_states.append(root.children[i])

        for i in range(len(root.children)):
            if(root.children[i].N == 0):
                next_node = random.choices(zero_states, k=1)
                next_node = next_node[0]
                break
            else: 
                ucb1 = root.children[i].get_ratio() + self.C*(np.log(root.N)/root.children[i].N)**0.5
                
                if ucb1 > max_val or (ucb1 == max_val and root.children[i].N > next_node.N):
                    next_node = root.children[i]
                    max_val = ucb1
        
        res,depth = self.selection(next_node,depth)
        return res,depth




    def expansion(self, leaf ,depth):
        
        curr_state = copy.deepcopy(leaf.state)

        possible_actions = np.ones(self.c)
        probabilites = np.ones(self.c, dtype = float)


        count = 0
        for i in range(self.c):
            if curr_state[0][i] == 0:
                count+=1
                new = copy.deepcopy(leaf.state)
                for x in range(self.r):
                    if(new[self.r-1-x][i] == 0):
                        if( depth%2 == 0):
                            new[self.r-1-x][i] = self.player
                        else:
                            new[self.r-1-x][i] = self.player%2 + 1
                        break

                leaf.children.append(Node(new))
        
        if (len(leaf.children) != 0 and leaf.N != 0):
            new_node = Node(leaf.children[0].state)
        else:
            return leaf,depth

        self.parents.append(new_node)

        return new_node,depth+1


    def simulation(self, child, depth):
            
        player1 = None
        player2 = None
        

        if( depth%2 == 0):
            player1 = Random_Player(self.player,self.r,self.c)
            player2 = Random_Player(self.player%2+1,self.r,self.c)
        else:
            player1 = Random_Player(self.player%2+1,self.r,self.c)
            player2 = Random_Player(self.player,self.r,self.c)
                
        
        game = []
        for x in range(self.r):
            row = []
            for y in range(self.c):
                row.append(child.state[x][y])
            game.append(row) 

        
        turn = 1
        result = ""
            
        while True:
            if(turn == 1):
                player1.set_state(game)
                game, end, result, last_action = player1.take_action()   
                turn = 2

            else :
                player2.set_state(game)
                game, end, result, last_action= player2.take_action()
                turn = 1

            
            if end == True:
                if(depth%2 == 0):
                    if(turn == 1 and result != "draw"):
                        result = "loss"
                    
                    elif(turn == 2 and result != "draw"):
                        result = "win"
                else:
                    if(turn == 2 and result != "draw"):
                        result = "loss"
                    
                    elif(turn == 1 and result != "draw"):
                        result = "win"
                
                break

        return result



    def back_propagation(self,result,child):

        x = 0
        if(result == "win"):
            x = 1
        elif(result == "loss"):
            x = -1

        for node in self.parents:
            node.N += 1
            node.win += x

    def construct_tree(self, root, depth):
        if depth <= 0:
            return

        for i in range(self.c):
            if root.state[0][i] == 0:
                child_state = []
                for x in range(self.r):
                    row = []
                    for y in range(self.c):
                        row.append(root.state[x][y])
                    child_state.append(row)

                for x in range(self.r):
                    if( child_state[self.r-1-x][i] == 0):
                        if( depth%2 == 0):
                            child_state[self.r-1-x][i] = self.player
                        else:
                            child_state[self.r-1-x][i] = self.player%2 + 1
                        break

                child = Node(child_state)
                root.children.append(child)
                self.construct_tree(child,depth-1)


    def take_action(self):

        if(self.play_outs == 0):
            action = self.random_action()
            for x in range(self.r):
                if(self.state[self.r-1-x][action] == 0):
                    self.state[self.r-1-x][action] = self.player
                    break

            isTS, result = self.is_terminal_state(self.state, action)
            return self.state, isTS, result

        root = Node(self.state)

        self.construct_tree(root,4)


        ## MCTS Algorithm
        for i in range(self.play_outs):
            # print(i)
            self.parents = []
            depth = 0

            leaf,depth = self.selection(root,0)
            child,depth = self.expansion(leaf,depth)
            result = self.simulation(child,depth)
            self.back_propagation(result,child)


        max_N = -math.inf
        res = None
        action = -1
    
        for i in root.children:
            if max_N < i.N:
                res = i
                max_N = i.N
                for x in range(self.r):
                    for y in range(self.c):
                        if(root.state[x][y] != i.state[x][y]):
                            action = y

        isTS, result = self.is_terminal_state(res.state, action)
        return res.state, isTS, result, action, res.win/res.N


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
