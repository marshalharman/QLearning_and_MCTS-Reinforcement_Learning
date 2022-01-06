import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import math
import copy
import gzip
import pickle
import shutil

from RandomPlayer import *
from MCTS import *
from QLearning import *

def print_grid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

def animate(i):
    graph_data = open('example.txt','r').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)

def mAverage(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



def MCTS_vs_MCTS(x, y):

    num_iterations = 1
    
    playouts = [x,y]
    winsP1 = []
    draws = []

    count_p1 = 0
    count_draw = 0
    last_action =""
    value = None
    C_MCTS = 2
    for i in range(num_iterations):
        print(i)
        game = np.zeros((6,5))
        game = game.astype(int)

        player1 = MCTS(playouts[0], 1,C_MCTS,6,5)
        player2 = MCTS(playouts[1], 2,C_MCTS,6,5)
          
        turn = 0
        result = ".."
            
        running = True

        while running:
          
            if(turn == 0):
                player1.set_state(game)
                game, end, result,last_action, value = player1.take_action()
                turn = 1

            else: 
                player2.set_state(game)
                game, end, result, last_action, value= player2.take_action()
                turn = 0
            
            print('Player' + str((turn+1)%2 + 1) +  '(MCTS with' + str(playouts[(turn+1)%2 ]) + ' playouts)')
            print('Action selected : '+ str(last_action))
            print('Total playouts for next state: '+ str(playouts[turn]))
            print('Value of next state according to MCTS : ' + str(value))
            print()

            if end == True:
                print()
                print()
                if result == "draw":
                    count_draw += 1
                    print("DRAW :")

                elif turn == 1:
                    count_p1 += 1
                    print("PLAYER 1 WINS :")
                elif turn == 0:
                    print("PLAYER 2 WINS :")
                running =  False
        
        print_grid(game)

    draw = (count_draw)/ num_iterations
    p1_win = (count_p1)/num_iterations
    winsP1.append(p1_win*100)
    draws.append(draw*100)
    # print("Draws: "+ str(draw))
    # print( "Win percentage of Player 1: " + str(p1_win))
    # print( "Win percentage of Player 2: " + str(1 - p1_win - draw))

    









def MCTS_vs_Q():

    count_p1 = 0
    count_draw = 0
    
    with gzip.open('q_data.dat.gz', 'rb') as f_in:
        with open('q_data.dat', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    with open('q_data.dat', 'rb') as handle:
        q_values = pickle.load(handle)

    testing_iter = 100
    n = 10
    C_MCTS = 2
    rows = 3
    cols = 5
    for i in range(testing_iter):
        print(i)
        
        game = np.zeros((rows,cols))
        game = game.astype(int)


        player1 = MCTS(n, 1, C_MCTS ,rows,cols)
        player2 = Q_Learning(2, 0, 0.8, 0 ,rows,cols)
        
        player2.set_Qvalues(q_values)

        turn = 0
        result = ".."
        last_action = None
        value = None
        running = True

        while running:
            if(turn == 0):
              
              player1.set_state(game)
              game, end, result, last_action, value = player1.take_action()
              turn = 1

            else :
                player2.set_state(game)
                game, end, result, last_action, value = player2.take_action()
                turn = 0

            if end == True:
                if turn == 1:
                    if result == "win":
                        count_p1 += 1
                        player2.game_status = "loss"
                        player2.take_action()
                        print("P1 win")
                    else:
                        count_draw += 1
                        player2.game_status = "draw"
                        player2.take_action()
                        print("Draw")

                if turn == 0:
                    if result == "win":
                        print("P2 win")
                    else:
                        count_draw += 1
                        print("Draw")
                break

            
            if(turn == 1):
                print('Player 1 (MCTS with' + str(n) +' playouts')
                print('Action selected : ' + str(last_action))
                print('Value of next state according to MCTS : ' + str(value))
            
            elif (turn == 0):
                print('Player 2 (Q-learning)')
                print('Action selected : '+ str(last_action))
                print('Value of next state : ' + str(value))
            
            print()

        print_grid(game)
        
        

    print("Draws: "+ str((count_draw)/ testing_iter))
    print( "Accuracy of P1, MCTS10: " + str((count_p1)/testing_iter))
    print( "Accuracy of P2, Q_Learning: " + str((testing_iter - count_p1 - count_draw)/testing_iter))

    print("Convergence till n = 10 can be tested as q learning is trained against MC10. Higher values can be tested but the results will show slightly less win percentage due to lack of training for higher values.")
    print("Currently, the number rows is set to: " + str(rows) )



def train_qlearning():
    rewards = []
    q_values = {}

    count_p1 = 0
    count_draw = 0

    #Hyper parameters:
    num_iterations = 50000
    n = 40
    C_MCTS = 2
    r = 4
    c = 5

    for i in range(num_iterations):
        print(i)
      
        game = np.zeros((r,c))
        game = game.astype(int)

        player1 = MCTS(n,1,2,r,c)
        player2 = Q_Learning(2, 0.6, 0.8, 0.1, r,c) 
      
        player2.set_Qvalues(q_values)
                
        turn = 0
        result = ".."
        
        running = True
        end = False
        while running:
      
            if(turn == 0):
                player1.set_state(game)
                game, end, result, last_action, value = player1.take_action()
                turn = 1

            else :
                player2.set_state(game)
                game, end, result, last_action, value = player2.take_action()
                turn = 0


            if end == True:

                if turn == 1:
                    if result == "win":
                        count_p1 += 1
                        player2.game_status = "loss"
                        player2.take_action()
                        print("P1 win")
                    else:
                        count_draw += 1
                        player2.game_status = "draw"
                        player2.take_action()
                        print("Draw")

                if turn == 0:
                    if result == "win":
                        print("P2 win")
                    else:
                        count_draw += 1
                        print("Draw")
                    
                running = False

        rewards.append(player2.total_rewards)
        # print_grid(game)
        # print("Reward" + str(player2.total_rewards))
        # print()
        


    # print("Draws: "+ str((count_draw)/ num_iterations))
    # print( "Accuracy of P1, MCTS: " + str((count_p1)/num_iterations))
    # print( "Accuracy of P2, Q: " + str((num_iterations - count_p1 - count_draw)/num_iterations))

    rewards = np.array(rewards)
    rewards = mAverage(rewards, 1000)
    fig = plt.figure()
    x_range = np.arange(1,num_iterations-998,1)
    y_range = np.arange(1,num_iterations+1,1)

    plt.plot(x_range, rewards)
    plt.xlabel('No. of Episodes')
    plt.ylabel('Rewards')
    plt.show()
    fig.savefig('MCTSvsQ.jpg')


    with open('q_data.dat', 'wb') as handle:
        pickle.dump(q_values, handle)

    with open('q_data.dat', 'rb') as f_in:
        with gzip.open('q_data.dat.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)






def main():
    
    print("Choose from the following: \n 1. MCTS agent vs MCTS agent \n 2. MCTS agent vs Q-Learning agent")
    choice = int(input())

    if(choice == 1):
        print("Enter number of playouts for player 1:")
        x = int(input())
        print("Enter number of playouts for player 2:")
        y = int(input())

        MCTS_vs_MCTS(x, y)
    if(choice == 2):
        print("Choose from the following: \n 1. Train Q-Learning against MCTS agent \n 2. Test Q-Learning against MCTS ")

        t = int(input())
        if t == 1:
            train_qlearning()
        if t == 2:
            MCTS_vs_Q()



if __name__=='__main__':
    main()