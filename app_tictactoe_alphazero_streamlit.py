import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
import random

import MCTSwithRL # MCTS aided by Actor-Critic RL

from Policy_NN import Policy

from copy import copy
import random

# load the weights from file
policy_alphazero = Policy()
policy_alphazero.load_state_dict(torch.load('Policy_alphazero_tictactoe.pth')) 

button_width = 200
button_height = 100

def Policy_Player_MCTS(game):
    tree = MCTSwithRL.Node(copy(game))
    for _ in range(50): # explore the tree 50 steps #50
        tree.explore(policy_alphazero) # This will compute all the U s, pick the branch with max U, search, 
                               # expand, backpropagate and increase the visit count
   
    treenext, (v, nn_v, p, nn_p) = tree.next(temperature=0.1) # Asking the tree to choose a next move based on the visit counts
    return treenext.game.last_move # returns the move after incrementing the Tree
    
from ConnectN import ConnectN
game_setting = {'size':(3,3), 'N':3}
game = ConnectN(**game_setting)

#player1=None
#player2=Policy_Player_MCTS

#player1=Policy_Player_MCTS
#player2=None


# Initialize session state for the board and current player
if 'board' not in st.session_state:
    st.session_state.board = [['' for _ in range(3)] for _ in range(3)]
if 'current_player' not in st.session_state:
    st.session_state.current_player = 'X'
if 'game' not in st.session_state:
    st.session_state.game = ConnectN(**game_setting)
if 'end' not in st.session_state:
    st.session_state.end = 0
if 'first_move_played' not in st.session_state:
    st.session_state.first_move_played = 0
if 'player1' not in st.session_state:
    st.session_state.player1 = None
if 'player2' not in st.session_state:
    st.session_state.player2 = Policy_Player_MCTS
if 'current_sts_msg' not in st.session_state:
    st.session_state.current_sts_msg = ''
    
def reset_game():
    st.session_state.board = [['' for _ in range(3)] for _ in range(3)]
    st.session_state.current_player = 'X'
    st.session_state.game = ConnectN(**game_setting)
    st.session_state.end = 0
    st.session_state.first_move_played = 0
    st.session_state.current_sts_msg = ''
    
    if selected_option == 'You play first':
        st.session_state.player1=None
        st.session_state.player2=Policy_Player_MCTS
    else:
        st.session_state.player1=Policy_Player_MCTS
        st.session_state.player2=None
        
#st.markdown('<style>body{background-color: Blue;}</style>',unsafe_allow_html=True)

backgroundColor = "#F0F0F0"

st.header('Play Tic Tac Toe with an AI agent (alphazero)', divider='rainbow')
st.subheader(st.session_state.current_sts_msg)  
    
# Define options for radio buttons
options = ["You play first", "You play second"]

# Display radio buttons
selected_option = st.radio("Select an option:", options)
    
def check_winner(board, player):
    # Check horizontal, vertical and diagonal conditions
    for i in range(3):
        if all([cell == player for cell in board[i]]):
            return True
        if all([board[j][i] == player for j in range(3)]):
            return True
    if board[0][0] == board[1][1] == board[2][2] == player or \
       board[0][2] == board[1][1] == board[2][0] == player:
        return True
    return False


def on_button_click(i, j):
    if st.session_state.end == 0:
        player = st.session_state.game.player
        succeed=st.session_state.game.move((i,j))
    
        # Place the player's mark on the board
        if st.session_state.board[i][j] == '':
            st.session_state.board[i][j] = st.session_state.current_player
            # Check for winner
            if check_winner(st.session_state.board, st.session_state.current_player):
                #st.write(f"Player {st.session_state.current_player} wins!")
                st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
                st.session_state.end = 1
                #reset_game()
            elif all(cell != '' for row in st.session_state.board for cell in row):
                #st.write("It's a draw!")
                st.session_state.current_sts_msg = "It's a draw!"
                st.session_state.end = 1
                #reset_game()
            else:
                # Switch player
                st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
    
        if (st.session_state.end == 0):
            if st.session_state.player1 is not None or st.session_state.player2 is not None:
                succeed = False
                player = st.session_state.game.player
                while not succeed:
                    if st.session_state.game.player == 1:
                        loc = st.session_state.player1(st.session_state.game)
                    else:
                        loc = st.session_state.player2(st.session_state.game)
                    succeed = st.session_state.game.move(loc)
                row,col=loc
        
                # Place the player's mark on the board
                if st.session_state.board[row][col] == '':
                    st.session_state.board[row][col] = st.session_state.current_player
                    # Check for winner
                    if check_winner(st.session_state.board, st.session_state.current_player):
                        #st.write(f"Player {st.session_state.current_player} wins!")
                        st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
                        st.session_state.end = 1
                        #reset_game()
                    elif all(cell != '' for row in st.session_state.board for cell in row):
                        #st.write("It's a draw!")
                        st.session_state.current_sts_msg = "It's a draw!"
                        st.session_state.end = 1
                        #reset_game()
                    else:
                        # Switch player
                        st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
            
    #if (st.session_state.end == 1):
        #reset_game()


        
if st.session_state.player1 is not None and st.session_state.first_move_played == 0:
    succeed = False
    while not succeed:
        loc = st.session_state.player1(st.session_state.game)
        succeed = st.session_state.game.move(loc)
    st.session_state.first_move_played = 1
    row,col=loc
    
    # Place the player's mark on the board
    if st.session_state.board[row][col] == '':
        st.session_state.board[row][col] = st.session_state.current_player
        # Check for winner
        if check_winner(st.session_state.board, st.session_state.current_player):
            #st.write(f"Player {st.session_state.current_player} wins!")
            st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
            st.session_state.end = 1
            #reset_game()
        elif all(cell != '' for row in st.session_state.board for cell in row):
            #st.write("It's a draw!")
            st.session_state.current_sts_msg = "It's a draw!"
            st.session_state.end = 1
            #reset_game()
        else:
            # Switch player
            st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
    
    #if (st.session_state.end == 1):
        #reset_game()


# Display the Tic Tac Toe board
for i in range(3):
    cols = st.columns(3)
    for j in range(3):
        cols[j].button(st.session_state.board[i][j] or " ", on_click=on_button_click, args=(i, j), key=f'button_{i}_{j}',type="primary")

m = st.markdown("""
<style>
div.stButton > button:first-child[kind="primary"] {
    background-color: #473D3B;
    color:#ffffff;
}
div.stButton > button:first-child[kind="secondary"] {
    background-color: purple;
    color:#ffffff;
}
</style>""", unsafe_allow_html=True)
        
st.button("Reset Board", on_click=reset_game,type="secondary")

st.write("")
st.write("The AI agent chooses moves based on Monte-Carlo-Tree-Search guided by Actor-Critic Reinforcement Learning Algorithm")
st.write("The stramlite code is available here: https://github.com/tamoghna21/alphazero-TicTacToe_streamlit")
st.write("The RL agent training code is available here: https://github.com/tamoghna21/deep-reinforcement-learning-python-pytorch/tree/main/alphazero-TicTacToe")
st.write("Created by : Tamoghna Das")


    

