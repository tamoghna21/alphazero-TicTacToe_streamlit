import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import *
import numpy as np
import random

import MCTSwithRL # MCTS aided by Actor-Critic RL

from Policy_NN import Policy # This is only for 3x3 board

from copy import copy
import random

from ConnectN import ConnectN  # The game environment
game_setting = {'size':(3,3), 'N':3}

# load the weights from file
policy_alphazero = Policy() ## Policy_NN is only for 3x3 board
policy_alphazero.load_state_dict(torch.load('Policy_alphazero_tictactoe.pth')) 

saved_policy_alphazero_6by6 = torch.load('policy_6-6-4-pie-4500_tictactoe.pth')

def get_button_color_css():
    return f"""
            <style>
                div[data-testid="column"]{{
                    width: 40px;
                    flex: unset;
                }}
                div[data-testid="column"] * {{
                    width: 40px;
                }}
                .stButton>button["primary"] {{
                    border: 1px solid #FF0000;
                    margin: 5px;
                    color: black;
                }}
                .stButton>button["secondary"] {{
                    border: 2px solid #FF0000;
                    margin: 5px;
                    outline: none;
                    color: #778899;
                }}
            </style>
            """

st.markdown(get_button_color_css(), unsafe_allow_html=True)

def Policy_Player_MCTS(game):
    tree = MCTSwithRL.Node(copy(game))
    for _ in range(50): # explore the tree 50 steps 
        tree.explore(policy_alphazero) # This will compute all the U s, pick the branch with max U, search, 
                               # expand, backpropagate and increase the visit count
   
    treenext, (v, nn_v, p, nn_p) = tree.next(temperature=0.1) # Asking the tree to choose a next move based on the visit counts
    return treenext.game.last_move # returns the move after incrementing the Tree

def Policy_Player_MCTS_6by6(game):
    tree = MCTSwithRL.Node(copy(game))
    for _ in range(100): # explore the tree 1000 steps
        tree.explore(saved_policy_alphazero_6by6) # This will compute all the U s, pick the branch with max U, search, 
                               # expand, backpropagate and increase the visit count
   
    treenext, (v, nn_v, p, nn_p) = tree.next(temperature=0.1) # Asking the tree to choose a next move based on the visit counts
    return treenext.game.last_move # returns the move after incrementing the Tree
    


if 'current_sts_msg' not in st.session_state:
    st.session_state.current_sts_msg = ''
if 'current_pierule_msg' not in st.session_state:
    st.session_state.current_pierule_msg = ''


def reset_game():
    st.session_state.board = [['' for _ in range(3)] for _ in range(3)]
    st.session_state.current_player = 'X'
    st.session_state.end = 0
    st.session_state.first_move_played = 0
    st.session_state.current_sts_msg = ''
    st.session_state.current_pierule_msg = ''
    
    if selected_option == 'You play first':
        st.session_state.player1=None
        st.session_state.player2=Policy_Player_MCTS
    else:
        st.session_state.player1=Policy_Player_MCTS
        st.session_state.player2=None
        
    if selected_board_option == '3x3 board':
        st.session_state.board_dimension = '3by3'
        st.session_state.board = [['' for _ in range(3)] for _ in range(3)]
        game_setting = {'size':(3,3), 'N':3}
        if selected_option == 'You play first':
            st.session_state.player1=None
            st.session_state.player2=Policy_Player_MCTS
        else:
            st.session_state.player1=Policy_Player_MCTS
            st.session_state.player2=None
    else:
        st.session_state.board_dimension = '6by6'
        st.session_state.board = [['' for _ in range(6)] for _ in range(6)]
        game_setting = {'size':(6,6), 'N':4, 'pie_rule':True}
        if selected_option == 'You play first':
            st.session_state.player1=None
            st.session_state.player2=Policy_Player_MCTS_6by6
        else:
            st.session_state.player1=Policy_Player_MCTS_6by6
            st.session_state.player2=None
    st.session_state.game = ConnectN(**game_setting)
        

st.header('Play Tic Tac Toe with an AI agent (alphazero)', divider='rainbow')
#st.subheader(st.session_state.current_sts_msg)

# Define options for radio buttons
options = ["You play first", "You play second"]
options_board = ["3x3 board", "6x6 board (pie rule on)"]

selected_option = st.radio("Select an option:", options=options, horizontal=True)
selected_board_option = st.radio("Select board:",options=options_board, horizontal = True)

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
if 'board_dimension' not in st.session_state:
    #st.session_state.board_dimension = '3by3'
    if selected_board_option == '3x3 board':
        st.session_state.board_dimension = '3by3'
        st.session_state.board = [['' for _ in range(3)] for _ in range(3)]
        game_setting = {'size':(3,3), 'N':3}
        st.session_state.player1 = None
        st.session_state.player2 = Policy_Player_MCTS
    else:
        st.session_state.board_dimension = '6by6'
        st.session_state.board = [['' for _ in range(6)] for _ in range(6)]
        game_setting = {'size':(6,6), 'N':4, 'pie_rule':True}
        st.session_state.player1 = None
        st.session_state.player2 = Policy_Player_MCTS_6by6
        
    st.session_state.game = ConnectN(**game_setting)

#st.subheader(st.session_state.game.score)
st.subheader(st.session_state.current_sts_msg)
#st.subheader(st.session_state.end)

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
        #player = st.session_state.game.player
        succeed=st.session_state.game.move((i,j))
    
        # Place the player's mark on the board
        if succeed:
            st.session_state.board[i][j] = st.session_state.current_player
            # Check for winner
            if st.session_state.game.score is None:
                # game is not over ; Switch player
                st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
            elif st.session_state.game.score == 0:
                # It's a draw
                st.session_state.current_sts_msg = "It's a draw!"
                st.session_state.end = 1
            else:
                st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
                st.session_state.end = 1
            
            # Automatic move generation by the agent
            if (st.session_state.end == 0):
                if st.session_state.player1 is not None or st.session_state.player2 is not None:
                    succeed = False
                    while not succeed:
                        if st.session_state.game.player == 1:
                            loc = st.session_state.player1(st.session_state.game)
                        else:
                            loc = st.session_state.player2(st.session_state.game)
                        succeed = st.session_state.game.move(loc)
                    row,col=loc
        
                    # Place the player's mark on the board
                    if succeed:
                        if (st.session_state.game.n_moves == 2 and abs(np.sum(st.session_state.game.state))==1):
                            st.session_state.current_pierule_msg = '*pie rule exercised*:sunglasses:'
                        elif (st.session_state.game.n_moves == 3 and abs(np.sum(st.session_state.game.state))==0):
                            st.session_state.current_pierule_msg = '*You exercised pie rule*:rage:,but I will still win'
                        else:
                            st.session_state.current_pierule_msg = ''
                        st.session_state.board[row][col] = st.session_state.current_player
                        # Check for winner
                        if st.session_state.game.score is None:
                            # game is not over ; Switch player
                            st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
                        elif st.session_state.game.score == 0:
                            # It's a draw
                            st.session_state.current_sts_msg = "It's a draw!"
                            st.session_state.end = 1
                        else:
                            st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
                            st.session_state.end = 1
                                           

# first player is the Agent
if st.session_state.player1 is not None and st.session_state.first_move_played == 0:
    succeed = False
    while not succeed:
        loc = st.session_state.player1(st.session_state.game)
        succeed = st.session_state.game.move(loc)
    st.session_state.first_move_played = 1
    row,col=loc
    
    # Place the player's mark on the board
    #if st.session_state.board[row][col] == '':
    if succeed:
        st.session_state.board[row][col] = st.session_state.current_player
        # Check for winner
        if st.session_state.game.score is None:
            # game is not over ; Switch player
            st.session_state.current_player = 'O' if st.session_state.current_player == 'X' else 'X'
        elif st.session_state.game.score == 0:
            # It's a draw
            st.session_state.current_sts_msg = "It's a draw!"
            st.session_state.end = 1
        else:
            st.session_state.current_sts_msg = f"Player {st.session_state.current_player} wins!"
            st.session_state.end = 1
            
        
# Display the Tic Tac Toe board
if st.session_state.board_dimension == '3by3':
    num_cols=3
else:
    num_cols=6
    
cols= st.columns(num_cols,gap='small')
for i in range(num_cols):
    for j in range(num_cols):
        cols[j].button(label=st.session_state.board[i][j] or "  ",on_click=on_button_click, args=(i, j), key=f'button_{i}_{j}',type="primary")


st.write(st.session_state.current_pierule_msg)        
st.button("Reset Board", on_click=reset_game,type="secondary")


st.write("The AI agent chooses moves based on Monte-Carlo-Tree-Search guided by Actor-Critic Reinforcement Learning Algorithm")
st.write("The stramlite code is available here: https://github.com/tamoghna21/alphazero-TicTacToe_streamlit")
st.write("The RL agent training code is available here: https://github.com/tamoghna21/deep-reinforcement-learning-python-pytorch/tree/main/alphazero-TicTacToe")
st.write("Created by : Tamoghna Das")


    

