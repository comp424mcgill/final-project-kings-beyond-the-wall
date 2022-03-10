# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import math
import numpy as np


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        # dummy return
        return my_pos, self.dir_map["u"]

    
    # Call mcts search from current game state.
    # Keep track of time with a while loop to set an upper bound on the 
    # simulation execution time.
    def find_next_move(self):
        # TODO
        # while time < 30 sec....
        # select best node
        return

        # Select most promising node 
    def select_promising_node(node):
        # TODO
        return
    
    # Select child node with best uct value
    def find_node_best_uct(node):
        #TODO
        return

    # Upper confidence bound function
    # wi: number of wins after the ith move
    # ni: number of simulations after the ith move
    # t: total number of simulations for the parent node
    # c: exploration parameter
    def uct_value(wi, ni, t,c=C):
        return wi/ni + c*math.sqrt(math.log(t)/ni)

    # Expand a node by finding all its possible states.
    def expand_node(node):
        # TODO
        return

    # Simulate random playouts from a node and return the board status
    def simulate_random_playout(node):
        # TODO
        return

    # Backpropagate simulation results
    def back_propagation(node, playerNo):
        # TODO
        return

    # return the current board status
    def board_status(self):
        #TODO
        return


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = list()

        # number of visits
        self.visit_count = 0

        # number of wins
        self.win_score = 0

class Tree:
    def __init__(self, state):
        self.root = Node(state)

class State:
    def __init__(self, chess_board, p0_pos, p1_pos, turn):
        self.chess_board = chess_board.copy()
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn # player number 0 or 1

    def toggle_player(self):
        # toggle between 0 ans 1
        self.turn = (self.turn + 1) % 2

    def copy(self):
        state_copy = State(self.chess_board, self.p0_pos, self.p1_pos, self.turn)
        return state_copy

    # Build a list of all possible states from current state.
    # A move is represented as a tuple (x,y,dir) where x is the final
    # position in x, y is the final position on y axis and dir is the
    # barrier direction: 'u','d','l','r'.    def all_possible_states(self):
    def all_possible_states(self):
        states = list()

        # current player and opponent positions
        if self.turn == 0:
            start_pos = self.p0_pos
            adv_pos = self.p1_pos
        else:
            start_pos = self.p1_pos
            adv_pos = self.p0_pos 

        my_pos = self.p0_pos if self.turn == 0 else self.p1_pos
        for i in range(self.world.max_step+1):

            for move in self.world.moves:
                end_pos = (start_pos[0] + move[0], start_pos[1] + move[1])



            

        return states

    # Play a random move among list of possible moves on the board.
    # Update state accordingly.
    def random_play(self):
        # TODO
        pass






