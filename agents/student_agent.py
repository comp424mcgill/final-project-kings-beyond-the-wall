# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import sys
import math
import numpy as np

dir_map = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

# Moves (Up, Right, Down, Left)
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

C = math.sqrt(2)

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"


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

    
    # ------------------------- MCTS -------------------------

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



    # Simulate random playouts from a node and return the board status
    def simulate_random_playout(node):
        # TODO
        return

    # Backpropagate simulation results
    def back_propagation(node, playerNo):
        # TODO
        return

    def board_status(state):
        """
        Return the current board status.
        
        Parameters
        ----------
        state: the game state
        """
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

    def expand(self):
        """
        Expand the node by finding all its possible states.
        
        Parameters
        ----------
        node: the node to expand
        """
        states = self.state.all_possible_states()
        for s in states:
            self.children.append(Node(s, parent=self))

class Tree:
    def __init__(self, state):
        self.root = Node(state)

class State:
    def __init__(self, chess_board, p0_pos, p1_pos, turn, max_step):
        self.chess_board = chess_board.copy()
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn # player number 0 or 1
        self.max_step = max_step

    def toggle_player(self):
        """
        Toggle player turn.
        """
        self.turn = (self.turn + 1) % 2

    def set_barrier(self, pos, barrier_dir):
        """
        Set barrier on chessboard.
        
        Parameters
        ----------
        pos: tuple
        barrier_dir: int
        """
        self.chess_board[pos[0], pos[1],barrier_dir] = 1

    def all_possible_states(self):
        """
        Get all possible states from current game state. (reachable and within max steps).
        """
        states = list()

        # Get current player start position
        start_pos = self.p0_pos if self.turn == 0 else self.p1_pos

        # Get position of the adversary
        adv_pos = self.p0_pos if self.turn == 1 else self.p1_pos

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(moves):

                # look for barrier
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move

                # look for adversary or already visited
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                
                # Endpoint already has barrier or is boarder
                r, c = next_pos

                for barrier_dir in dir_map.values():
                    if self.chess_board[r, c, barrier_dir]:
                        continue

                    # Append new state
                    new_state = deepcopy(self)
                    # Toggle turn
                    new_state.toggle_player()
                    # Move player
                    if self.turn == 0:
                        new_state.p0_pos = next_pos
                    else:
                        new_state.p1_pos = next_pos
                    # Add barrier
                    new_state.set_barrier(next_pos, barrier_dir)
                    states.append(new_state)

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

    def check_endgame(self):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_0_score : int
            The score of player 0.
        player_1_score : int
            The score of player 1.
        """
        board_size = self.chess_board.shape[0]
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 0
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = 1
            win_blocks = p1_score
        else:
            player_win = -1  # Tie
        return True, p0_score, p1_score

    def random_play(self):
        """
        Play a random move among list of possible moves on the board.
        Update state accordingly.
        """
        # TODO
        pass






