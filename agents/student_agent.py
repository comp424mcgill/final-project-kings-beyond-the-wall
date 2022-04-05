# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import sys
import math
import numpy as np
import time
import random

dir_map = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

# Moves (Up, Right, Down, Left)
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

C = math.sqrt(2)

P0_WIN = 0
P1_WIN = 1
DRAW = 2
IN_PROGRESS = 3
WIN_SCORE = 1

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
    def find_next_move(self, chess_board, state, turn):
        # TODO
        tree = Tree(state)
        root_node = tree.root
        root_node.state.chess_board = chess_board
        root_node.state.turn = turn
        start_time = time.time()

        while ((time.time() - start_time) < 29) {
            promising_node = select_promising_node(root_node)
            if (!(check_endgame(self)[0])) {
                expand(promising_node); 
            }
            new_node = promising_node;
            if (len(promising_node.children) > 0) {
                new_node = promising_node.get_random_child_node()
            }
            playout_result = simulate_random_playout(new_node)
            back_propagation(new_node, playout_result)
        }
        winner_node = root_node.get_child_with_max_score()
        return winner_node.state

    # Upper confidence bound function
    # wi: number of wins after the ith move
    # ni: number of simulations after the ith move
    # t: total number of simulations for the parent node
    # c: exploration parameter
    def uct_value(wi, ni, t,c=C):
        return wi/ni + c*math.sqrt(math.log(t)/ni)


    def board_status(state):
        """
        Return the current board status.
        
        Parameters
        ----------
        state: the game state
        """
        #TODO
        #should we evaluate whether the game is lost or won?
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

    def expand(self): #idk if this should be a method in the agent and we pass it a node?
        """
        Expand the node by finding all its possible states.
        
        Parameters
        ----------
        node: the node to expand
        """
        states = self.state.all_possible_states()
        for s in states:
            self.children.append(Node(s, parent=self))

    # Select most promising node 
    def select_promising_node(node): #should this be in the agent as well? passing root node
        #iterate down the node and find the best uct 
        while (len(node.children != 0)):
            node = find_node_best_uct(node)
        return node
    
    # Select child node with best uct value
    def find_node_best_uct(node): #should this be static and passing a node
        #TODO
        parent_visit = node.state.visit_count
        children_length = len(node.children)
        children = node.children
        uct_values = []
        for i in range(children_length):
            uct_values.append(uct_value(parent_visit, children[i].state.win_score, children[i].state.visit_count))
        max_uct_value_index = np.argmax(uct_values)
        return children[i]
    
    def check_board_status(state):
        is_endgame, p0, p1 = state.check_endgame()
        if (is_endgame):
            if (p0 > p1):
                return P0_WIN
            elif (p1 > p0):
                return P1_WIN
            else:
                return DRAW
        else:
            return IN_PROGRESS

    # Simulate random playouts from a node and return the board status
    def simulate_random_playout(node):
        # TODO
        state = node.state
        board_status = check_board_status(state)
        if (state.turn = 0):
            if (board_status == P1_WIN):
                node.parent.state.win_score = -sys.maxint -1
        else:
            if (board_status == P0_WIN):
                node.parent.state.win_score = -sys.maxint -1
        while (board_status == IN_PROGRESS): 
            state.toggle_player()
            if (state.turn = 0):
                my_pos = state.p0_pos
                adv_pos = state.p1_pos
            else: 
                my_pos = state.p1_pos
                adv_pos = state.p0_pos
            state.random_walk(my_pos, adv_pos)
            board_status = check_board_status(state)
        return board_status

    # Backpropagate simulation results
    def back_propagation(self, playout_result):
        # TODO
        node = self
        while (node is not None):
            node.state.increment_visit()
            if (node.state.turn == playout_result):
                node.state.add_score(WIN_SCORE) #add draw after maybe
            node = node.parent

    # Get the child node with the highest score which corresponds to the child with highest visit count
    def get_child_with_max_score():
        children_scores= list()
        length = len(self.children)
        for i in range(length)
            children_scores.append(self.children[i].state.win_score/self.children[i].state.visit_count)
        return max(children_scores)

    # Get a random node from the list of possible moves
    def get_random_child_node():
        int moves = len(self.children)-1
        int random_index = random.randint(0,moves)
        return self.children[random_index]

class Tree:
    def __init__(self, state):
        self.root = Node(state)

class State:
    def __init__(self, chess_board, p0_pos, p1_pos, turn, max_step, visit_count, win_score):
        self.chess_board = chess_board.copy()
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn # player number 0 or 1
        self.max_step = max_step
        self.visit_count = visit_count
        self.win_score = win_score

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
    
    def increment_visit():
        self.visit_count += 1

    def add_score(score)
        if (self.win_score != (-sys.maxint - 1)): #change for actual default we set 
            this.win_score += score;

    def random_walk(self, my_pos, adv_pos):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, self.max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = self.moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir






