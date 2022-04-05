# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import sys
import math
import numpy as np
import time
import random
import sys

DIR_MAP = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

# Moves (Up, Right, Down, Left)
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))

MAX_STEP = 3

C = math.sqrt(2)

P0_WIN = 0
P1_WIN = 1
DRAW = 2
IN_PROGRESS = 3
WIN_SCORE = 1

MAX_SIMULATION_TIME = 29.5

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        # super(StudentAgent, self).__init__()
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
        MAX_STEP = max_step

        p0_pos, p1_pos = my_pos, adv_pos # assume current player is p0 and opponent is p1
        turn = 0

        cur_state = State(chess_board, p0_pos, p1_pos, turn) # initial state

        next_move = self.find_next_move(cur_state)
        return next_move

    
    # ------------------------- MCTS -------------------------


    def find_next_move(self, state):
        '''
        Call MCTS from current game state. Simulate until maximum time 
        is reached.

        Parameters
        ----------
        state: the current state of the game before the move

        Return
        ----------
        return a tuple ((x,y), dir) where (x,y) is the next position of your agent 
        and dir is the direction where to put the wall.
        '''
        mcts = MCTS(state)
        root_node = mcts.root
        start_time = time.time()

        while ((time.time() - start_time) < MAX_SIMULATION_TIME):

            promising_node = root_node.select_promising_node() # select promising child node based on UCT

            if (promising_node.state.check_board_status() == IN_PROGRESS):
                promising_node.expand_node() # create child node for selected promising node
            
            node_to_explore = promising_node
            if (promising_node.children):
                node_to_explore = promising_node.get_random_child_node()
            
            playout_result = node_to_explore.simulate_random_playout()

            node_to_explore.back_propagation(playout_result)
        
        winner_node = root_node.get_child_with_max_score()

        diff = np.where(winner_node.state.chess_board - root_node.state.chess_board)
        
        next_move = diff[0][0], diff[1][0]
        dir = diff[2][0]

        return next_move, dir

def uct_value(wi, ni, t,c=C):
    '''
    Upper confidence boudn function

    Parameters
    ----------
    wi: number of wins of the ith move
    ni: number of simulations of the ith move
    t: number of simulations from the parent node
    c: exploration parameter   

    Return
    ----------
    return the node with the highest UCT value.
    '''
    return wi/ni + c*math.sqrt(math.log(t)/ni)



class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = list()

        # number of visits
        self.visit_count = 1

        # number of wins
        self.win_score = 0.5

    def increment_visit(self):
        """
        Increment number of visits to current node.
        """
        self.visit_count += 1

    def add_score(self, score):
        """
        Update win score of current node
        """
        if (self.win_score != (-sys.maxsize - 1)): #change for actual default we set 
            self.win_score += score

    def expand_node(self): 
        """
        Expand the node by finding all its possible states and creating
        a child node for each state.
        """
        children_states = self.state.all_possible_states()
        for s in children_states:
            self.children.append(Node(s, parent=self)) # create new node
        return self

    def select_promising_node(self): 
        """
        Go down the tree by selecting child node with best UCT value 
        at each level.

        Return
        ----------
        return the most promising child node
        """
        node = self
        #iterate down the node and find the best uct 
        while node.children:
            node = node.find_node_best_uct()
        return node
    
    def find_node_best_uct(self): 
        """
        Select child node with best UCT value

        Return
        ----------
        return the chosen node.
        """
        parent_visit_count = self.visit_count
        children = self.children
        number_of_children = len(children)
        uct_values = np.zeros(number_of_children)

        for i in range(number_of_children):
            uct_values[i] = uct_value(wi=children[i].win_score, ni=children[i].visit_count, t=parent_visit_count)
        
        max_index = np.argmax(uct_values)
        return children[max_index]
    

    def simulate_random_playout(self):
        """
        Simulate a random full game playout from current node.

        Return
        ----------
        return the playout result status.possible values: P0_WIN, P1_WIN, DRAW
        """
        status = self.state.check_board_status()

        # if state is a loss, do not consider it
        if self.state.turn == 0:
            if status == P1_WIN:
                self.parent.state.win_score = (-sys.maxsize -1) # minimal integer value
        else:
            if status == P0_WIN:
                self.parent.state.win_score = (-sys.maxsize -1)

        # Simulate game with random moves 
        cur_state = self.state

        while status == IN_PROGRESS: 
          cur_state = cur_state.random_play()
  
          status = cur_state.check_board_status()
        return status

    def back_propagation(self, playout_result):
        """
        Bapropagate the playout result from current node to root node. 

        Parameters
        ----------  
        - playout_result: the playout result. possible values: P0_WIN, P1_WIN, DRAW     
        """
        node = self
        while (node is not None):
            node.increment_visit()

            if (playout_result == node.state.turn):
                node.add_score(WIN_SCORE) 
            elif (playout_result == DRAW):
                node.add_score(WIN_SCORE/2) 

            node = node.parent

    def get_child_with_max_score(self):
        """
        Select child with max score from current node.

        Return
        ----------
        return the child with max score
        """
        number_of_children = len(self.children)
        scores= np.zeros(number_of_children)

        for i in range(number_of_children):
            scores[i] = (self.children[i].win_score)/(self.children[i].visit_count)

        max_index =  np.argmax(scores)
        return self.children[max_index]


    def get_random_child_node(self):
        """
        Select a random child node from current node

        Return
        ----------
        return the most promising child node
        """
        number_of_children = len(self.children)
        random_index = random.randint(0,number_of_children-1)
        return self.children[random_index]


class MCTS:
    def __init__(self, initial_state):
        self.root = Node(initial_state)


class State:
    def __init__(self, chess_board, p0_pos, p1_pos, turn):
        self.chess_board = chess_board.copy()
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn # player number 0 or 1

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
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

        self.chess_board[pos[0], pos[1],barrier_dir] = 1
        if barrier_dir == 0: # up
          if pos[0] - 1 >= 0:
            self.chess_board[pos[0]-1, pos[1],DOWN] = 1

        elif barrier_dir == 1: # right
          if pos[1] + 1 < self.chess_board.shape[1]:
            self.chess_board[pos[0], pos[1]+1,LEFT] = 1

        elif barrier_dir == 2: # down
          if pos[0] + 1 < self.chess_board.shape[0]:
            self.chess_board[pos[0]+1, pos[1],UP] = 1

        elif barrier_dir == 3: # left
          if pos[1] - 1 >= 0:
            self.chess_board[pos[0], pos[1]-1,RIGHT] = 1
        return self
        

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
            
            if cur_step == MAX_STEP:
                break

            for dir, move in enumerate(MOVES):
                r, c = cur_pos
                # look for barrier
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos[0] + move[0], cur_pos[1] + move[1]

                # look for adversary or already visited
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                
                # Endpoint already has barrier or is boarder
                r, c = next_pos

                for barrier_dir in DIR_MAP.values():
                    if self.chess_board[r, c, barrier_dir]:
                        continue

                    # Append new state
                    new_state = deepcopy(self)

                    # Move player
                    if new_state.turn == 0:
                        new_state.p0_pos = next_pos
                    else:
                        new_state.p1_pos = next_pos

                    # Add barrier
                    new_state.set_barrier(next_pos, barrier_dir)

                    # Toggle turn
                    new_state.toggle_player()

                    states.append(new_state)
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return states

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
                    MOVES[1:3]
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

    def check_board_status(self):
        """
        Return the current board status.
        
        Parameters
        ----------
        state: the game state

        Return
        ----------
        return the board state code. Possible values: IN_PROGRESS, P0_WIN, P1_WIN, DRAW
        """
        is_endgame, p0, p1 = self.check_endgame()
        if (is_endgame):
            if (p0 > p1):
                return P0_WIN
            elif (p1 > p0):
                return P1_WIN
            else:
                return DRAW
        else:
            return IN_PROGRESS


    def random_play(self):
        """
        Current player turn performs a random move and game state is updated
        accordingly.

        Return
        ----------
        return the updated game state.
        """

        if self.turn == 0:
            my_pos = self.p0_pos
            adv_pos = self.p1_pos
        else:
            my_pos = self.p1_pos
            adv_pos = self.p0_pos 

        ori_pos = deepcopy(my_pos)
        moves_inds = list(range(len(MOVES)))

        # Random Walk
        steps = np.random.randint(0, MAX_STEP + 1)
        valid_move_found = True

        for _ in range(steps):

          r, c = my_pos
          # pick random direction
          random.shuffle(moves_inds)

          for dir in moves_inds:
            m_r, m_c = MOVES[dir]
            new_pos = (r + m_r, c + m_c)
            if self.chess_board[r, c, dir] or new_pos == adv_pos:
              valid_move_found = False
            else:
              valid_move_found = True
              my_pos = new_pos
              break

          if not valid_move_found:
            break

        random.shuffle(moves_inds)
        valid_move_found = False
        r, c = my_pos
        for dir in moves_inds:
          if self.chess_board[r, c, dir] == 0:
            valid_move_found = True
            self.set_barrier(my_pos, dir)
            break

        if not valid_move_found:
          print("ERROR: CANT MOVE")
          sys.exit(-2)

        # Update state position
        if self.turn == 0:
            self.p0_pos = my_pos
        else:
            self.p1_pos = my_pos

        self.toggle_player()

        return self






