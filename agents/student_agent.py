# Student agent: Add your own agent here
from asyncio import FIRST_COMPLETED
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
import math
import numpy as np
import time
import random
import gc
import resource

DIR_MAP = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

# Params
C = math.sqrt(2)
WIN_SCORE = 1.5
FIRST__SIMULATION_TIME = 29.5
SIMULATION_TIME = 1.9
MIN_SCORE = float('-inf')
DETECT_TRAPS = False # whether or not check for moves that lead to 3-walls traps\
VERBOSE = False

# Status Codes
P0_WIN = 0
P1_WIN = 1
DRAW = 2
IN_PROGRESS = 3
P0_TRAP = 4
P1_TRAP = 5

# Direction Codes
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.mcts = None
        self.round = 0
        print(resource.getrlimit(resource.RLIMIT_CORE))
        resource.setrlimit(resource.RLIMIT_CORE,(500*1024,-1))


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

        p0_pos, p1_pos = my_pos, adv_pos # assume current player is p0 and opponent is p1
        turn = 0
        state = State(chess_board=chess_board, p0_pos=p0_pos, p1_pos=p1_pos, turn=turn, max_step=max_step) # initial state

        if self.round == 0:
            max_simulation_time = FIRST__SIMULATION_TIME
            self.mcts = MCTS(state)
        else:
            max_simulation_time = SIMULATION_TIME
            self.mcts.update_root(state)

        next_move, dir = self.mcts.find_next_move(self.mcts.root.state, max_simulation_time=max_simulation_time)

        self.round += 1
        return next_move, dir

class Node:
    def __init__(self, state, parent=None):
        self.state = deepcopy(state)
        self.parent = parent
        self.children = list()
        
        if self.parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

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
        if (self.win_score != MIN_SCORE): 
            self.win_score += score

    def expand_node(self): 
        """
        Expand the node by finding all its possible states and creating
        a child node for each state.

        Return
        ------
        return the expanded node
        """
        possible_moves = self.state.all_possible_moves()

        for move in possible_moves:
            new_state = self.state.apply_move(move)
            self.children.append(Node(new_state, parent=self)) # create new node
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
        while len(node.children) != 0:
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
            uct_values[i] = MCTS.uct_value(wi=children[i].win_score, ni=children[i].visit_count, t=parent_visit_count)
 
        max_index = np.argmax(uct_values)
        return children[max_index]
    

    def simulate_random_playout(self):
        """
        Simulate a random full game playout from current node.

        Return
        ----------
        return the playout result status.possible values: P0_WIN, P1_WIN, DRAW
        """
        cur_status = self.state.check_board_status()

        # if state is a loss, do not consider it
        if self.state.turn == 0:
            if cur_status == P1_WIN:
                self.parent.state.win_score = MIN_SCORE # minimal integer value
                return cur_status
        else:
            if cur_status == P0_WIN:
                self.parent.state.win_score = MIN_SCORE
                return cur_status
            
        # Simulate game with random moves 
        cur_state = deepcopy(self.state)
        cur_status = cur_state.check_board_status()

        while cur_status == IN_PROGRESS: 
            cur_state = cur_state.random_play()
            cur_status = cur_state.check_board_status()
        return cur_status

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

            if (playout_result == (node.state.turn+1)%2):
                node.add_score(WIN_SCORE) 
            elif (playout_result == DRAW):
                node.add_score(WIN_SCORE/4)                

            node = node.parent

    def get_child_with_max_score(self):
        """
        Select child with max score from current node.

        Return
        ----------
        return the child with max score
        """
        max_score = 0
        best_child = None

        for i, child in enumerate(self.children):
            score = child.win_score/child.visit_count
            if  score > max_score:
                best_child = child
                max_score = score

        return best_child


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

    
    def find_next_move(self, state, max_simulation_time=2):
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

        start_time = time.time()

        count = 0
        root_node = self.root
        while ((time.time() - start_time) < max_simulation_time):

            promising_node = root_node.select_promising_node() # select promising child node based on UCT

            if (promising_node.state.check_board_status() == IN_PROGRESS):
                promising_node.expand_node() # create child node for selected promising node
            
            node_to_explore = promising_node
            if (len(promising_node.children) != 0):
                node_to_explore = promising_node.get_random_child_node()
            
            playout_result = node_to_explore.simulate_random_playout()

            node_to_explore.back_propagation(playout_result)
            count += 1
        
        winner_node = root_node.get_child_with_max_score()
        
        if root_node.state.turn == 0:
            cur_pos = root_node.state.p0_pos
            next_move = winner_node.state.p0_pos
        else:
            cur_pos = root_node.state.p1_pos
            next_move = winner_node.state.p1_pos

        old = root_node.state.chess_board[next_move]
        new = winner_node.state.chess_board[next_move]
        diff = np.bitwise_xor(old,new)
        dir = np.argmax(diff)

        if VERBOSE:
            print("Number of simulations per second: {x:.06f}".format(x=count/(time.time()-start_time)))
        self.root = winner_node
        gc.collect()
        return next_move, dir

    def update_root(self, state):
        """
        Scan the children of the root node to find the next root state.

        Params
        ------
        - state: the state we are looking for

        Return
        ------
        return the new root node
        """
        root_node = self.root
        children = root_node.children
        root_updated = False
        for child in children:
            if child.state.same_state(state):
                self.root = child
                root_updated = True
                break
        if not root_updated:
            new_root = Node(state, parent=None)
            self.root = new_root

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

class State:
    def __init__(self, chess_board, p0_pos, p1_pos, turn, max_step):
        self.chess_board = chess_board.copy()
        self.p0_pos = p0_pos
        self.p1_pos = p1_pos
        self.turn = turn # player number 0 or 1
        self.max_step = max_step
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def toggle_player(self):
        """
        Toggle player turn.
        """
        self.turn = (self.turn + 1) % 2

    def set_barrier(self, pos, barrier_dir):
        """
        Set barrier on chessboard. Important to set
        both barriers.
        
        Parameters
        ----------
        pos: tuple
        barrier_dir: int

        Return
        -------
        return the updated state
        """

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
        

    def all_possible_moves(self):
        """
        Get all possible moves of the form ((y,x),dir) from current game state. (reachable and within max steps).

        Return
        -------
        return a list of tuples (pos, dir).
        """
        moves = list()
        # Get current player start position
        my_pos = self.p0_pos if self.turn == 0 else self.p1_pos

        # Get position of the adversary
        adv_pos = self.p0_pos if self.turn == 1 else self.p1_pos

        # BFS
        for barrier_dir in DIR_MAP.values():
            if self.chess_board[my_pos[0], my_pos[1], barrier_dir]:
                continue
            moves.append((my_pos, barrier_dir))

        move_queue = [(my_pos, 0)]
        visited = {tuple(my_pos)}
        while move_queue:
            cur_pos, cur_step = move_queue.pop(0)
            
            if cur_step == self.max_step:
                break

            for dir, move in enumerate(self.moves):
                r, c = cur_pos
                # look for barrier
                if self.chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos[0] + move[0], cur_pos[1] + move[1]

                # look for adversary or already visited
                if next_pos == adv_pos or tuple(next_pos) in visited:
                    continue
                
                # Endpoint already has barrier or is boarder
                r, c = next_pos

                for barrier_dir in DIR_MAP.values():
                    if self.chess_board[r, c, barrier_dir]:
                        continue
                    moves.append((next_pos, barrier_dir))

                visited.add(tuple(next_pos))
                move_queue.append((next_pos, cur_step + 1))
        return moves
    
    def apply_move(self, move):
        """
        Apply a move to current state and return new state.

        Params
        ------
        - move: tuple of the form (pos, dir) where pos is the
                next position and dir is the barrier direction.
        
        Return
        -------
        return the new state created from the move.
        """ 
        pos, dir = move
        # Append new state
        new_state = deepcopy(self)

        # Move player
        if new_state.turn == 0:
            new_state.p0_pos = pos
        else:
            new_state.p1_pos = pos

        # Add barrier
        new_state.set_barrier(pos, dir)

        # Toggle turn
        new_state.toggle_player()
        return new_state

    def is_trap(self, move):
        """
        Check whether the move would lead to a trap. A trap occurs when
        the player is surrounded by 3 walls. This 
        is a position that we tend to avoid.

        Params
        ------
        - move: the move that we want to verify.

        Return
        ------
        return True if move leads to a trap, False otherwise.
        """
        next_pos, dir = move
        barriers = self.chess_board[next_pos].copy()

        barriers[dir] = 1
        count = np.sum(barriers)
        return count >= 3

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
        if is_endgame:
            if p0 > p1:
                return P0_WIN
            elif p1 > p0:
                return P1_WIN
            else:
                return DRAW
        else:
            return IN_PROGRESS


    def random_play(self):
        """
        Current player turn performs a valid random move and game state 
        is updated accordingly.

        Return
        ----------
        return the updated game state.
        """

        moves = self.all_possible_moves()

        if DETECT_TRAPS:
            trap_free_moves = [move for move in moves if not self.is_trap(move)]
            if trap_free_moves:
                random_pos, random_dir = random.choice(trap_free_moves)
            else:
                random_pos, random_dir = random.choice(moves)
        else:
            random_pos, random_dir = random.choice(moves)

        # Update position, turn and barriers
        if self.turn == 0:
            self.p0_pos = random_pos
        else:
            self.p1_pos = random_pos

        self.toggle_player()
        self.set_barrier(random_pos, random_dir)
        return self
    
    def same_state(self, state):
        """
        Indicates whether or not a state is same as
        the current state.

        Params
        ------
        - state: the state that we want to compare

        Return
        ------
        return True if the state is same as the current state,
        return False otherwise.
        """

        if self.p0_pos != state.p0_pos:
            return False
        if self.p1_pos != state.p1_pos:
            return False
        if self.turn != state.turn:
            return False
        if not np.array_equal(self.chess_board, state.chess_board):
            return False
        return True



