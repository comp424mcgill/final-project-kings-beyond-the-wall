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
# import resource
# import os, psutil

DIR_MAP = {
    "u": 0,
    "r": 1,
    "d": 2,
    "l": 3,
}

# Params
C = math.sqrt(2)
WIN_SCORE = 1.5
FIRST__SIMULATION_TIME = 28
SIMULATION_TIME = 1.5
MIN_SCORE = float('-inf')
AVOID_TRAPS = True # if set to True --> random play avoids traps
# IMPROVED_RANDOM_PLAY = False # if set to True, random play avoid losing moves
VERBOSE = False

# Status Codes
STATUS_P0_WIN = 0
STATUS_P1_WIN = 1
STATUS_DRAW = 2
STATUS_IN_PROGRESS = 3
STATUS_NONE = -1

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
        self.autoplay = True


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

        # Print RAM memory usage
        #print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

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
        possible_moves = self.state.all_possible_moves(flag_traps=AVOID_TRAPS)

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
        return the playout result status.possible values: STATUS_P0_WIN, STATUS_P1_WIN, STATUS_DRAW
        """
        cur_status = self.state.get_board_status()

        # if state is a loss, do not consider it
        if self.state.turn == 0:
            if cur_status == STATUS_P1_WIN:
                self.parent.state.win_score = MIN_SCORE # minimal integer value
                return cur_status
        else:
            if cur_status == STATUS_P0_WIN:
                self.parent.state.win_score = MIN_SCORE
                return cur_status
            
        # Simulate game with random moves 
        cur_node = deepcopy(self)
        while cur_status == STATUS_IN_PROGRESS: 
            cur_node = cur_node.random_play()
            cur_status = cur_node.state.get_board_status()
        return cur_status

    def random_play(self):
        """
        Current player turn performs a valid random move and game state 
        is updated accordingly.

        Return
        ----------
        return the updated game state.
        """
        moves = self.state.all_possible_moves(flag_traps=AVOID_TRAPS)

        if AVOID_TRAPS:
            # filter out traps
            trap_free_moves = [move for move in moves if not move[2]]
            if trap_free_moves:
                moves = trap_free_moves
        
        ##############################################
        # TODO: Fix bug in IMPROVED_RANDOM_PLAY mode #
        ##############################################

        # # The improved random play scan all possible moves
        # # to find a winning node.
        # if IMPROVED_RANDOM_PLAY:
        #     # shuffle moves
        #     inds = list(range(len(moves)))
        #     random.shuffle(inds)

        #     win_moves = list() # list of moves leading to win
        #     ok_moves = list() # list of moves that neither lead to win/loss
        #     loss_moves = list() # list of moves leading to loss

        #     # remember previous state information
        #     if self.state.turn == 0:
        #         prev_turn = 0
        #         prev_pos = self.state.p0_pos
        #     else:
        #         prev_turn = 1
        #         prev_pos = self.state.p1_pos
        #     prev_is_trap = self.state.trap

        #     tmp_state = deepcopy(self.state) 

        #     for ind in inds:
        #         random_pos, random_dir, random_is_trap = moves[ind]
        #         tmp_state = tmp_state.apply_move((random_pos, random_dir, random_is_trap), create_new_state=False)

        #         status = tmp_state.get_board_status()

        #         if status == prev_turn: # found a winning state, stop there
        #             win_moves.append((random_pos, random_dir, random_is_trap))
        #             break
        #         else:
        #             if status == (prev_turn + 1) % 2: # opponent's turn
        #                 loss_moves.append((random_pos, random_dir, random_is_trap))
        #             else: # non-terminal state
        #                 ok_moves.append((random_pos, random_dir, random_is_trap))
        #             # revert move
        #             tmp_state.revert_move((prev_pos,random_dir, prev_is_trap))

        #     del tmp_state
        #     gc.collect()

        #     if win_moves:
        #         random_pos, random_dir, random_is_trap = win_moves[0]
        #     elif ok_moves:
        #         random_pos, random_dir, random_is_trap = ok_moves[0]
        #     else:
        #         random_pos, random_dir, random_is_trap = loss_moves[0]
        
        # Normal random play
        # else:
        #     random_pos, random_dir, random_is_trap = random.choice(moves)

        random_pos, random_dir, random_is_trap = random.choice(moves)

        # Update position, turn and barriers
        self.state.apply_move((random_pos, random_dir, random_is_trap), create_new_state=False)
        return self

    def back_propagation(self, playout_result):
        """
        Bapropagate the playout result from current node to root node. 

        Parameters
        ----------  
        - playout_result: the playout result. possible values: STATUS_P0_WIN, STATUS_P1_WIN, STATUS_DRAW     
        """
        node = self
        while (node is not None):
            node.increment_visit()

            if (playout_result == STATUS_P0_WIN):
                node.add_score(WIN_SCORE) 
            elif (playout_result == STATUS_DRAW):
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
        return a random child node
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

            if (promising_node.state.get_board_status() == STATUS_IN_PROGRESS):
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
        self.trap = False # indicates if trap state or not

    def toggle_player(self):
        """
        Toggle player turn.
        """
        self.turn = (self.turn + 1) % 2


    def set_barrier(self, pos, barrier_dir, value=1):
        """
        Set barrier on chessboard. Important to set
        both barriers.
        
        Parameters
        ----------
        pos: tuple
        barrier_dir: int
        value: 1 for set, 0 for unset

        Return
        -------
        return the updated state
        """

        self.chess_board[pos[0], pos[1],barrier_dir] = value
        if barrier_dir == UP: # up
          if pos[0] - 1 >= 0:
            self.chess_board[pos[0]-1, pos[1],DOWN] = value

        elif barrier_dir == RIGHT: # right
          if pos[1] + 1 < self.chess_board.shape[1]:
            self.chess_board[pos[0], pos[1]+1,LEFT] = value

        elif barrier_dir == DOWN: # down
          if pos[0] + 1 < self.chess_board.shape[0]:
            self.chess_board[pos[0]+1, pos[1],UP] = value

        elif barrier_dir == LEFT: # left
          if pos[1] - 1 >= 0:
            self.chess_board[pos[0], pos[1]-1,RIGHT] = value
        return self
        

    def all_possible_moves(self, flag_traps=True):
        """
        Get all possible moves of the form ((y,x),dir) from current game state. (reachable and within max steps).

        Params
        ------
        Indicates if moves leading to traps must be flagged
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

            # trap detection
            is_trap = False
            if flag_traps:
                is_trap = self.is_trap(my_pos, barrier_dir)

            if self.chess_board[my_pos[0], my_pos[1], barrier_dir]:
                continue
            moves.append((my_pos, barrier_dir, is_trap))

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

                    # trap detection
                    is_trap = False
                    if flag_traps:
                        is_trap = self.is_trap(next_pos, barrier_dir)

                    moves.append(((next_pos, barrier_dir, is_trap)))

                visited.add(tuple(next_pos))
                move_queue.append((next_pos, cur_step + 1))
        return moves
    
    def apply_move(self, move, create_new_state=True):
        """
        Apply a move to current state and return new state.

        Params
        ------
        - move: tuple of the form (pos, dir, is_trap) where pos is the
                next position and dir is the barrier direction.
        
        Return
        -------
        return the new state created from the move.
        """ 
        pos, dir, is_trap = move

        # Append new state
        if create_new_state:
            new_state = deepcopy(self)
        else:
            new_state = self

        # Move player
        if new_state.turn == 0:
            new_state.p0_pos = pos
        else:
            new_state.p1_pos = pos

        # Add barrier
        new_state.set_barrier(pos, dir)

        # Toggle turn
        new_state.toggle_player()

        # Set trap flag
        new_state.trap = is_trap

        return new_state

    def revert_move(self, move):
        """
        Revert a move 

        Params
        ------
        - move: tuple of the form (pos, dir, is_trap) where pos is the
                previous position and dir is the barrier direction
                 to remove.
        
        Return
        -------
        return the state
        """ 

        # Toggle turn
        self.toggle_player()

        prev_pos, dir, prev_is_trap = move

        # Set Trap Flag
        self.trap = prev_is_trap

        # Move player
        if self.turn == 0:
            self.p0_pos = prev_pos
        else:
            self.p1_pos = prev_pos

        # Remove barrier
        self.set_barrier(prev_pos, dir, value=0)

        return self

    def is_trap(self, next_pos, dir):
        """
        Check whether the move would lead to a trap. A trap occurs when
        the player is surrounded by 3 walls. This 
        is a position that we tend to avoid.

        Params
        ------
        - next_pos: the move position that we want to verify.
        - dir: the barrier direction that we want to verify

        Return
        ------
        return True if move leads to a trap, False otherwise.
        """
        barriers = self.chess_board[next_pos]

        count = np.sum(barriers) + 1
        if count >= 3:
            return True
        else:
            return False

    def check_endgame(self):
        """
        (Method adapated from the check_endgame method from world.py)
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


    def get_board_status(self):
        """
        Return the current board status.
        
        Parameters
        ----------
        state: the game state

        Return
        ----------
        return the board state code. Possible values: STATUS_IN_PROGRESS, STATUS_P0_WIN, STATUS_P1_WIN, DRAW
        """
        is_endgame, p0, p1 = self.check_endgame()
        if is_endgame:
            if p0 > p1:
                status = STATUS_P0_WIN
            elif p1 > p0:
                status = STATUS_P1_WIN
            else:
                status = STATUS_DRAW
        else:
            status = STATUS_IN_PROGRESS
        return status

    
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