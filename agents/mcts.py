import math

class MCTS:
    def __init__(self):
        pass

    # Call mcts search from current game state.
    # Keep track of time with a while loop to set an upper bound on the 
    # simulation execution time.
    def find_next_move(self, board, my_pos, adv_pos, max_step):
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


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = list()

class Tree:
    def __init__(self, state):
        self.root = Node(state)

class State:
    def __init__(self, board, my_pos, adv_pos, max_step, playerNo):
        self.chess_board = board.copy()
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self.playerNo = playerNo # agent is player 1, opponent is player 2

        # number of visits
        self.visit_count

        # number of wins
        self.win_score

        self.moves = self.all_possible_moves()

    def toggle_player(self):
        self.playerNo = 3 - self.playerNo

    def copy():
        #TODO state copy
        return



    # Construct a list of all possible moves from current state.
    # A move is represented as a tuple (x,y,dir) where x is the final
    # position in x, y is the final position on y axis and dir is the
    # barrier direction: 'u','d','l','r'.
    def all_possible_moves(self):
        # TODO
        pass

    # from the list of all possible moves, build and return all possible states
    def all_possible_states(self):
        states = list()
        for move in self.moves:
            #TODO
            pass
        return states

    # Play a random move among list of possible moves on the board.
    # Update state accordingly.
    def random_play(self):
        # TODO
        pass


# ---------------------- Utils ----------------------


# Upper confidence bound function
# wi: number of wins after the ith move
# ni: number of simulations after the ith move
# t: total number of simulations for the parent node
# c: exploration parameter
C = math.sqrt(2)

def uct_value(wi, ni, t,c=C):
    return wi/ni + c*math.sqrt(math.log(t)/ni)


BOARD_STATUS_IN_PROGRESS = 0
BOARD_STATUS_P1_WIN = 1
BOARD_STATUS_P2_WIN = 2
BOARD_STATUS_TIE = 3
# return the current board status
def board_status(board, my_pos, adv_pos):
    #TODO
    return