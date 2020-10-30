from collections import namedtuple
from random import choice, uniform
from monte_carlo_tree_search import MCTS, Node

_T = namedtuple("node", "tup terminal")

class Tree(_T, Node):

    def find_children(node):
        if node.terminal: 
            return set()
        #all next nodes with unacquired features
        return {
            node.make(i) for i, value in enumerate(node.tup) if value == -1
        }

    def find_random_child(node):
        if node.terminal:
            return None 
        empty_spots = [i for i, value in enumerate(node.tup) if value == -1]
        return node.make(choice(empty_spots)) #one random child node

    def reward(node):
        
        # So far, reward is always calculated at the terminal state
        # How to incoroporate costs and rewards at each state?

        # classification reward matrix
        # CHD/CHD = 1 (probCHD) true positive
        # noCHD/noCHD = 1 (probnoCHD) true negative
        # noCHD/CHD = -1 (probCHD) false negative
        # CHD/noCHD = -1 (probnoCHD) false positive

        # test costs
        # divide by total_cost - add up to 1 if terminal state
        # features # 0 - 8 = $1
        # features # 9 - 14 = $7
        # total costs = 42 + 9 = 51

        # probCHD from total data set
        # probCHD = 0.15235229759299782

        # probCHD from train data set 
        # For patients with all features measured, this probability gives the percentage with CHD
        probCHD = 0.14988104678826328
        classification = 0
        cost = 0
        total_cost = 51

        for i in range(len(node.tup)):
            if node.tup[i] == 1:
                if i <= 8:
                    cost += 1
                else:
                    cost += 7
            else:
                cost += 0
        
        if node.terminal: #terminal
            if uniform(0,1) < probCHD:
                if uniform(0,1) < 0.5:
                    classification = 1 #CHD/CHD
                else:
                    classification = -1 #noCHD/CHD
            else:
                if uniform(0,1) < 0.5:
                    classification = 1 #noCHD/noCHD
                else:
                    classification = -1 #CHD/noCHD

        return_value = classification - cost/total_cost
        return return_value

    def is_terminal(node):
        return node.terminal

    def make(node,index):
        #set the newly aquired feature to be 1
        tup = node.tup[:index] + (1,) + node.tup[index + 1 :] 
        is_terminal = not any(v == -1 for v in tup) 
        return Tree(tup, is_terminal) #new node