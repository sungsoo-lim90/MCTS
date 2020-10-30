from monte_carlo_tree_search import MCTS, Node
from tree import Tree

action = []
n = 100
for i in range(n):
	print(i)
	act = []
	#start with a patient with zero features
	tp = (-1,)*15
	node = Tree(tup=tp, terminal=False)
	tree = MCTS()
	while not node.terminal:
		#based on the given node, find the MCTS next state
		for _ in range(1000): 
			tree.train(node)
		next_node = tree.choose(node) #choose best score next state
		act.append([i for i, val in enumerate(zip(node.tup,next_node.tup)) if node.tup[i] != next_node.tup[i]])
		node = next_node
	act = [item for next_action in act for item in next_action]
	action.append(act)
    #if tree._simulate(node) == 0: #true positive and negative
    #	break

