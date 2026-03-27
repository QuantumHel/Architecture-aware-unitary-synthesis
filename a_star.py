from astar import AStar

class BasicAStar(AStar):
    def __init__(self, nodes):
        self.nodes = nodes

    def neighbors(self, n):
        return list(self.nodes[n])

    def distance_between(self, n1, n2):
        return 1
            
    def heuristic_cost_estimate(self, current, goal):
        return 1
    
    def is_goal_reached(self, current, goal):
        return current == goal