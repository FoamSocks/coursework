# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from math import exp

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

# Node class to solve saved path problem
class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state 
        self.action = action
        self.parent = parent
        self.path_cost = path_cost

    def getPath(self):
        node = self
        path = []
        while node and node.parent:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def ourHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.
    """
    initial_x, initial_y = problem.startState
    goal_x, goal_y = problem.goal
    state_x, state_y = state

    manhattan_calc = abs(state_x - goal_x) + abs( state_y - goal_y)
    man_mag_init = abs(initial_x- goal_x) + abs( initial_y- goal_y)
    man_mag_current = abs(state_x - goal_x) + abs( state_y - goal_y)

    ratio = (man_mag_current/man_mag_init)
    k1 = .7
    k2 = k1 * ratio
    val = manhattan_calc *(1 + k1 * ratio) - k2 * (1-ratio)

    return val 

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
    Berkeley website: http://ai.berkeley.edu/search.html
    python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
    python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=ourHeuristic
    '''
    startState = problem.getStartState()

    # PriorityQueue in util.py: push(item, priority), pop(), update(item, priority)
    frontier = util.PriorityQueue()
    reached = list()

    # no path cost for root 
    start_fCost = heuristic(startState, problem)

    # add start state to frontier
    # Node(state, parent=None, action=None, path_cost=0)
    startNode = Node(startState)
    frontier.push(startNode, start_fCost)

    while frontier:
        # node is a Node object, position tuple is in node.state, parent is node.parent
        node = frontier.pop()

        if node.state not in reached:
            reached.append(node.state)

            # check goal state after pop
            if problem.isGoalState(node.state):
                # node.path contains reversed list of parents for node == path back
                path = node.getPath()
                return path

            parentPathCost = node.path_cost

            # getSuccessors() already accounts for walls/valid actions -- searchAgents.py
            successors = problem.getSuccessors(node.state) #returns [successor, action, stepcost]
            for successor in successors:
                child_pos = successor[0]
                child_action = successor[1]
                child_step_cost = successor[2]
                if child_pos not in reached and child_pos not in frontier.heap:
                    childPathCost = parentPathCost + child_step_cost
                    child_fCost = float(childPathCost) + heuristic(child_pos, problem)
                    childNode = Node(child_pos, node, child_action, childPathCost)
                    frontier.update(childNode, child_fCost)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch