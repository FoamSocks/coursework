# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.

      python pacman.py -p ReflexAgent -l testClassic
      python autograder.py -q q1
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        oldPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        import math

        #Ghost Distance
        ghostscore=0
        oldGhostPositions = currentGameState.getGhostPositions()
        newGhostPositions = successorGameState.getGhostPositions()
        newminghostdist=800
        th=2
        for pos in newGhostPositions:
            newghostdistance = math.sqrt((newPos[0]-pos[0])**2 + (newPos[1]-pos[1])**2)
            if newghostdistance<newminghostdist:
                newminghostdist=newghostdistance
        oldminghostdist=800
        for pos in oldGhostPositions:
            oldghostdistance = math.sqrt((oldPos[0]-pos[0])**2 + (oldPos[1]-pos[1])**2)
            if oldghostdistance<oldminghostdist:
                oldminghostdist=oldghostdistance
        #Ghost Scared?
        for time in newScaredTimes:
            if time==0: #not scared = stay away
                if newminghostdist<th:
                    ghostscore=-1
            if time>0: #scared = chase it
                if newminghostdist<oldminghostdist:
                    ghostscore=3

        #Old Food distance
        foodscore=0
        oldminpos=800
        oldclosestfood=None
        for row in range(oldFood.height): #y
            for col in range(oldFood.width): #x
                if oldFood[col][row]:
                    fdpos=math.sqrt((oldPos[0]-col)**2 + (oldPos[1]-row)**2)
                    if fdpos<oldminpos:
                        oldminpos=fdpos
                        oldclosestfood=tuple((col,row))
        #New Food distance
        newmindis=800
        newclosestfood=None
        for row in range(newFood.height): #y
            for col in range(newFood.width): #x
                if newFood[col][row]:
                    fddis=math.sqrt((newPos[0]-col)**2 + (newPos[1]-row)**2)
                    # fddis=abs(newPos[0]-col) + abs(newPos[1]-row)
                    if fddis<newmindis:
                        newmindis=fddis
                        newclosestfood=tuple((col,row))
        #Food Score calculation
        if newmindis<oldminpos:
            foodscore=1
        
        #Old Capsule distance
        capsulescore=0
        capPositions = currentGameState.getCapsules() #constant positions
        oldmincapdis=800
        newmincapdis=800
        for pos in capPositions:
            newcapdistance = math.sqrt((newPos[0]-pos[0])**2 + (newPos[1]-pos[1])**2)
            if newcapdistance<newmincapdis:
                newmincapdis=newcapdistance
        for pos in capPositions:
            oldcapdistance = math.sqrt((oldPos[0]-pos[0])**2 + (oldPos[1]-pos[1])**2)
            if oldcapdistance<oldmincapdis:
                oldmincapdis=oldcapdistance
        #Capsule Score calculation
        if newmincapdis<oldmincapdis:
            capsulescore=2

        return (successorGameState.getScore()+foodscore+ghostscore+capsulescore)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)

      python autograder.py -q q2
    """
    def maxValue(self, gameState, currDepth):
        pacman = 0
        # stop if terminal state or depth exhausted
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState), None #returns 'v' but no Move
        else:
            v = float('-inf')
            for action in gameState.getLegalActions(pacman):
                successor = gameState.generateSuccessor(pacman, action)
                v2, a2 = self.minValue(successor, 1, currDepth)
                if v2 > v:
                    v = v2
                    move = action
            return v, move

    def minValue(self, gameState, ghostNo, currDepth):
        # must account for multiple ghosts -> multiple min layers 
        numGhosts = gameState.getNumAgents()-1
        if gameState.isWin() or gameState.isLose():
            '''
            final=self.evaluationFunction(gameState)
            print '      Ghost ',ghostNo,'terminal: Win=',gameState.isWin(), ', Lose=',gameState.isLose(), ', Depth=',self.depth, ', utility=',final
            '''
            return self.evaluationFunction(gameState), None #returns 'v' but no Move
        elif ghostNo == numGhosts:
            # last ghost -> back to pacman (max)
            v = float('inf')
            for action in gameState.getLegalActions(ghostNo): #expand tree below last ghost
                successor = gameState.generateSuccessor(ghostNo, action)
                v2, a2 = self.maxValue(successor, currDepth+1)
                if v2 < v:
                    v = v2
                move = action
            return v, move
        else:
            # next agent is ghost, continue to min
            v = float('inf')
            newGhost = ghostNo+1
            for action in gameState.getLegalActions(ghostNo):
                successor = gameState.generateSuccessor(ghostNo, action)
                v2, a2 = self.minValue(successor, newGhost, currDepth)
                if v2 < v:
                    v = v2
                    move = action
            return v, move


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        
        # ply number is depth: depth = 2, all agents move twice.        
        # if numAgents = 7, 0 is pacman and 1-6 are 6 ghosts
        # dispatch logic is within minValue and maxValue
        # starts with pacman (max)
        #determines initial move (top of tree)
        value, move = self.maxValue(gameState, 0)
        return move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, currDepth, alpha, beta):
        # track alpha and beta by passing to max or min functions through layers of recursion
        move = None
        pacman = 0
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState), None #returns 'v' but no Move
        else:
            v = float('-inf')
            for action in gameState.getLegalActions(pacman):
                successor = gameState.generateSuccessor(pacman, action)
                v2, a2 = self.minValue(successor, 1, currDepth, alpha, beta)
                if v2 > v:
                    v = v2
                    move = action
                # added for alpha/beta pruning
                if v2 > beta:
                    return v, move
                alpha = max(alpha, v2)
            return v, move

    def minValue(self, gameState, ghostNo, currDepth, alpha, beta):
        move = None
        numGhosts = gameState.getNumAgents()-1
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), None #returns 'v' but no Move
        elif ghostNo == numGhosts:
            v = float('inf')
            for action in gameState.getLegalActions(ghostNo): #expand tree below last ghost
                successor = gameState.generateSuccessor(ghostNo, action)
                v2, a2 = self.maxValue(successor, currDepth+1, alpha, beta)
                if v2 < v:
                    v = v2
                    move = action
                # added for alpha/beta pruning
                if v2 < alpha:
                   return v, move
                beta = min(beta, v2)
            return v, move
        else:
            v = float('inf')
            newGhost = ghostNo+1
            for action in gameState.getLegalActions(ghostNo):
                successor = gameState.generateSuccessor(ghostNo, action)
                v2, a2 = self.minValue(successor, newGhost, currDepth, alpha, beta)
                if v2 < v:
                    v = v2
                    move = action
                # added for alpha/beta pruning
                if v2 < alpha:
                    return v, move
                beta = min(beta, v2)
            return v, move


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # init alpha to neg infinity and beta to pos infinity
        alpha = float('-inf')
        beta = float('inf')
        value, move = self.maxValue(gameState, 0, alpha, beta)
        return move

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def maxValue(self, gameState, currDepth):
        move = None
        pacman = 0
        if gameState.isWin() or gameState.isLose() or currDepth == self.depth:
            return self.evaluationFunction(gameState), None #returns 'v' but no Move
        else:
            v = float('-inf')
            for action in gameState.getLegalActions(pacman):
                successor = gameState.generateSuccessor(pacman, action)
                v2 = self.expValue(successor, 1, currDepth)
                if v2 > v:
                    v = v2
                    move = action
            return v, move

    def expValue(self, gameState, ghostNo, currDepth):
        numGhosts = gameState.getNumAgents()-1
        numLegalActions = len(gameState.getLegalActions(ghostNo))
        
        if numLegalActions == 0:
            probability = float(0)
        else:
            # all actions have equal probability
            probability = float(1)/float(numLegalActions)
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        elif ghostNo == numGhosts:
            v = float(0)
            for action in gameState.getLegalActions(ghostNo):
                successor = gameState.generateSuccessor(ghostNo, action)
                successorValue, a2 = self.maxValue(successor, currDepth+1)
                v+=probability*float(successorValue)
        else:
            v = float(0)
            newGhost = ghostNo+1
            for action in gameState.getLegalActions(ghostNo):
                successor = gameState.generateSuccessor(ghostNo, action)
                successorValue = self.expValue(successor, newGhost, currDepth)
                v+=probability*float(successorValue)
        return v 

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.

         Idea: replacing min nodes with chance nodes (model all ghosts moving randomly)
        """
        value, move = self.maxValue(gameState, 0)
        return move

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # Idea: 
    # calc nearest food pellet in state
    #   for nearest food, execute A* to find optimal path, 
    #   weight foodscore according to A* path length
    # Implement linear scoring function similar to #1
    # Priority:
    #   Danger ghost threshold distance
    #   Scared Ghost optimal path
    #   Capsule optimal path
    #   Food optimal path
    import math
    import sys

    foodScore = 0
    ghostScore = 0
    capsuleScore = 0    
    pacPos= currentGameState.getPacmanPosition()
    foodArr = currentGameState.getFood()
    capPositions = currentGameState.getCapsules() #constant positions
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    eps = sys.float_info.epsilon

    wtGhostnotscared=-1
    wtGhostscared=3
    wtFood=3
    wtCap=2

    #Ghost distance
    GhostPositions = currentGameState.getGhostPositions()
    th=1
    minghostdist=800
    for pos in GhostPositions:
        ghostdistance = math.sqrt((pacPos[0]-pos[0])**2 + (pacPos[1]-pos[1])**2)
        if ghostdistance<minghostdist:
            minghostdist=ghostdistance
    #Ghost Scared?
    for time in scaredTimes:
        if time==0: #not scared = stay away
            if minghostdist<th:
                ghostScore=wtGhostnotscared*float(1)/float(minghostdist+1)
        if time>0: #scared = chase it
            ghostScore=3*float(1)/float(minghostdist+1)

    #Food distance
    foodminpos=800
    for row in range(foodArr.height): #y
        for col in range(foodArr.width): #x
            if foodArr[col][row]:
                foodpos=math.sqrt((pacPos[0]-col)**2 + (pacPos[1]-row)**2)
                if foodpos<foodminpos:
                    foodminpos=foodpos
    #Food Score calculation
    foodScore=3*float(1)/float(foodminpos+1)
    
    #Old Capsule distance
    mincapdis=800
    for pos in capPositions:
        capdistance = math.sqrt((pacPos[0]-pos[0])**2 + (pacPos[1]-pos[1])**2)
        if capdistance<mincapdis:
            mincapdis=capdistance
    #Capsule Score calculation
    capsuleScore=2*float(1)/float(mincapdis+1)


    return float(currentGameState.getScore()) + foodScore + ghostScore + capsuleScore

# Abbreviation
better = betterEvaluationFunction