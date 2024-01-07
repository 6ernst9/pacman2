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


import random

import util
from game import Agent
from game import Directions
from pacman import GameState
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        # Food distance
        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / closestFoodDist

        # Ghost distance
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0 and ghostDist < 2:  # Ghost is dangerous
                score -= 10  # Large penalty for being too close to a ghost

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman's turn (maximizer)
                return max(minimax(1, depth, gameState.generateSuccessor(agent, action))
                           for action in gameState.getLegalActions(agent))
            else:  # Ghosts' turn (minimizer)
                nextAgent = agent + 1  # Next agent
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1

                return min(minimax(nextAgent, depth, gameState.generateSuccessor(agent, action))
                           for action in gameState.getLegalActions(agent))

        # Start with Pacman (agent 0) at depth 0
        maximum = float("-inf")
        bestAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            value = minimax(1, 0, gameState.generateSuccessor(0, action))
            if value > maximum or maximum == float("-inf"):
                maximum = value
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def alphaBeta(agent, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman's turn (maximizer)
                value = float("-inf")
                for action in gameState.getLegalActions(agent):
                    value = max(value, alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (minimizer)
                nextAgent = agent + 1  # Next agent
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1

                value = float("inf")
                for action in gameState.getLegalActions(agent):
                    value = min(value,
                                alphaBeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        alpha = float("-inf")
        beta = float("inf")
        bestAction = Directions.STOP
        value = float("-inf")

        for action in gameState.getLegalActions(0):
            newValue = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if newValue > value:
                value = newValue
                bestAction = action
            alpha = max(alpha, value)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def expectimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            if agent == 0:  # Pacman's turn (maximizer)
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, action))
                           for action in gameState.getLegalActions(agent))
            else:  # Ghosts' turn (expectation/chance node)
                nextAgent = agent + 1  # Next agent
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    depth += 1

                actions = gameState.getLegalActions(agent)
                return sum(expectimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                           actions) / len(actions)

        # Start with Pacman (agent 0) at depth 0
        return max(gameState.getLegalActions(0), key=lambda x: expectimax(1, 0, gameState.generateSuccessor(0, x)))


def betterEvaluationFunction(currentGameState: GameState):
    """
    A more advanced evaluation function.
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    capsules = currentGameState.getCapsules()

    # Basic score from state
    score = currentGameState.getScore()

    # Calculate the distance to the closest food pellet
    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        score += 1.0 / min(foodDistances)

    # Consider the distance to each ghost and their scared state
    for ghostState in newGhostStates:
        ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
        if ghostState.scaredTimer > 0:
            # Ghost is scared; prefer closer distance to eat ghost
            score += max(ghostState.scaredTimer - ghostDistance, 0)
        else:
            # Ghost is not scared; maintain a safe distance
            if ghostDistance < 4:
                score -= 10 * (4 - ghostDistance)

    # Consider the distance to capsules
    if capsules:
        capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
        closestCapsuleDistance = min(capsuleDistances)
        score += 5.0 / closestCapsuleDistance

    # Penalize the game state by the number of remaining food pellets
    remainingFood = len(foodDistances)
    score -= 2 * remainingFood

    # Additional penalty for each remaining capsule
    score -= 20 * len(capsules)

    return score


# Abbreviation
better = betterEvaluationFunction
