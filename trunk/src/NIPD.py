import random, copy, sys

# TODO: factor out the coevolutionary mechanism.

class NPlayerIteratedPrisonersDilemmaFitness():
    """N-player iterated prisoners dilemma.

    From title={{An Experimental Study of N-Person Iterated Prisoner's
    Dilemma Games}}, author={Yao, X. and Darwen, P.J.}

    Each player choses cooperation(C) or defection(D). D is dominant
    for each player. Dominant strategies intersect in a deficient
    equilibrium. If all players choose C the outcome is preferable
    from every players view to the one in which everyone chooses D,
    but noone is motivated to deviate unilaterally from D

    The number of cooperators in each game is n_c. The payoff for
    cooperation is 2n_c - 2 and payoff for defection is 2n_c + 1. If
    N_c cooperative moves are made from N moves of an n-player game,
    then the average per-round payoff a is: a = 1 + N_c/N(2n-3)
    """
    maximise = True
    COEVOLUTION = True

    def _lt(self, x, y):
        return x < y

    def _gt(self, x, y):
        return x > y

    def _eq(self, x, y):
        return x == y

    def __init__(self, iterations, players, number_of_games=1000):
        self.number_of_rounds = iterations
        self.n = players
        self.number_of_games = number_of_games

    def __call__(self, individuals):
        if len(individuals) < self.n:
            print("Number of players (arg 2 to NPlayerIteratedPrisonersDilemma)")
            print("is greater than populations size: %d > %d"
                  % (self.n, len(individuals)))
            sys.exit(1)

        # Compile everyones strategies
        for individual in individuals:
            try:
                c = compile(individual.phenotype,'<string>','exec')
                d = {}
                exec(c, d)
                individual.compiled_phenotype = d["f"]
            except TypeError as e:
                print(e)
                sys.exit(1)

        for individual in individuals:
            individual.fitness = 0
            individual.NIPD_games_played = 0
        # Players in each game are drawn randomly...
        for t in range(self.number_of_games):
            players = random.sample(individuals, self.n)
            self.NIPD(players)
        # ...so must divide each player's total payoff by his number
        # of games played.
        for individual in individuals:
            individual.fitness /= float(individual.NIPD_games_played)


    # NIPD represents a single game of multiple rounds with fixed
    # players.
    def NIPD(self, players):
        # Each player's history is blank at the beginning.
        history = [[] for player in players]

        for r in range(self.number_of_rounds):
            moves = []
            for i, player in enumerate(players):
                # TODO optimise by avoiding this copy.
                inputs = copy.copy(history)
                del inputs[i]
                moves.append(player.compiled_phenotype(history[i], inputs))
            ncooperators = moves.count(True)
            for i, (player, move) in enumerate(zip(players, moves)):
                # Save this player's move
                history[i].append(move)
                # Calculate this player's payoff for this round. True
                # means cooperate, False means defect.
                if move:
                    # 2*n_c, where n_c is number of cooperators among
                    # *other* players, ie must subtract this player.
                    payoff = 2 * (ncooperators - 1)
                else:
                    # 2*n_c + 1: defect always dominates cooperate
                    payoff = 2 * ncooperators + 1
                player.fitness += payoff
                player.NIPD_games_played += 1
