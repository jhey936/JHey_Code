# coding=utf-8
"""
Contains various methods to select parents from the pool on which to perform mating.
"""
import numpy as np


class BaseSelector(object):
    """This is the base selector class object.

    Other selectors must inherit and then overload these class methods.

    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialises the selector
        """
        pass

    def __call__(self, population, *args, **kwargs) -> list:
        """Allows instances of selectors to be called like functions once instantiated"""
        return self.get_parents(population, *args, **kwargs)
    #
    # def __repr__(self) -> float:
    #     raise NotImplementedError

    def get_parents(self, population: list, number_of_parents=2, *args, **kwargs) -> iter:
        """Method to return an iterable (tuple or list) of parents selected from the population

        This base class just returns random parents (fully stochastic selection)

        Attributes:
            population: list of clusters representing the current population
            args: other positional arguments
            kwargs: other keyword arguments

        Returns:
            parents, tuple of Cluster objects selected for mating

        """
        return np.random.choice(population, size=number_of_parents, replace=False)


class RouletteWheelSelection(BaseSelector):
    """
    Implements the Roulette Wheel method for selection of parents for mating.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
  
    def get_parents(self, population: list, weights=None,
                    number_of_parents=2, replacement=False,
                    *args, **kwargs) -> list:
        """Returns parents selected randomly weighted according to their cost

        Attributes:
            population: List of members of the population, required
            weights: List of weights, optional (default = weights based of cost)
            number_of_parents: int, Number of parents to be returned, optional, (default = 2)
            replacement: bool, setting to False allows the same parent to be returned multiple times,
                                    optional, (default = True)
            *args: list, optional, other positional arguments
            **kwargs: dict, optional, other keyword arguments

        Returns:
            tuple() of clusters for mating

        """
        p = []
        for member in population:
            p.append(member.cost)

        p = np.array(p)/np.sum(p)

        try:
            parents = np.random.choice(population, size=number_of_parents, replace=replacement, p=p)
        except ValueError:
            p = []
            for member in population:
                p.append(member.cost)

            p_norm = np.array(p) / np.sum(p)

            raise ValueError(f"p = {p}, sum_p = {np.sum(p)}, p_norm = {p_norm}")

        return list(parents)


class BoltzmannSelector(BaseSelector):
    """
    Implements Boltzmann selection:

    See:
    https://pdfs.semanticscholar.org/cb68/1ec48a4882c07037864588ce841e3be2a4cc.pdf

    P_{i} = 1/N * exp(-dE_{i}/kbT)

    N = sum(exp(-dE/kbT))

    Attributes:
        kb: Boltzmann's constant in whatever units make sense for your cost function.
        T: The current theoretical temperature. Modify to change distribution over time.


    """
    def __init__(self, kb: float, temperature=1000.0):
        """Initialises Boltzmann Tournament selection class

        Attributes:
            kb: float, required, boltzmann's constant in whatever units make sense for your cost function
            temperature: float, optional, temperature for weighting

        """
        self.T = temperature
        self.kb = kb
        super().__init__()

    def get_parents(self, population: list, number_of_parents: int=2,
                    temperature: float=None, *args, **kwargs) -> list:
        """ Selects parents according to a Boltzmann distribution

        Attributes:
            population: list(Cluster), required, population to draw parents from
            number_of_parents: int, optional, number of parents to return (default=2)
            temperature: float, optional, new temperature -- will be saved for future calls (default=None)
            *args: list, optional, other positional arguments
            **kwargs: dict, optional, other keyword arguments

        Returns:
            parents: tuple, selected parents

        """
        # Updates self.temperature going forward if you want to change the distribution dynamically
        if temperature is not None:
            self.T = temperature

        costs = [m.cost for m in population]
        gm_cost = min(costs)
        beta = 1.0/(self.kb*self.T)

        weights = []
        weight_sum = 0.0

        for member in population:
            minus_cost_delta = gm_cost - member.cost
            weight = np.exp(minus_cost_delta*beta)
            weights.append(weight)
            weight_sum += weight

        weights = np.array(weights)
        weights = weights/weight_sum

        return list(np.random.choice(population, size=number_of_parents, replace=False, p=weights))


class RankSelector(BaseSelector):
    """
    Implements selection based on the Linear rank selection methods as described here:
    http://shodhganga.inflibnet.ac.in/bitstream/10603/32680/16/16_chapter%206.pdf
    """
    def __init__(self) -> None:
        super().__init__()

    def get_parents(self, population: list, max_prob=1.5,
                    number_of_parents: int = 2, *args, **kwargs) -> iter:
        """

        Args:
            population: list(Cluster), required, population to draw parents from
            number_of_parents: int, optional, number of parents to return (default=2)
            max_prob: float, optional, The probability of selecting the most highly ranked individual.
                         Must lie in interval [1, 2)
            *args: list, optional, other positional arguments
            **kwargs: dict, optional, other keyword arguments

        Returns:
            tuple() of parents for mating

        """

        min_prob = 2 - max_prob
        pop_size = len(population)
        weights = []

        population = sorted(population, key=lambda x: 1/x.cost)  # = population[::-1]

        for rank in list(range(1, pop_size+1)):
            weights.append((1/pop_size) * (min_prob + ((max_prob - min_prob) * ((rank - 1) / (pop_size - 1)))))

        return list(np.random.choice(population, size=number_of_parents, replace=False, p=weights))


class TournamentSelector(BaseSelector):
    """Implements tournament selection

    Tournaments are organised by randomly selecting k members from the population (without replacement)

    Probabilities are then calculated according to [p*(1-p)**0, p*(1-p)**1, p*(1-p)**2, ... p*(1-p)**N]
        with N equal to the competitors rank within the competition

    A winner of the tournament is then selected according to this probability
    """
    def __init__(self, pop_in_tournament: float=0.3, p: float=0.5, k: int=None) -> None:
        """Implements tournament selection

        Tournaments are organised by randomly selecting k members from the population (without replacement)

        Probabilities are then calculated according to [p*(1-p)**0, p*(1-p)**1, p*(1-p)**2, ... p*(1-p)**N]
            with N equal to the competitors rank within the competition

        A winner of the tournament is then selected according to this probability

        When p=1 then selection is deterministic
        When k=1 then selection is wholly stochastic

        Args:
            pop_in_tournament: float, optional, fraction of the population to use for k (default=0.3)
            p: float, optional, probability of selecting the fittest competitor in the tournament. (default=0.5)
            k: int, optional, number of competitors in each tournament. Overloads pop_in_tournament. (default=None)
        """
        self.pop_in_tournament = pop_in_tournament
        self.p = p
        self.k = k
        super().__init__()

    def get_parents(self, population: list, number_of_parents: int=2, *args, **kwargs) -> list:
        """Returns parents selected according to tournament selection

        Tournaments are organised by randomly selecting k members from the population (without replacement)

        Probabilities are then calculated according to [p*(1-p)**0, p*(1-p)**1, p*(1-p)**2, ... p*(1-p)**N]
            with N equal to the competitors rank within the competition

        A winner of the tournament is then selected according to this probability

        When p=1 then selection is deterministic
        When k=1 then selection is wholly stochastic

        Args:
            population:
            number_of_parents:
            *args: list, optional, other positional arguments
            **kwargs: dict, optional, other keyword arguments

        Returns:
            tuple() of parents for mating

        """

        parents = []

        for parent in range(number_of_parents):
            parents.append(self._tournament(population))

        return list(parents)

    def _tournament(self, population: list):
        """Performs the actual tournament

        The actual implementation and variables

        Args:
            population: list, required, the population to be selected from

        Returns:
            tournament_winner: A single member of the population selected according to the formula

        """

        if self.k is None:
            k = int(self.pop_in_tournament * len(population))
        else:
            k = self.k

        competitors = sorted(np.random.choice(population, size=k, replace=False), key=lambda x: x.cost)

        # costs = [m.cost for m in competitors]
        weights = [self.p*((1-self.p)**n) for n in range(k)]

        weights /= np.sum(weights)
        tournament_winner = np.random.choice(competitors, size=1, p=weights)[0]
        return tournament_winner
