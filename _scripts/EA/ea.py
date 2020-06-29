from abc import abstractmethod
from multiprocessing import Pool

from .callbacks import CallbackList
import random
import numpy as np
import copy
import configparser as cp
import os
import time

#global threadpool
#threadpool = Pool(processes=1)

class Setup(object):
    '''
    Class that implements a simple setup.

    '''

    def __init__(self):
        self.multiprocesses = 1
        self.population_size = 50
        self.random_seed = 0
        self.generations = 10
        self.torn_size = 3
        self.mutation_rate = 0.05
        self.crossover_rate = 0.7
        self.objective = max
        self.individuals_dict = {}

    def load_setup(self):
        pass

    def save_setup(self, path):
        parser = cp.ConfigParser()
        parser['Setup'] = { 
                           'multiprocesses' : str(self.multiprocesses),
                           'population_size' : str(self.population_size),
                           'random_seed' : str(self.random_seed),
                           'generations' : str(self.generations),
                           'torn_size' : str(self.torn_size),
                           'mutation_rate' : str(self.mutation_rate),
                           'crossover_rate' : str(self.crossover_rate),
                           'multiprocesses' : str(self.multiprocesses)
                           }
        
        temp_dict = {}
        for key in self.individuals_dict.keys():
            temp_dict[key] = str(self.individuals_dict[key])
        parser['Individual'] = temp_dict
        
        if not os.path.isdir(path):
            os.makedirs(path, mode=0o777)
        with open(os.path.join(path, "setup.cfg"), "w+") as f:
            parser.write(f)


class Selection(object):
    '''
        Simple Stub for the Selection Algorithm
    '''

    def __init__(self, ea_ref):
        self.ea_ref = ea_ref

    def select(self, current_population):
        new_pop = list(current_population)
        return new_pop, []


class Tournament(Selection):

    def __init__(self, ea_ref):
        super(Tournament, self).__init__(ea_ref)

        self.torn_size = 2 if self.ea_ref.setup is None else self.ea_ref.setup.torn_size
        self.elitism = 1
        self.objective = max if self.ea_ref.setup is None else self.ea_ref.setup.objective

    def select(self, current_population):
        fitnesses = [individual.fitness for individual in current_population]
        elite_indexes = np.argpartition(fitnesses, -self.elitism)[-self.elitism:]

        # clone elite for elite popuplation
        elite_population = [copy.deepcopy(current_population[elite_index]) for elite_index in elite_indexes]
        new_population = elite_population + self.tournament(current_population)

        return new_population, elite_population

    def tournament(self, current_population):
        pop_size = len(current_population) - self.elitism
        fitnesses = np.array([individual.fitness for individual in current_population])
        torn_indexes_v = [np.random.choice(pop_size, self.torn_size, replace=False) for i in range(pop_size)]
        if self.objective == max:
            new_population = [current_population[torn_indexes[np.argmax(fitnesses[torn_indexes])]] for torn_indexes in torn_indexes_v]
        else:
            new_population = [current_population[torn_indexes[np.argmin(fitnesses[torn_indexes])]] for torn_indexes in torn_indexes_v]

        return new_population

class FullRandomRestart(Selection):

    def __init__(self, ea_ref):
        super(FullRandomRestart, self).__init__(ea_ref)

    def select(self, current_population):
        new_pop = [self.ea_ref.individual(i, self.ea_ref, self.ea_ref.setup.individuals_dict) for i in range(len(current_population))]
        for individual in new_pop:
            individual.initialize()
        return new_pop, []


class Individual():
    '''
        Simple Stub for an individual
    '''

    @abstractmethod
    def __init__(self, id, ref, individuals_dict):
        self.id = id
        self.fitness = 0.0
        self.individuals_dict = individuals_dict
        self.model = ref # model to which it belongs

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def genotype_to_phenotype(self):
        pass

    @abstractmethod
    def mutate(self, probability=None):
        pass

    @abstractmethod
    def crossover(self, other):
        pass

    @abstractmethod
    def genotype(self):
        pass


def randomeval(individual):
    individual.fitness = random.random()
    individual.evaluated = True


def zeroeval(individual):
    individual.fitness = 0
    individual.evaluated = True


class EA():

    def __init__(self):
        self.setup = None
        self.generator = None
        self.elite_population = []
        self.current_population = []
        self.best_individual = None
        self.current_generation = 0
        self.individual = None
        self.selection = None
        self.evaluation = None
        self.pool = None

    def initialize(self, setup=Setup(), individual_ref=Individual, selection_ref=Selection, evaluation_ref=randomeval, 
                   callbacks=None):

        if callbacks is not None:
            CallbackList(callbacks).on_initialize_begin()

        # set random seed
        self.setup = setup
        random.seed(setup.random_seed)
        np.random.seed(setup.random_seed)
        self.individual = individual_ref
        self.selection = selection_ref(self)
        self.evaluation = evaluation_ref
        n_processes = self.setup.multiprocesses if self.setup is not None else 1

        # multi processing disabled..
        #self.pool = Pool(processes=n_processes)

        if self.setup.individuals_dict is None:
            self.individuals_dict = {}
        else:
            self.individuals_dict = self.setup.individuals_dict

        # print(n_processes)
        #

        self.current_population = [individual_ref(i, self, self.individuals_dict) for i in range(setup.population_size)]

        for individual in self.current_population:
            individual.initialize()
            # print(individual.genotype())
            # print(individual.fitness)

        if callbacks is not None:
            CallbackList(callbacks).on_initialize_end()

    def evaluate(self, callbacks=None):
        if callbacks is not None:
            CallbackList(callbacks).on_evaluation_begin(population=self.current_population)

        # single process..
        # print(self.setup.multiprocesses)
        if self.setup.multiprocesses == 1:
            for indiv in self.current_population:
                # print(indiv)
                self.evaluation(indiv)
        else:
            # multiprocess fixme ! use the pool as a global not as a member.. to allow the mapping..
            # func = partial(f, a, b)
            print('warning not finished yet...')
            #self.current_population = threadpool.map(self.evaluation, self.current_population)

        if callbacks is not None:
            CallbackList(callbacks).on_evaluation_end(population=self.current_population)

    def variation_operators(self, n_to_generate):
        altered_population = []
        pop_size = len(self.current_population)
        
        for i in range(n_to_generate):
            indexes = np.random.choice(pop_size, 2, replace=False)
            # copy individual
            ind_copy = self.current_population[indexes[0]].copy()
            if np.random.rand() < self.setup.crossover_rate:  # perform crossover of src ([0]) with dst ([1])
                # crossover with other parent
                ind_copy.crossover(self.current_population[indexes[1]], 0.5)
                altered_population.append(ind_copy)
            else:
                # copy unchanged
                altered_population.append(ind_copy)
        
        for i in range(n_to_generate):
            # perform mutation after crossover
            altered_population[i].mutate(self.setup.mutation_rate)

        return altered_population

    @abstractmethod
    def check_termination_criterion(self):
        return True

    def evolve(self, callbacks=None):

        # evaluate population
        self.evaluate(callbacks)

        # update population, sort by fitness and update id by position on the population
        self.current_population.sort(reverse=(True if self.setup.objective==max else False))
        for ind_i in range(len(self.current_population)):
            self.current_population[ind_i].id = ind_i
        
        while not self.check_termination_criterion():

            self.current_generation += 1

            if callbacks is not None:
                CallbackList(callbacks).on_generation_begin(self.current_generation)
            
            # perform selection
            self.current_population, self.elite_population = self.selection.select(self.current_population)
            
            # Apply variation operators
            self.current_population = self.variation_operators(len(self.current_population) - len(self.elite_population))
            self.current_population = self.elite_population + self.current_population
            
            # evaluate population
            self.evaluate(callbacks)

            # update population, sort by fitness and update id by position on the population
            self.current_population.sort(reverse=(True if self.setup.objective==max else False))
            for ind_i in range(len(self.current_population)):
                self.current_population[ind_i].id = ind_i

            if callbacks is not None:
                CallbackList(callbacks).on_generation_end(self.current_generation)
