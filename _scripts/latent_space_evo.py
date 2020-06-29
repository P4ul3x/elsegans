from .utils import custom_reshape, int_to_shape

from .EA.ea import EA, Setup, Individual, Selection, Tournament, FullRandomRestart
from .EA.callbacks import Callback, CallbackList

#from image_path_selector import dataset_similarity
#from image_similarity import rmse, ssim, euclidean_distance
#from image_similarity import norm_cross_correlation as ncc 
#from image_similarity import zero_mean_norm_cross_correlation as zmncc 

#from keras.models import load_model

import numpy as np
import math
import argparse
import time
import datetime
import random
import os
import cv2
import copy

class LatentIndividualCallback(Callback):
    def __init__(self, directory, prefix=""):
        super(LatentIndividualCallback, self).__init__()

        self.directory = directory
        self.prefix = prefix
        self.prevBOG = None

    def on_evaluation_end(self, population, logs=None):
        super(LatentIndividualCallback, self).on_evaluation_end(population, logs)
        #print('finished the evaluation of generation ', self.model.current_generation)

        # update best individual overall
        objective = self.model.setup.objective
        if self.model.best_individual is None:
            self.model.best_individual = objective(self.model.current_population)
        else :
            self.model.best_individual = objective(self.model.best_individual, objective(self.model.current_population))

        self.export_stats()

        if (self.model.current_generation==0 or self.model.current_generation==self.model.setup.generations):
            self.export_best_png()

        if (self.model.current_generation==self.model.setup.generations):
            np.save(os.path.join(self.directory, 'best-overall'), self.model.best_individual.genes)

        print('total_time ', time.time() - start)

    def export_stats(self):

        saveBOG = False
        # simple exporter per generation...
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory, mode=0o777)
        with open(os.path.join(self.directory, '{}-{}-seed-{}.csv'.format(self.prefix, timestamp, self.setup.random_seed)), 'a') as stats_f:
            if self.model.current_generation == 0:
                # diversity_per_gene = []
                # for gene in range(len(self.model.current_population[0].genes)):
                #     diversity_per_gene.append(str(gene))
                # diversity_per_gene_str = '_gene,'.join(diversity_per_gene)
                stats_f.write('generation,avg,std,min,max,best_individual_generation,best_individual_overall\n')

            fitness_values = [t_indiv.fitness for t_indiv in self.model.current_population]
            popavg = np.mean(fitness_values)
            popstd = np.std(fitness_values)
            popmin = np.min(fitness_values)
            popmax = np.max(fitness_values)

            best_individual = self.model.setup.objective(self.model.current_population)
            if self.prevBOG is None or self.model.setup.objective([best_individual, self.prevBOG]).fitness != self.prevBOG.fitness:
                self.prevBOG = best_individual
                saveBOG = True


            print('S[{}] G[{}] BOGeneration {} BOAll {}'.format(self.setup.random_seed, self.model.current_generation, best_individual.fitness, self.model.best_individual.fitness))
            stats_f.write(
                '{},{},{},{},{},{},{}'.format(self.model.current_generation, popavg, popstd, popmin, popmax, best_individual.fitness, self.model.best_individual.fitness).replace('\n',' ') +'\n')
        
        if saveBOG == True:
            directory = os.path.join(self.directory, '{}-{}-seed-{}'.format(self.prefix, timestamp, self.setup.random_seed))
            if not os.path.isdir(directory):
                os.makedirs(directory, mode=0o777)
            np.save(os.path.join(directory, 'BOGeneration-%d'%self.model.current_generation), best_individual.genes)
            saveBOG = False

    def export_best_png(self):

        dataset = self.model.best_individual.genotype_to_phenotype(self.model.generator)

        # export each image individually
        directory = os.path.join(self.directory, '{}-{}-seed-{}-gen-{}-dataset'.format(self.prefix, timestamp, self.setup.random_seed, self.model.current_generation))
        if not os.path.isdir(directory):
            os.makedirs(directory, mode=0o777)

        i=1
        for image in dataset:
            path = os.path.join(directory, "fig-%04d.jpg"%i)
            cv2.imwrite(path, image*255, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            i+=1

        # export full dataset in single image
        r,c = int_to_shape(dataset.shape[0])
        dataset = custom_reshape(dataset, (r,c,dataset.shape[1],dataset.shape[2],dataset.shape[3]))
        dataset_image = cv2.vconcat([cv2.hconcat(image) for image in dataset])
        path = os.path.join(directory, "full-dataset.jpg")
        cv2.imwrite(path, dataset_image*255, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


class LatentIndividual(Individual):

    def __init__(self, creation_index, model, individuals_dict):
#        super(LatentIndividual, self).__init__(creation_index, model, individuals_dict)
        self.id = creation_index
        self.fitness = 0.0
        self.individuals_dict = individuals_dict
        #print(ref)
        #print(id)
        self.model = model # model to which it belongs..
        self.genes = None
        self.latent_dim = individuals_dict.get('latent_dim', 100)
        self.nr_images = individuals_dict.get('nr_images', 100)
        self.genes_length = self.latent_dim * self.nr_images
        self.distribution_center = individuals_dict.get('distribution_center', 0)
        self.distribution_deviation = individuals_dict.get('distribution_deviation', 1)

        # to define custom ranges for each gene.. define the range per gene using lists.. or it just uses the same value for all
        """
        if not isinstance(self.max_values_per_gen, list):
            self.max_values_per_gen = [self.max_values_per_gen] * self.genes_length
        if not isinstance(self.min_values_per_gen, list):
            self.min_values_per_gen = [self.min_values_per_gen] * self.genes_length
        """

    def initialize(self):
        self.genes = np.array([self.pickvalue(gene_i) for gene_i in range(self.genes_length)])
        self.fitness = 0.0

    def pickvalue(self, index):
        """
        if self.min_values_per_gen[index] == self.max_values_per_gen[index]:
            return self.min_values_per_gen[index]
        """
        return np.random.normal(self.distribution_center, self.distribution_deviation)
        
    def genotype_to_phenotype(self, generator):
        noise = np.reshape(self.genes, (self.nr_images, self.latent_dim))
        images = generator.predict(noise)
        images = 0.5 + images * 0.5
        return images

    def mutate(self, probability=0.05):
        for i in range(self.genes_length):
            if random.random() <= probability:
                self.genes[i] = self.pickvalue(i)

    def mutate_single(self, probability=None):
        # mutate only a random gene... generate a new value
        index_to_mutate = int(np.random.randint(0, self.genes_length - 1))
        self.genes[index_to_mutate] = self.pickvalue(index_to_mutate)

    def crossover(self, other, probability):
        #2-point
        points = np.sort(np.random.choice(self.genes_length, 2, replace=False))
        self.genes = np.array([other.genes[n] if points[0] < n < points[1] else self.genes[n] for n in range(self.genes_length)])

        # uniform
        #self.genes = np.array([self.genes[n] if np.random.random() < probability else other.genes[n] for n in range(self.genes_length)])

    def genotype(self):
        return self.genes

    def copy(self):
        new = copy.copy(self)
        new.genes = copy.deepcopy(self.genes)
        return new

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __str__(self):
        return str(self.genes) + " fitness: " + str(self.fitness)


def ganeval(generator, metric, **argv):
    def evaluation_ref(individual):
        individual.fitness = dataset_similarity(individual.genotype_to_phenotype(generator), metric, **argv)
        return individual
    return evaluation_ref

class LatentTournament(Tournament):

    def select(self, current_population):
        fitnesses = [individual.fitness for individual in current_population]
        if self.objective == max:
            elite_indexes = np.argpartition(fitnesses, -self.elitism)[-self.elitism:]
        else:
            elite_indexes = np.argpartition(fitnesses, self.elitism)[:self.elitism]
        

        # clone elite for elite popuplation
        elite_population = [current_population[elite_index].copy() for elite_index in elite_indexes]
        new_population = elite_population + self.tournament(current_population)

        return new_population, elite_population

class LatentEvolution(EA):

    def __init__(self, generator):
        super(LatentEvolution,self).__init__()
        self.generator = generator

        global timestamp
        global start

        start = time.time()
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')

    def check_termination_criterion(self):
        return self.current_generation >= self.setup.generations

"""
def run(seeds=[0],
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/mnist/dc_all_full/model/generator.h5",
        metric=ncc,
        metric_args={},
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection=LatentTournament,
        directory="test"
        ):

    global timestamp
    global start

    for seed in seeds:
        start = time.time()
        timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')

        generator = load_model(generator_path)

        ea_testing = LatentEvolution(generator, metric)
        ea_cb = LatentIndividualCallback(directory=directory)

        # setup GA
        a_setup = Setup()
        a_setup.generations = 500
        a_setup.population_size = 50
        a_setup.multiprocesses = 1
        a_setup.random_seed = seed
        a_setup.torn_size = 3
        a_setup.mutation_rate = mutation_rate
        a_setup.crossover_rate = crossover_rate
        a_setup.objective = min
        a_setup.individuals_dict['latent_dim'] = 100
        a_setup.individuals_dict['nr_images'] = 50
        a_setup.save_setup(directory)

        # callback needs reference to the GA model and the Setup
        ea_cb.set_model(ea_testing)
        ea_cb.set_setup(a_setup)

        ea_testing.initialize(setup=a_setup,
                              individual_ref=LatentIndividual,
                              selection_ref=selection, 
                              evaluation_ref=ganeval(generator, metric, **metric_args))

        # pass reference to callback!
        ea_testing.evolve(callbacks=[ea_cb])

        print('*********')
        print('total_time ', time.time() - start)
"""
if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    #########
    """
    run(seeds=[4], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/mnist-fashion/dc_all_full/model/generator_800.h5",
        metric=ncc,
        metric_args={"blackwhite" : True},
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection=LatentTournament,
        directory="mnist-fashion/evo-ncc"
        )
    
    run(seeds=[0,1,2,3,4], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/mnist-fashion/dc_all_full/model/generator_800.h5",
        metric=rmse,
        metric_args={"blackwhite" : True},
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection=LatentTournament,
        directory="mnist-fashion/evo-rmse"
        )
    ##########
    
    run(seeds=[], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/faces/1000/model/generator.h5",
        metric=ncc,
        metric_args={"blackwhite" : True},
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection=LatentTournament,
        directory="facity/evo-ncc"
        )
    run(seeds=[0], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/faces/1000/model/generator.h5",
        metric=rmse,
        metric_args={"blackwhite" : True},
        mutation_rate=0.02,
        crossover_rate=0.7,
        selection=LatentTournament,
        directory="facity/evo-rmse"
        )
    run(seeds=[], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/faces/1000/model/generator.h5",
        metric=ncc,
        metric_args={"blackwhite" : True},
        mutation_rate=0.0,
        crossover_rate=0.0,
        selection=FullRandomRestart,
        directory="facity/random-ncc"
        )
    run(seeds=[0,1,2], 
        generator_path="/media/Storage/jncor_last/old_experiments/dcgan/faces/1000/model/generator.h5",
        metric=rmse,
        metric_args={"blackwhite" : True},
        mutation_rate=0.0,
        crossover_rate=0.0,
        selection=FullRandomRestart,
        directory="facity/random-rmse"
        )
    

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    #a = np.zeros((7,8,28,28,1))
    generator = load_model("/media/Storage/jncor_last/old_experiments/dcgan/mnist/dc_all_full/model/generator.h5")
    noise = np.load("mnist-evo-rmse/-2019-10-28-225749-seed-15/best-of-generation-99.npy")
    noise = np.reshape(noise,(50,100))
    noise = np.random.uniform(0, 1, (50, 100))
    images = generator.predict(noise)
    images = custom_reshape(images,(7,8,28,28,1))
    images = images*0.5+0.5
    x = cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in images])
    cv2.imwrite("a.jpg", x*255, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(images.shape)
    """