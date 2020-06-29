from abc import abstractmethod
from .image_similarity import paired_similarity, centroid_similarity
from .EA.ea import Setup

class Supervisor():

    def __init__(self,
                 latent_dim,
                 nr_images):

        self.latent_dim=latent_dim
        self.nr_images=nr_images

    @abstractmethod
    def supervise(self):
        pass

class GeneticSupervisor(Supervisor):

    def __init__(self,
                 generator,
                 latent_dim,
                 nr_images,
                 generations,
                 population_size,
                 multiprocesses,
                 random_seed,
                 objective,
                 evolutionary_ref,
                 callback_ref,
                 selection_ref,
                 individual_ref,
                 evaluation_ref,
                 tournament_size,
                 mutation_rate,
                 crossover_rate,
                 path):

        super(GeneticSupervisor, self).__init__(latent_dim, nr_images)

        self.ea_motor = evolutionary_ref(generator)
        self.ea_cb = callback_ref(directory=path)

        # setup GA
        self.setup = Setup()
        self.setup.generations = generations
        self.setup.population_size = population_size
        self.setup.multiprocesses = multiprocesses
        self.setup.random_seed = random_seed
        self.setup.torn_size = tournament_size
        self.setup.mutation_rate = mutation_rate
        self.setup.crossover_rate = crossover_rate
        self.setup.objective = objective
        self.setup.individuals_dict['latent_dim'] = latent_dim
        self.setup.individuals_dict['nr_images'] = nr_images
        self.setup.save_setup(path)

        # callback needs reference to the GA model and the Setup
        self.ea_cb.set_model(self.ea_motor)
        self.ea_cb.set_setup(self.setup)
        self.ea_motor.initialize(setup=self.setup,
                                 individual_ref=individual_ref,
                                 selection_ref=selection_ref, 
                                 evaluation_ref=evaluation_ref)
    def supervise(self):
        self.ea_motor.evolve(callbacks=[self.ea_cb])
        return self.ea_motor.best_individual.genotype_to_phenotype(self.ea_motor.generator)