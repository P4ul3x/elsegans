from abc import abstractmethod
from .image_similarity import centroid_similarity

from keras.utils import to_categorical
import numpy as np

class Evaluator():

    def __init__(self):
        pass

    @abstractmethod
    def evaluate():
        pass

class CentroidEvaluator(Evaluator):

    def __init__(self,
                 generator,
                 metric,
                 original_images,
                 metric_args):

        super(CentroidEvaluator, self).__init__()
        self.generator = generator
        self.metric = metric
        self.original_images = original_images
        self.metric_args = metric_args

    def evaluate(self, individual):
        if self.original_images is not None:
            images = np.concatenate((self.original_images, individual.genotype_to_phenotype(self.generator)), axis=0)
        else:
            images = individual.genotype_to_phenotype(self.generator)
        individual.fitness = centroid_similarity(images, self.metric, self.metric_args)
        return individual

class ClassificationEvaluator(Evaluator):

    def __init__(self,
                 generator,
                 classifier,
                 number_of_classes,
                 class_number):

        super(ClassificationEvaluator, self).__init__()
        self.generator = generator
        self.classifier = classifier
        self.number_of_classes = number_of_classes
        self.class_number = class_number

    def class_fitness(self, images):
        t = np.ones((images.shape[0], 1))*self.class_number
        t = to_categorical(t, self.number_of_classes)
        return self.classifier.test_on_batch(images, t)


    def evaluate(self, individual):
        images = individual.genotype_to_phenotype(self.generator)
        individual.fitness = self.class_fitness(images)[0]
        return individual