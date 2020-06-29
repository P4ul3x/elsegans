from _scripts.classifier import Classifier as CL

from _scripts import dcgan
from _scripts.dcgan import DCGAN

from _scripts import wgan
from _scripts.wgan import WGAN

from _scripts.evaluator import CentroidEvaluator, ClassificationEvaluator
from _scripts.supervisor import GeneticSupervisor
from _scripts.latent_space_evo import LatentEvolution, LatentIndividual, LatentIndividualCallback, LatentTournament

#from image_similarity import rmse, ssim, euclidean_distance
from _scripts.image_similarity import norm_cross_correlation as ncc 
#from image_similarity import zero_mean_norm_cross_correlation as zmncc 

import plots

from _scripts import dataset_load as dl
from _scripts import utils

from keras.callbacks.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.utils.multiclass import unique_labels

import numpy as np
import pandas as pd
import os
import glob

from sklearn.utils.multiclass import unique_labels


def run_classifier(model_name,
                   path,
                   nr_classes,
                   learn_rate,
                   img_rows,
                   img_cols,
                   channels,
                   model,
                   x_train,
                   y_train,
                   x_test,
                   y_test,
                   epochs,
                   batch_size,
                   nr_folds,
                   shuffle,
                   callbacks):

    M = CL(model_name, path, nr_classes, learn_rate, img_rows, img_cols, channels, model)

    M.train(x_train, y_train, x_test, y_test, epochs, batch_size, nr_folds, shuffle, callbacks)

    M.save_configuration(epochs, batch_size, nr_folds, shuffle)

    plots.classifier_learning_curve(pd.read_csv(os.path.join(path,model_name,'train_stats.csv')),
                                    os.path.join(path,model_name,'classifier_learning_curve.jpg'))

    plots.confusion_from_numpy(np.load(os.path.join(path,model_name,'confusion_matrix.npy'), allow_pickle=False),
                               os.path.join(path,model_name,'confusion_matrix.jpg'))

    #M.save_model()
    return M.model.model

def run_part_classifier(model_name,
                         path,
                         nr_classes,
                         learn_rate,
                         img_rows,
                         img_cols,
                         channels,
                         model,
                         x_train,
                         y_train,
                         x_test,
                         y_test,
                         x_sup,
                         y_sup,
                         warmup,
                         epochs_per_part,
                         parts,
                         images_per_part,
                         batch_size,
                         nr_folds,
                         shuffle,
                         callbacks):

    M = CL(model_name, path, nr_classes, learn_rate, img_rows, img_cols, channels, model)

    M.train_part(x_train, y_train, x_test, y_test, x_sup, y_sup, warmup, epochs_per_part, parts, images_per_part, batch_size, nr_folds, shuffle, callbacks)

    #M.save_configuration(epochs, batch_size, nr_folds, shuffle)
    #M.save_model()

    plots.classifier_learning_curve(pd.read_csv(os.path.join(path,model_name,'train_stats.csv')),
                                    os.path.join(path,model_name,'train_learning_curve.jpg'))

    plots.confusion_from_numpy(np.load(os.path.join(path,model_name,'confusion_matrix.npy'), allow_pickle=False),
    						   os.path.join(path,model_name,'confusion_matrix.jpg'))

    return M.model.model

def run_dcgan(model_name,
              path,
              build_discriminator,
              build_generator,
              learn_rate,
              img_rows,
              img_cols,
              channels,
              latent_dim,
              x_train,
              epochs,
              batch_size,
              save_interval_batch,
              save_interval_epoch):

    M = DCGAN(model_name, path, build_discriminator, build_generator, learn_rate, img_rows, img_cols, channels, latent_dim)

    M.train(x_train, epochs, batch_size, save_interval_batch,save_interval_epoch)
    M.save_configuration(epochs, batch_size, save_interval_batch,save_interval_epoch)
    M.save_dcgan()
    return M

def run_wgan(model_name,
              path,
              build_discriminator,
              build_generator,
              gradient_penalty_weight,
              training_ratio,
              learn_rate,
              img_rows,
              img_cols,
              channels,
              latent_dim,
              x_train,
              epochs,
              batch_size,
              save_interval_batch,
              save_interval_epoch):

    M = WGAN(model_name, path, build_discriminator, build_generator, gradient_penalty_weight, learn_rate, img_rows, img_cols, channels, latent_dim)

    M.train(x_train, epochs, batch_size, training_ratio, save_interval_batch,save_interval_epoch)
    M.save_configuration(epochs, batch_size, training_ratio, save_interval_batch,save_interval_epoch)
    M.save_gan()
    return M


def run_supervisor(evaluator,
                   supervisor_ref,
                   generator,
                   latent_dim,
                   nr_images,
                   generations,
                   population_size,
                   multiprocesses,
                   random_seed,
                   objective,
                   evolution_ref,
                   callback_ref,
                   selection_ref,
                   individual_ref,
                   tournament_size,
                   mutation_rate,
                   crossover_rate,
                   path):

    M = supervisor_ref(generator,
                   latent_dim,
                   nr_images,
                   generations,
                   population_size,
                   multiprocesses,
                   random_seed,
                   objective,
                   evolution_ref,
                   callback_ref,
                   selection_ref,
                   individual_ref,
                   evaluator.evaluate,
                   tournament_size,
                   mutation_rate,
                   crossover_rate,
                   path)

    return M.supervise()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    
    np.random.seed(seed=0)

    paths = glob.glob("datasets/hushem/*")
    paths.sort()
    print(paths)
    #x, y, _, _ = dl.load_mnist()
    x, y = dl.local_dataset([p+"/*" for p in paths], (132,132,3))
    labels = unique_labels(y)
    print(labels)

    x = x / 127.5 - 1.

    _x = np.split(x[np.where(y==0)], [40], axis=0)
    _y = np.split(y[np.where(y==0)], [40], axis=0)
    x_train=_x[0]
    x_test =_x[1]
    y_train=_y[0]
    y_test =_y[1]
    for l in labels[1:]:
      _x = np.split(x[np.where(y==l)], [40], axis=0)
      _y = np.split(y[np.where(y==l)], [40], axis=0)
      x_train=np.concatenate((x_train, _x[0]), axis=0)
      x_test =np.concatenate((x_test , _x[1]), axis=0)
      y_train=np.concatenate((y_train, _y[0]), axis=0)
      y_test =np.concatenate((y_test , _y[1]), axis=0)
    
    #x_train = x_train / 127.5 - 1.
    #x_test = x_test / 127.5 - 1.

    """
    generators =  [run_wgan(model_name=f"class_{l}",
                      path="experiments/hushem/wgans/10000epochs_rgb", # 
                      build_discriminator=wgan.build_discriminator,
                      build_generator=wgan.build_hushem_generator,
                      gradient_penalty_weight=10,
                      training_ratio=1,
                      learn_rate=0.0001,
                      img_rows=132,
                      img_cols=132,
                      channels=3,
                      latent_dim=100,
                      x_train=x[np.where(y==l)],
                      epochs=10000,
                      batch_size=32,
                      save_interval_batch=50,
                      save_interval_epoch=10001).generator for l in labels[:]]
    
    
    generators =  [run_dcgan(model_name=f"class_{l}",
                      path="experiments/hushem/dcgans/20000epochs_rgb", # 
                      build_discriminator=dcgan.build_discriminator,
                      build_generator=dcgan.build_hushem_generator,
                      learn_rate=0.0002,
                      img_rows=132,
                      img_cols=132,
                      channels=3,
                      latent_dim=100,
                      x_train=x_train[np.where(y==l)],
                      epochs=20000,
                      batch_size=32,
                      save_interval_batch=50,
                      save_interval_epoch=10000).generator for l in labels[:2]]
    """

    generators = [load_model(f"/home/pcastillo/supervisor_gan/experiments/hushem/dcgans/20000epochs_rgb/class_{l}/model/generator.h5") for l in labels]
    individuals = [np.load(f'/home/pcastillo/supervisor_gan/experiments/hushem/supervisors/supervisor_similarity_add40_e10000_rgb/seed0/sup_{l}/best-overall.npy')*2-1 for l in labels]
    
    individuals = [generators[l].predict(np.reshape(individuals[l], (len(individuals[l])//100, 100))) for l in labels]

    x_sup = np.concatenate(individuals, axis=0)
    y_sup = np.concatenate([np.full(len(individuals[l]), l, dtype=int) for l in labels])

    w = [0,15,15]
    epp = [15,15,10]
    p = [17,16,24]
    ipp = [55,55,55]

    for i in range(1):

     for s in range(30):
     
      np.random.seed(seed=s)
      """
      eval_classifier = run_classifier(model_name='classifier',
                           path=f"experiments/hushem/original_singles_rgb/seed{s}",
                           nr_classes=len(labels),
                           learn_rate=0.0002,
                           img_rows=132,
                           img_cols=132,
                           channels=3,
                           model=None,
                           x_train=x_train,
                           y_train=y_train,
                           x_test=x_test,
                           y_test=y_test,
                           epochs=250,
                           batch_size=32,
                           nr_folds=1,
                           shuffle=True,
                           callbacks=[])
      """
      #eval_classifier = load_model(f"/home/pcastillo/supervisor_gan/experiments/hushem/original_single_classifiers/seed{s}/classifier/model/discriminator.h5")

      """
                                  ClassificationEvaluator(
                                      generator=generators[l],
                                      classifier=eval_classifier,
                                      number_of_classes=len(labels),
                                      class_number=l)
                                  CentroidEvaluator(
                                      generator=generators[l],
                                      metric=ncc,
                                      original_images=x_train[np.where(y_train==l)]*0.5+0.5,
                                      metric_args={"blackwhite":False})
      

      individuals = [run_supervisor(
                        evaluator=CentroidEvaluator(
                                      generator=generators[l],
                                      metric=ncc,
                                      original_images=x_train[np.where(y_train==l)]*0.5+0.5,
                                      metric_args={"blackwhite":False}),
                        supervisor_ref=GeneticSupervisor,
                        generator=generators[l],
                        latent_dim=100,
                        nr_images=40,
                        generations=10000,
                        population_size=100,
                        multiprocesses=1,
                        random_seed=s,
                        objective=min,
                        evolution_ref=LatentEvolution,
                        callback_ref=LatentIndividualCallback,
                        selection_ref=LatentTournament,
                        individual_ref=LatentIndividual,
                        tournament_size=3,
                        mutation_rate=0.05,
                        crossover_rate=0.7,
                        path=f"experiments/hushem/supervisor_similarity_add40_e10000_rgb/seed{s}/sup_{l}")*2-1 for l in labels]
      """
      
      run_classifier(model_name='classifier',
                     path=f"experiments/hushem/classifiers/similarity_add160_e250/seed{s}",
                     nr_classes=len(labels),
                     learn_rate=0.0002,
                     img_rows=132,
                     img_cols=132,
                     channels=3,
                     model=None,
                     x_train=np.concatenate((x_train, x_sup), axis=0), 
                     y_train=np.concatenate((y_train, y_sup), axis=0),
                     x_test=x_test,
                     y_test=y_test,
                     epochs=250,
                     batch_size=32,
                     nr_folds=5,
                     shuffle=True,
                     callbacks=[])
      """
      run_part_classifier(model_name='classifier',
                     path=f"experiments/hushem/classifiers/classeval_e10k_w{w[i]}_epp{epp[i]}_ipp{ipp[i]}/seed{s}",
                     nr_classes=len(labels),
                     learn_rate=0.0002,
                     img_rows=132,
                     img_cols=132,
                     channels=3,
                     model=None,
                     x_train=x_train, 
                     y_train=y_train,
                     x_test=x_test,
                     y_test=y_test,
                     x_sup=x_sup,
                     y_sup=y_sup,
                     warmup=w[i], 
                     epochs_per_part=epp[i], 
                     parts=p[i], 
                     images_per_part=ipp[i],
                     batch_size=32,
                     nr_folds=5,
                     shuffle=True,
                     callbacks=[])
      """