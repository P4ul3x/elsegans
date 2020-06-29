import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import cv2
import glob
import os

from confusion_matrix_pretty_print import pretty_plot_confusion_matrix

### CLASSIFIER TRAINING STATISTICS ###
def classifier_learning_curve(stats, save_destination=None):
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    
    axes[0].grid()
    axes[0].plot(range(len(stats['loss_mean'])), stats['loss_mean'], '-b', label='Training loss')
    #axes[0].fill_between(len(stats['loss_mean']), 
    #                     stats['loss_mean'] - stats['loss_std'],
    #                     stats['loss_mean'] + stats['loss_std'],
    #                     alpha=0.1)
    axes[0].plot(range(len(stats['val_loss_mean'])), stats['val_loss_mean'], '-r', label='Validation loss')
    #axes[0].fill_between(len(stats['val_loss_mean']), 
    #                     stats['val_loss_mean'] - stats['val_loss_std'],
    #                     stats['val_loss_mean'] + stats['val_loss_std'],
    #                     alpha=0.1)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss training curve")
    axes[0].legend(loc='best')

    axes[1].grid()
    axes[1].plot(range(len(stats['accuracy_mean'])), stats['accuracy_mean'], '-b', label='Training accuracy')
    #axes[1].fill_between(len(stats['accuracy_mean']), 
    #                     stats['accuracy_mean'] - stats['accuracy_std'],
    #                     stats['accuracy_mean'] + stats['accuracy_std'],
    #                     alpha=0.1)
    axes[1].plot(range(len(stats['val_accuracy_mean'])), stats['val_accuracy_mean'], '-r', label='Validation accuracy')
    #axes[1].fill_between(len(stats['val_accuracy_mean']), 
    #                     stats['val_accuracy_mean'] - stats['val_accuracy_std'],
    #                     stats['val_accuracy_mean'] + stats['val_accuracy_std'],
    #                     alpha=0.1)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy training curve")
    axes[1].legend(loc='best')

    if save_destination is not None:
        plt.savefig(save_destination)

    plt.close('all')


### DCGAN TRAINING STATISTICS ###
def dcgan_learning_curve(stats, save_destination=None):
    # Get epoch statistics
    stats_epoch_last = stats[stats['batch'] == max(stats['batch'])]
    
    stats_epoch_mean = stats.groupby('epoch')[[col for col in stats.columns]].sum() / (max(stats['batch'])+1)
    
    _, axes = plt.subplots(2, 1, figsize=(10, 10))

    # -----------------
    # - Plot Accuracy -
    # -----------------
    
    # iteration
    """
    stats.plot(kind='line',
              y=['d_acc_real', 'd_acc_fake', 'd_acc'], 
              color=['green', 'blue', 'red'],
              ax=axes[0,0],
              alpha=0.25, subplots=True, sharex = True) 
    axes[0,0].set_xlabel("Batch")
    axes[0,0].set_ylabel("Accuracy")
    axes[0,0].set_title("Accuracy training curve per batch")
    axes[0,0].legend(loc='best')
    """
    #epoch mean
    stats_epoch_mean.plot(kind='line',
                    x='epoch',
                    y=['d_acc_real', 'd_acc_fake', 'd_acc'],
                    color=['green', 'blue', 'red'],
                    ax=axes[0],
                    alpha=0.60)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_title("Accuracy training curve per epoch")
    axes[0].legend(loc='best')
   
    # -----------------
    # --- Plot Loss ---
    # -----------------
    
    # iteration
    """
    stats.plot(kind='line',
              y=['d_loss_real', 'd_loss_fake', 'd_loss', 'g_loss'],
              color=['green', 'blue', 'red', 'cyan'],
              ax=axes[1,0],
              alpha=0.25, subplots=True, sharex = True)
    axes[1,0].set_xlabel("Batch")
    axes[1,0].set_ylabel("Loss")
    axes[1,0].set_title("Loss training curve per batch")
    axes[1,0].legend(loc='best')
    """
    # epoch mean
    stats_epoch_mean.plot(kind='line',
                    x='epoch',
                    y=['d_loss_real', 'd_loss_fake', 'd_loss', 'g_loss'],
                    color=['green', 'blue', 'red', 'cyan'],
                    ax=axes[1],
                    alpha=0.60)
    axes[1].set_xlabel("Batch")
    axes[1].set_ylabel("Mean Loss")
    axes[1].set_title("Loss training curve per epoch")
    axes[1].legend(loc='best')

    plt.show()

    if save_destination is not None:
        plt.savefig(save_destination)

    plt.close('all')

def supervisor_evolution_curve(stats, save_destination=None):

    _, axes = plt.subplots(1, 1, figsize=(10, 5))
    
    axes.grid()
    axes.plot(range(len(stats['avg'])), stats['avg'], '-g', label='Average of Generation')
    axes.plot(range(len(stats['min'])), stats['min'], '-b', label='Minimum Fitness')
    axes.plot(range(len(stats['max'])), stats['max'], '-r', label='Maximum Fitness')
    axes.plot(range(len(stats['best_individual_overall'])), stats['best_individual_overall'], '-y', label='Best Overall')
    
    axes.set_xlabel("Generations")
    axes.set_ylabel("Fitness")
    axes.set_title("Supervisor fitness")
    axes.legend(loc='best')

    if save_destination is not None:
        plt.savefig(save_destination)

    plt.close('all')

def boxplots(stats, filter_like=None, save_dir=None):
    
    if filter_like is not None:
        stats = stats.filter(like=filter_like)

    for col in stats.columns:
        _, axes = plt.subplots(1, 1, figsize=(5, 5))
        axes.set_ylim(0,1)
        stats.boxplot(column=col, ax=axes, grid=True)
        axes.set_title(col)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,col+".jpg"))
        plt.close('all')

def joint_boxplots(stats_list, labels, filter_like=None, save_dir=None):
    
    assert len(stats_list) == len(labels)

    if filter_like is not None:
        stats_list = [s.filter(like=filter_like) for s in stats_list]

    for col in stats_list[0].columns:
        _, axes = plt.subplots(1, 1, figsize=(10, 7))
        axes.set_ylim(0.5,0.9)
        stat = np.transpose([s[col].to_numpy() for s in stats_list])
        stat = pd.DataFrame(stat, columns=labels)
        stat.boxplot(ax=axes, grid=True)
        axes.set_title(col)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir,col+".jpg"))
        plt.close('all')

def confusion_from_numpy(confusion_matrix, save_destination=None):
    df_cm = pd.DataFrame(confusion_matrix, index=range(1, confusion_matrix.shape[0]+1), columns=range(1,confusion_matrix.shape[1]+1))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)

    if save_destination is not None:
        plt.savefig(save_destination)
    plt.close('all')

def test_image(img_path, save_path, shape):
    if shape[2] == 1:
        img = cv2.imread(img_path, 0)
    else: #IMG_SHAPE[2] == 3
        img = cv2.imread(img_path, 1)
    img = cv2.resize(img, shape[:2], interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path,img)

if __name__ == '__main__':
    """
    s = [pd.read_csv(x) for x in glob.glob("/home/pcastillo/supervisor_gan/experiments/hushem/supervisor_classeval_add20/seed*/classifier/test_stats_summary.csv")]
    c = pd.concat(s)
    by_row_index = c.groupby(c.index)
    df_means = by_row_index.mean()
    df_stds = by_row_index.std()
    df_means.to_csv("/home/pcastillo/supervisor_gan/experiments/hushem/supervisor_classeval_add20/test_means.csv", index = False, float_format='%.6f')
    df_stds.to_csv("/home/pcastillo/supervisor_gan/experiments/hushem/supervisor_classeval_add20/test_stds.csv", index = False, float_format='%.6f')
    #classifier_learning_curve(df_means, '/home/pcastillo/supervisor_gan/experiments/hushem/supervisor_classeval_add20/train_means_curve.jpg')
    
    #print(glob.glob("/home/pcastillo/supervisor_gan/experiments/hushem/supervisor_classeval_add20/seed*/sup_*/*-seed-*.csv"))
    
    for x in glob.glob("/home/pcastillo/supervisor_gan/experiments/hushem/dcgans/20000epochs_rgb/class_*/stats.csv"):
        #try:
            stats=pd.read_csv(x)
            dcgan_learning_curve(stats, os.path.dirname(x)+'/learning_curve.jpg')
        #except:
            print(x)
    
    test_image("/media/Storage/jncor_last/old_experiments/supervised_gan/datasets/hushem/01_Normal/image_054.BMP",
               "tests/hushem_test_image.jpg",
               (132,132,1))
    
    stats=[]
    for x in glob.glob("/home/pcastillo/supervisor_gan/experiments/hushem/classifiers/classeval_e10k_w5_epp15_ipp55/seed*/classifier/test_stats_summary.csv"):
        try:
            stats.append(pd.read_csv(x))
        except:
            print(x)
    stats = pd.concat(stats)
    boxplots(stats, filter_like='mean', save_dir='/home/pcastillo/supervisor_gan/experiments/hushem/classifiers/classeval_e10k_w5_epp15_ipp55')
    
	"""
    labels = ["original",
              "simple \n+160 \nsim",
              "simple \n+160 \ncle",
              "part \nW15 \nEPP10 \nIPP40",
              "part \nW15 \nEPP10 \nIPP55",
              "part \nW15 \nEPP15 \nIPP40",
              "part \nW15 \nEPP15 \nIPP55",
              "part \nW15 \nEPP15 \nIPP80",
              "part \nW15 \nEPP20 \nIPP55",
              "part \nW0 \nEPP15 \nIPP55",
              "part \nW25 \nEPP15 \nIPP55",
              "part \nW40 \nEPP15 \nIPP55",
              "part \nW50 \nEPP15 \nIPP55",
              "part \nW75 \nEPP15 \nIPP55"
              ]

    globs = ["experiments/hushem/classifiers/original_crossval_rgb/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/similarity_add40_e10000_rgb/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_add40_e10000_rgb/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp10_ipp40/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp10_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp15_ipp40/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp15_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp15_ipp80/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w15_epp20_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w0_epp15_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w25_epp15_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w40_epp15_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w50_epp15_ipp55/seed*/classifier/test_stats_summary.csv",
             "experiments/hushem/classifiers/classeval_e10k_w75_epp15_ipp55/seed*/classifier/test_stats_summary.csv"
            ]

    all_stats = []

    for g in globs:
        stats=[]
        for x in glob.glob(g):
            try:
                stats.append(pd.read_csv(x))
            except:
                print(x)
        stats = pd.concat(stats)
        all_stats.append(stats)

    joint_boxplots(all_stats, labels, filter_like='mean', save_dir='experiments/hushem/classifiers')
    