import shutil

import numpy as np

import configparser as cfgp

import csv
import cv2
import math

# delete a specified folder
def delete_folder(path):
    shutil.rmtree(path, ignore_errors=True)

# in place, shuffle a set of arrays in unison
def shuffle_in_unison(arrays):
    if (not isinstance(arrays, list) and not isinstance(arrays, tuple)):
        arrays = [arrays]
        
    rng_state = np.random.get_state()
    np.random.shuffle(arrays[0])
    for i in range(len(arrays[1:])):
        np.random.set_state(rng_state)
        np.random.shuffle(arrays[i+1])

def configuration_parse_value(parser, var_type, option, default):
    if parser.has_option('Configuration', option): 
        try:
            if (var_type == bool):
                return parser.getboolean('Configuration', option)
            elif (var_type == int):
                return parser.getint('Configuration', option)
            elif (var_type == float):
                return parser.getfloat('Configuration', option)
            else:
                print("<Parse Error> Invalid variable type. Use <int, float, bool>")
        except ValueError:
            print("<Parse Error> %S must be %s" % (option, var_type))
            return   
    else: 
        return default

def configuration_parse_string(parser, option, default):
    val = default
    if parser.has_option('Configuration', option):
        val = parser.get('Configuration', option)
        if val in ('None',''):
            val = default
    return val

def join_resumes(model_directories, output_file):
    with open(output_file,"w") as output:
        
        for m_dir in model_directories:

            parser = cfgp.ConfigParser()
            with open(m_dir+"/config.cfg","r") as config:
                try:
                    parser.read_file(config)
                except cfg.ParsingError:
                    print("<Error> File could not be parsed")
                    return None
                assert parser.has_section('Configuration'), "<Error> Configuration section not found"
                
                model_name = configuration_parse_string(parser, 'model_name', None)

                nr_original_images = configuration_parse_string(parser, 'nr_original_images', None)
                if nr_original_images is not None:
                    try:
                        nr_original_images = int(nr_original_images)
                    except TypeError:
                        print("<Error> Number of images should be integer")
                        return
                
                nr_generated_images = configuration_parse_string(parser, 'nr_generated_images', None)
                if nr_generated_images is not None:
                    try:
                        nr_generated_images = int(nr_generated_images)
                    except TypeError:
                        print("<Error> Number of images should be integer")
                        return
                
                nr_folds = configuration_parse_string(parser, 'nr_folds', None)
                if nr_folds is not None:
                    try:
                        nr_folds = int(nr_folds)
                    except TypeError:
                        print("<Error> Number of images should be integer")
                        return

            with open(m_dir+"/stats_fold_resume.csv", "r") as stats:
                reader = csv.reader(stats, delimiter=',')
                next(reader)

                writer = csv.writer(output, delimiter=',')
                writer.writerow([model_name, None, nr_folds, nr_original_images, nr_generated_images]+next(reader))

#https://stackoverflow.com/questions/58136674/numpy-reshape-automatic-filling-or-removal
# reshape an array filling missing values with 0's or cutting overflow from the end
def custom_reshape(array, shape):
    result = np.zeros(shape).ravel()
    result[:min(result.size, array.size)] = array.ravel()[:min(result.size, array.size)]
    return result.reshape(shape)

# generate a shape from a value with a size as big or bigger than it and as close to a square as possible
def int_to_shape(val):
    rows = round(math.sqrt(val))
    columns = rows + (val/rows>rows)
    return rows, columns

def bgr2rgb(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(img_path, img)

def npy_to_jpg(npy_path, jpg_path, generator, latent_dim, scale_color=255):
    
    genes = np.load(npy_path)
    noise = np.reshape(genes, (int(len(genes)/latent_dim), latent_dim))
    
    images = generator.predict(noise)
    images = (0.5 + images * 0.5)*scale_color
    images = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images])

    r,c = int_to_shape(images.shape[0])
    images = custom_reshape(images, (r,c,images.shape[1],images.shape[2],images.shape[3]))
    comb_image = cv2.vconcat([cv2.hconcat(image) for image in images])

    cv2.imwrite(jpg_path, comb_image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == '__main__':
    join_resumes(["/media/Storage/jncor_last/old_experiments/dcgan/mnist/%d+0"%i for i in range(7100,15001,100)],"join_resumes.csv")
