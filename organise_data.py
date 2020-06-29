import pandas as pd
import os
from shutil import copyfile

def organise_data(train_structure, train_folder, artists_folders, styles=[], artists=[]):
    if (not isinstance(styles, list) and not isinstance(styles, tuple)):
        styles = [styles]
    if (not isinstance(artists, list) and not isinstance(artists, tuple)):
        artists = [artists]

    data = pd.read_csv(train_structure)
    artist_folder = {}

    i=1
    for index, row in data.iterrows():
        if not styles or row.style in styles:
            if (not artists or row.artist in artists):
                if not row.artist in artist_folder:
                    artist_folder[row.artist] = f"{i:03d}_"+row.artist
                    i+=1
                    os.makedirs(os.path.join(artists_folders, artist_folder[row.artist]), mode=0o777)
                copyfile(os.path.join(train_folder, row.filename), os.path.join(artists_folders, artist_folder[row.artist], row.filename))

if __name__ == '__main__':
    organise_data('/home/cdv/Documents/fga/painters/train_info.csv', '/home/cdv/Documents/fga/painters/train',
                  'datasets/wikiart-post-impressionism', 'Post-Impressionism')