import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
from radiant_mlhub import Dataset
import tarfile
from glob import glob
from pathlib import Path
import json
import pandas as pd
# from .FetchData import FetchStormDataset


class DataChecks():
  def __init__(self, download_dir, source, labels=None, provide_df = False, df=None, storm_id="*"):
    """
    Arguments
    ----------
    download_dir : pathlib.PosixPath
      Path to the directory that the dataset is downloaded to

    source : str
      directory name with image.jpeg file and features.json and stac.json

    labels : str
      directory name with wind speed and stac.json

    df : pandas.DataFrame
      A dataframe containing columns Image ID, Storm ID,
      Relative Time, Ocean, and Wind Speed

    storm_id : str
      Specific storm to be extracted from tarfile
    """

    self.download_dir = download_dir
    self.source = source
    self.labels = labels
    self.anomalous_images = []
    self.anomalous_shapes = []
    self.image_size_warning = False
    self.indices = []
    self.times = []
    self.large_gap_warning = False
    self.df = None
    # dataframe is passed in constructor
    if provide_df:
      self.df = df
      
    # load dataframe from file
    else:

      if self.source == 'nasa_tropical_storm_competition_train_source':
        dataset = Dataset.fetch('nasa_tropical_storm_competition')
        archive_paths = dataset.download(output_dir=self.download_dir)
        train_archive_paths=[archive_paths[0], archive_paths[2]]
        if storm_id == "*":
          for archive_path in train_archive_paths:
            print(f'Extracting {archive_path}...')
            with tarfile.open(archive_path) as tfile:
                tfile.extractall(path=self.download_dir)
          print('Done')
      
        else:
          for archive_path in train_archive_paths:
            print(f'Extracting {archive_path}...')
            tar = tarfile.open(archive_path)
            for name in tar.getnames():
              if storm_id in name:
                tar.extract(name, path=self.download_dir)
          print('Done')

        train_data = []

        # train_source = 'nasa_tropical_storm_competition_train_source'
        # train_labels = 'nasa_tropical_storm_competition_train_labels'

        jpg_names = glob(str(self.download_dir / source / '**' / '*.jpg'))

        print("Creating dataframe...")

        for jpg_path in jpg_names:
            jpg_path = Path(jpg_path)
            
            # Get the IDs and file paths
            features_path = jpg_path.parent / 'features.json'
            image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
            storm_id = image_id.split('_')[0]
            labels_path = str(jpg_path.parent / 'labels.json').replace(source, labels)


            # Load the features data
            with open(features_path) as src:
                features_data = json.load(src)
                
            # Load the labels data
            with open(labels_path) as src:
                labels_data = json.load(src)

            train_data.append([
                image_id, 
                storm_id, 
                int(features_data['relative_time']), 
                int(features_data['ocean']), 
                int(labels_data['wind_speed'])
            ])

        train_df = pd.DataFrame(
            np.array(train_data),
            columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
        ).sort_values(by=['Image ID']).reset_index(drop=True)

        train_df['Wind Speed'] = train_df['Wind Speed'].apply(int)
        train_df['Relative Time'] = train_df['Relative Time'].apply(int)

        self.df = train_df

        print("Done")
        
      elif self.source == 'nasa_tropical_storm_competition_test_source':
        dataset = Dataset.fetch('nasa_tropical_storm_competition')
        archive_paths = dataset.download(output_dir=self.download_dir)
        test_archive_paths = [archive_paths[1], archive_paths[3]]
        if storm_id == "*":
          for archive_path in test_archive_paths:
            print(f'Extracting {archive_path}...')
            with tarfile.open(archive_path) as tfile:
                tfile.extractall(path=self.download_dir)
          print('Done')
        
        else:
          print("In else statement")
          for archive_path in test_archive_paths:
              print(f'Extracting {archive_path}...')
              tar = tarfile.open(archive_path)
              for name in tar.getnames():
                if storm_id in name:
                  tar.extract(name, path=self.download_dir)
          print('Done')

        test_data = []

        jpg_names = glob(str(self.download_dir / source / '**' / '*.jpg'))

        print('Creating dataframe...')

        for jpg_path in jpg_names:
            jpg_path = Path(jpg_path)

            # Get the IDs and file paths
            features_path = jpg_path.parent / 'features.json'
            image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
            storm_id = image_id.split('_')[0]

            # Load the features data
            with open(features_path) as src:
                features_data = json.load(src)

            test_data.append([
                image_id, 
                storm_id, 
                int(features_data['relative_time']), 
                int(features_data['ocean']), 
            ])

        test_df = pd.DataFrame(
            np.array(test_data),
            columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean']
        ).sort_values(by=['Image ID']).reset_index(drop=True)

        self.df = test_df

        print('Done')

      else:
        data = []
        jpg_names = glob(str(self.download_dir / source / '**' / '*.jpg'))
        for jpg_path in jpg_names:
            jpg_path = Path(jpg_path)
            
            # Get the IDs and file paths
            features_path = jpg_path.parent / 'features.json'
            image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
            storm_id = image_id.split('_')[0]
            labels_path = str(jpg_path.parent / 'labels.json').replace(source, labels)


            # Load the features data
            with open(features_path) as src:
                features_data = json.load(src)
                
            # Load the labels data
            with open(labels_path) as src:
                labels_data = json.load(src)

            data.append([
                image_id, 
                storm_id, 
                int(features_data['relative_time']), 
                int(features_data['ocean']), 
                int(labels_data['wind_speed'])
            ])

        temp_df = pd.DataFrame(
            np.array(data),
            columns=['Image ID', 'Storm ID', 'Relative Time', 'Ocean', 'Wind Speed']
        ).sort_values(by=['Image ID']).reset_index(drop=True)

        temp_df['Wind Speed'] = temp_df['Wind Speed'].apply(int)
        temp_df['Relative Time'] = temp_df['Relative Time'].apply(int)

        # print(temp_df.head())
        self.df = temp_df
    

  def check_shapes(self, verbose=False):
      """
      Check size of images in df. Raise warning if shape is unequal.
      """ 
      for i in range(len(self.df)):
          image_id = self.df['Image ID'][i]
          img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.source+'/'+self.source+ '_' + image_id + '/image.jpg')
          if(img.shape != torch.Size([1, 366, 366])):
              self.anomalous_images.append(image_id)
              self.anomalous_shapes.append(img.shape)
              if verbose:
                print("The image", image_id, " has different shape: ", img.shape)
      
      if len(self.anomalous_images) > 0:
          self.image_size_warning = True
          print("Some image sizes are unequal")
      else:
          print("All images are of size 366x366 with 1 channel")

  def get_anomalous_shapes(self):
      """
      Returns lists storing image id of anomalous images
      and shapes of those images
      """
      return self.anomalous_images, self.anomalous_shapes

  def check_time_gaps(self, verbose=False):
    """
    Check time gap between images. Raise warning if
    the gap is larger than 1 hour.
    """
    relative_times = self.df['Relative Time']
    diff_times = relative_times.diff()
    diff_times = diff_times.fillna(0)
    diff_times = diff_times.apply(int)

    for i in range(1, len(diff_times)):
      if diff_times[i] > 3600:
        self.indices.append(i)
        self.times.append(diff_times[i])
        if verbose:
          print('Gap between indices ', i-1, 'and', i, 'is', diff_times[i])

    if len(self.indices) > 0 and len(self.times) > 0:
      self.large_gap_warning = True
      print('Largest time gap', max(self.times), 'at index', self.indices[self.times.index(max(self.times))])

  def get_time_gaps(self):
      """
      Return lists storing index of large time gap
      and the magnitude of the time gap
      """
      return self.indices, self.times

  def average_wind_speeds(self):
    """
    Returns average wind speed over dataframe
    """
    wind_speeds = self.df['Wind Speed'].to_numpy()
    return np.mean(wind_speeds)

  def count_images(self):
    """
    Returns number of images in dataframe
    """
    return len(self.df['Image ID'])

  def plot_wind_speeds(self):
    """
    Create plot of wind speed over relative time
    """
    plt.figure(figsize=(12, 4))
    plt.scatter(self.df['Relative Time'], self.df['Wind Speed'], color="lightgray")
    plt.xlabel("Relative Time")
    plt.ylabel("Wind Speed")
    plt.title(f"Wind Speed over Relative Time: Storm {self.df['Storm ID'][0]}")

  def get_warnings(self):
    """
    Print out warning if image sizes are unequal or large time gap is found.
    """
    if self.image_size_warning:
      print("Unequal image sizes")
    if self.large_gap_warning:
      print("Large time gaps (> 1 hour) exist")


  def get_image(self, image_id):
    """
    A function that returns the image represented by image_id as a tensor.
    """
    
    img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.source+'/'+self.source+ '_' + image_id + '/image.jpg')
    return img
