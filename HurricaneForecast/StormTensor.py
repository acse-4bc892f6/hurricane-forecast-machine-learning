import torchvision
import torch
from torch.utils.data import TensorDataset
# import torchvision.transforms as T
from PIL import Image
# import numpy as np

class StormTensorDataset(TensorDataset):
    def __init__(self, train_df, storm_id, num_sequence, download_dir, train_source):
        """
        Arguments:
        ----------
        train_df : pandas.DataFrame
            A dataframe containing training data with Image ID, Storm ID,
            Relative Time, Ocean, and Wind Speed

        storm_id : str
            Specific storm

        download_dir : pathlib.PosixPath
            Path to the directory that the dataset is downloaded to

        transforms : torchvision.transforms
            Transformations on an image tensor
        """
        self.df = train_df[train_df['Storm ID'] == storm_id][:-5]
        self.storm_id = storm_id
        self.image_ids = self.df['Image ID'].to_numpy()
        self.relative_times = self.df['Relative Time'].to_numpy(dtype=int)
        self.oceans = self.df['Ocean'].to_numpy(dtype=int)
        self.wind_speeds = self.df['Wind Speed'].to_numpy(dtype=int)
        self.download_dir = download_dir
        self.train_source = train_source
        self.num_sequence = num_sequence
        self.Resize_space()
        # self.df['Relative Time'] = self.df['Relative Time'].apply(int)
        # self.df['Wind Speed'] = self.df['Wind Speed'].apply(int)
        
        
    def get_image_tensor(self, image_id):
        """
        Obtain image tensor from Image ID.
        
        Arguments:
        ----------
        image_id : str
            Image ID from self.df

        Returns:
        ----------
        img : torch.tensor
            Tensor of the image specified by image_id
        """
        img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
        return img

    def get_storm_images(self, storm_id):
        """
        Create a tensor of all images in the data for storm with Storm ID = storm_id
        """
        storm_df = self.df[self.df['Storm ID'] == storm_id]
        storm_images = []

        for image_id in storm_df['Image ID']:
            img_tensor = self.get_image_tensor(image_id)
            storm_images.append(img_tensor)
        storm_images = torch.cat(storm_images, dim=0)

        return storm_images

    def calculate_mean_std(self, storm_id):
        """
        Calculate the mean and standard deeviation of all the pixels in the training dataset
        This will be used for normalisation
        """
        # Take the tensor of all images and standardise by 255
        storm_images_array = self.get_storm_images(storm_id)/255

        # Remove the last 5 images which will be used for test data
        storm_train_images = storm_images_array   #train_data_array

        # Calculate the mean and standard deviation
        mean = [torch.mean(storm_train_images.flatten()).item()]
        std = [torch.std(storm_train_images.flatten()).item()]
        
        return mean, std

    def get_tensor_combination(self, start_pos, number):
        """
        Obtain the combination tensor of several images from the given start point and number of images required.
        
        Arguments:
        ----------
        start_pos : int
            The start index to combine the images.

        number:  int
            Combined the number of images in one tensor

        Returns:
        ----------
        img : torch.tensor
            The combination tensor for numbers of images.
        """
        mean, std = self.calculate_mean_std(self.storm_id)
        mean = mean[0]
        std = std[0]
        image_id = self.image_ids[0][:4] + str(int(start_pos/100)) + str(int((start_pos/10) % 10)) + str(start_pos % 10)
        img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+ self.train_source+ '_' + image_id + '/image.jpg')
        img = img.type(torch.FloatTensor)
        img = img/255.0
        img = (img - mean)/std
        imgs = [img]
        for i in range(number):
          image_add = int(image_id[4:]) + 1
          image_id = image_id[:4] + str(int(image_add/100)) + str(int((image_add/10) % 10)) + str(image_add % 10)
          temp_img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
          temp_img = temp_img.type(torch.FloatTensor)
          temp_img /= 255.0
          temp_img =(temp_img-mean)/std
          imgs.append(temp_img)
        
        img = torch.stack(imgs, dim=0)
        return img
    
    def get_last_five(self):
        number = 4
        start_pos = 96
        mean, std = self.calculate_mean_std(self.storm_id)
        mean = mean[0]
        std = std[0]
        image_id = self.image_ids[0][:4] + str(int(start_pos/100)) + str(int((start_pos/10) % 10)) + str(start_pos % 10)
        img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+ self.train_source+ '_' + image_id + '/image.jpg')
        img = img.type(torch.FloatTensor)
        img = img/255.0
        img = (img - mean)/std
        imgs = [img]
        for i in range(number):
          image_add = int(image_id[4:]) + 1
          image_id = image_id[:4] + str(int(image_add/100)) + str(int((image_add/10) % 10)) + str(image_add % 10)
          temp_img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
          temp_img = temp_img.type(torch.FloatTensor)
          temp_img /= 255.0
          temp_img = (temp_img - mean) / std
          imgs.append(temp_img)
        img = torch.stack(imgs, dim=0)
        return img


    def Resize_space(self):
        """
        Resize all the images in the train_df by (366, 366).  
        """ 
        image_id = self.image_ids[0]
        for i in range(len(self.df)):
            img = torchvision.io.read_image(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
            if(img.shape != torch.Size([1, 366, 366])):
                image = Image.open(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
                image = image.resize((366,366), Image.ANTIALIAS)
                image.save(str(self.download_dir)+ '/'+self.train_source+'/'+self.train_source+ '_' + image_id + '/image.jpg')
                print("The image number: ", str(image_id[4:]), " has been reshaped")
            
            image_add = int(image_id[4:]) + 1
            image_id = image_id[:4] + str(int(image_add/100)) + str(int((image_add/10) % 10)) + str(image_add % 10)
        print("All the data are in shape (366,366)")  

    def tensor_to_image(self, tensor):
        """
        Convert image tensor to a 2D array

        Arguments:
        ----------
        tensor : torch.tensor
            Image tensor with one channel
        """
        return tensor[0,:,:]
          

    def __len__(self):
        return len(self.df)// (self.num_sequence + 1)

    def get_last_10_img(self):
        """
        Get the last tem 10 imgs to predict the last five

        """
        return self.get_tensor_combination(len(self.df) - 11, 9), self.get_last_five()

    def __getitem__(self, idx):
        """
        Arguments:
        ----------
        idx: int
            Index in dataframe

        Returns features and image at the corresponding index
        from the dataframe
        """
        # get features and image
        # one for
        start_position = idx * (self.num_sequence+1) + 1
        test_start_position = start_position + self.num_sequence
        return self.get_tensor_combination(test_start_position, self.num_sequence - 1), self.get_tensor_combination(test_start_position, 0)
