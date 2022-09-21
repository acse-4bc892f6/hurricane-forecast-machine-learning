# from matplotlib.transforms import Transform
import torch
# from .Model import pureLSTM
# from .Model import EncoderDecoderConvLSTM
from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import torchvision.transforms as T
# import PIL
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
class Train_Validate():
    
    def __init__(self,  device) :
        """
        Initial the train class, which renders u to train directly 
        Parameters
        ----------
        model type 0 for Pure LSTM model, 1 for Convolutional model
        train
        device is to decide whether cpu or cuda. passes as string, should be used directly

        """ 
        self.device = device
    def train(self, dataloader, optimizer, criterion, model):
        
        model.train()    # set model to train mode
        train_loss = 0   # initialise the loss
        
        for x, y in dataloader:  # loop over dataset
            x = x.to(self.device)
            y = y.to(self.device)
            # send data to device
            optimizer.zero_grad()                # reset the gradients
            y_pred = model(x)            # get output and hidden state
            loss = criterion(y_pred, y)  # compute the loss (change shape as crossentropy takes input as batch_size, number of classes, d1, d2, ...)
            train_loss += loss                   

            loss.backward()                      # backpropagate
            optimizer.step()  
                                # update weights
        return train_loss/len(dataloader)
    def validate(self, dataloader, criterion, model):
        model.eval()
        validation_loss= 0.
        for X, y in dataloader:
            with torch.no_grad():
                X, y = X.to(self.device), y.to(self.device)
                a2 = model(X)
                #a2 = model(X.view(-1, 28*28)) #What does this have to look like for our conv-net? Make the changes!
                loss = criterion(a2.view(-1, 366*366), y.view(-1,366*366))
                validation_loss += loss*X.size(0)
    
                
        return validation_loss/len(dataloader.dataset)
    
    def train_whole_epoch(self, num_epoch, lr, momentum, trainloader, validationloader):
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        liveloss = PlotLosses()
        for epoch in range(num_epoch):
            logs = {}
            train_loss = self.train(trainloader,optimizer, criterion)
            print(epoch, train_loss)
            logs['' + 'log loss'] = train_loss.item()
            logs['' + 'log loss'] = train_loss.item()
            logs['val_'+ 'log loss'] = self.validate( validationloader, criterion).item()
            liveloss.update(logs)
            liveloss.draw()

    def show_result_for_last_five(self, x, y, mean, std, model, device, seq_num):
        def scale2range(x, range):
        # Scale x into a range, both expected to be floats
            return (x - x.min()) * (max(range) - min(range)) / (x.max() - x.min()) + min(range)
        model.eval()
        transform = T.ToPILImage() 
        # old one
        x = x.view(1, -1, 1, 366, 366).to(device)
        y = y.view(1, -1, 1, 366, 366)
        img = model(x)
        newx = x
        imgs = []
        imgs.append(img)
        for _ in range(4): 
            newx_temp = newx.clone()
            newx_temp[:,:seq_num - 1,0,:,:] = newx[:,1:seq_num,0,:,:]
            newx_temp[:,seq_num- 1,0,:,:] = img.view(-1, 366, 366)
            img = model(newx_temp)
            imgs.append(img)
            newx = newx_temp
 
        original_imgs = []
        single_imgs = []

        for index, img in enumerate(imgs):
            print("prediction:", index,' is shown')
            # mean and std
            single_img = img.view(-1, 366, 366)[0, :]*std + mean
            single_imgs.append(single_img)
            plt.imshow(transform(single_img))
            plt.show()
            target_img = y[:,index, 0, :, :].view(-1, 366, 366)[0,:]*std + mean
            original_imgs.append(target_img)
            plt.imshow(transform(target_img))
            plt.show()
        s_old = []
        m_old = []
        print(len(original_imgs))
        for i in range(5):
  
            original_imgs[i] = original_imgs[i].view(366,366)
            single_imgs[i] = (single_imgs[i]) * 255.0
            original_imgs[i] = (original_imgs[i]) * 255.0

            single_imgs[i] = single_imgs[i].cpu().detach().numpy() 
            original_imgs[i] = original_imgs[i].cpu().detach().numpy()


            original_imgs[i] = scale2range(original_imgs[i], [single_imgs[i].min(), single_imgs[i].max()])

            single_imgs[i], original_imgs[i] = single_imgs[i].astype(np.uint8), original_imgs[i].astype(np.uint8)

            s = ssim(single_imgs[i], original_imgs[i])
            m = mse(single_imgs[i], original_imgs[i])
            s_old.append(s)
            m_old.append(m)

        print("std:", s_old)
        print("ssim:", m_old)
