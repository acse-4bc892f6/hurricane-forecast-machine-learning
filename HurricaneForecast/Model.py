import torch
import torch.nn as nn
# import torchvision.models as models
# import torch.nn.functional as F

class pureLSTM(nn.Module):
    def __init__(self, size_of_image, device):
        """
        Initialize Pure LSTM network to predict the next storm.
        Parameters
        ----------
        size_of_image: int
            the size of img( as all of our img should be square).
        device: string
            Whether CPU GPU or other things and so on.
        """
        super(pureLSTM, self).__init__()
        self.size_of_image = size_of_image
        self.num_layers = 1
        self.hidden_size = 400
        self.device = device
        self.lstm_enc = nn.LSTM(input_size=size_of_image*size_of_image, hidden_size=400, num_layers=1, batch_first=True)
        self.fc_enc = nn.Linear(400, 250)
        self.fc_enc2 = nn.Linear(250, 100)
        self.fc_dec1 = nn.Linear(100, 500)
        self.fc_dec2 = nn.Linear(500, 600)
        self.fc_dec3 = nn.Linear(600, size_of_image*size_of_image)
    def init_states_h_c(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=True).to(self.device),
        torch.zeros(self.num_layers, batch_size, self.hidden_size,  requires_grad=True).to(self.device))  # initialise the hidden and cell states
    def forward(self, x_3d):
        batch_size = x_3d.size(0)
        hidden_enc = self.init_states_h_c(batch_size)
        out, hidden_enc = self.lstm_enc(x_3d.view(-1, x_3d.size(1), x_3d.size(3)*x_3d.size(4)), hidden_enc)   
        x = self.fc_enc(out[:, -1 ,:])
        x = torch.tanh(x)
        x = self.fc_enc2(x)
        x = torch.tanh(x)
        x = self.fc_dec1(x)
        x = torch.tanh(x)
        x = self.fc_dec2(x)
        x = torch.tanh(x)
        x = self.fc_dec3(x)
        x = x.view(-1, 1, 1, x_3d.size(3), x_3d.size(4))
        return x



class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, in_chan):
        """ ARCHITECTURE 
        # in_chan for the input channel number 
        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        super(EncoderDecoderConvLSTM, self).__init__()


        self.encoder_1_convlstm = ConvLSTMCell(input_dim=in_chan,
                                               hidden_dim=4,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.encoder_2_convlstm = ConvLSTMCell(input_dim=4,
                                               hidden_dim=16,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_1_convlstm = ConvLSTMCell(input_dim=16,  # nf + 1
                                               hidden_dim=8,
                                               kernel_size=(3, 3),
                                               bias=True)

        self.decoder_2_convlstm = ConvLSTMCell(input_dim=8,
                                               hidden_dim=5,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.encoder_3_convlstm = ConvLSTMCell(input_dim=5,
                                               hidden_dim=3,
                                               kernel_size=(3, 3),
                                               bias=True)
        self.decoder_CNN1 = nn.Conv3d(in_channels=3,
                                     out_channels=3,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        self.decoder_CNN2 = nn.Conv3d(in_channels=3,
                                     out_channels=1,
                                     kernel_size=(1, 3, 3),
                                     padding=(0, 1, 1))
        self.batch_norm1 = nn.BatchNorm2d(16)
    def autoencoder(self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5):

        outputs = []
        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(input_tensor=x[:, t, :, :],
                                               cur_state=[h_t, c_t])  # we could concat to provide skip conn here

            h_t2, c_t2 = self.encoder_2_convlstm(input_tensor=h_t,
                                                 cur_state=[h_t2, c_t2])  # we could concat to provide skip conn here

        encoder_vector = h_t2
        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(input_tensor=encoder_vector,
                                                 cur_state=[h_t3, c_t3])  # we could concat to provide skip conn here
            
            h_t4, c_t4 = self.decoder_2_convlstm(input_tensor=h_t3,
                                                 cur_state=[h_t4, c_t4])  # we could concat to provide skip conn here
            h_t5, c_t5 = self.encoder_3_convlstm(input_tensor=h_t4,
                                    cur_state=[h_t5, c_t5])  # we could concat to provide skip conn here
            encoder_vector = h_t5
            outputs += [h_t5]  # predictions


        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN1(outputs)
        outputs = torch.nn.Tanh()(outputs)
        outputs = self.decoder_CNN2(outputs)

        return outputs

    def forward(self, x, future_seq=1, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t5, c_t5 = self.encoder_3_convlstm.init_hidden(batch_size=b, image_size=(h, w))

        # autoencoder forward
        outputs = self.autoencoder(x, seq_len, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4, h_t5, c_t5)
        return outputs
