import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, alphasize, cnn_dim = 256, emb_dim = 1024, dropout = 0):
        super(TextEncoder, self).__init__()
        self.cnn_net = nn.Sequential(
        # 201 x alphasize(71)
        # the input should be (N, C_in, L) = [#batch, 71, 201]
        # which in torch7 the input is (N, L, C_in) -> (N, L, C_out)

        nn.Conv1d(alphasize, 384, 4), # (201, 71) -> (198, 384)
        nn.ReLU(),
        nn.MaxPool1d(3, 3), # (198, 384) -> ((198-3)/3+1, 384) = (66, 384)
        # 66 x 384
        nn.Conv1d(384, 512, 4), # (66, 384) -> (63, 512)
        nn.ReLU(),
        nn.MaxPool1d(3, 3), # (63, 512) -> (21, 512)
        # 21 x 512
        nn.Conv1d(512, cnn_dim, 4), # (21, 512) -> (18, 256)
        nn.ReLU(),
        nn.MaxPool1d(3, 2) # (18, 256) -> ((18-3)/2+1, 256) = (8, 256)
        # 8 x 256
        )

        # put #batch x 8 x 256 sequence into an rnn model
        # two way to get hidden layer's outputs
        #   1. customize an rnn
        #   2. try to get the intermidite value of rnn

        # length, batch size,
        self.rnn_net = nn.RNNCell(cnn_dim, cnn_dim, nonlinearity = 'relu')

        self.linear_net = nn.Linear(cnn_dim, emb_dim)

    def forward(self, input):
        """
        Args:
            input (array): onehot text vector, 201 * 71, [doc_length] * [alphabet_size]
        """
        output = self.cnn_net(input)
        output = output.permute(2,0,1) # batch, channel, length -> length, batch, channel

        length, batch, channel = output.shape

        h0 = torch.zeros(batch, channel)
        input0 = torch.ones(batch, channel)
        h = self.rnn_net(input0, h0)

        sum_hiden_layer = h0
        for i in range(length):
            h = self.rnn_net(output[i], h)
            sum_hiden_layer = sum_hiden_layer + h
        hiden_layer = sum_hiden_layer / length
        # print(hiden_layer.shape)
        hiden_layer = self.linear_net(hiden_layer)
        return hiden_layer
        # return #batch x 256, the encoder

if __name__ == '__main__':
    import birds_loader
