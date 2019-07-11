import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, alphasize, cnn_dim = 256, dropout = 0):
        super(TextEncoder, self).__init__()
        self.cnn_net = nn.Sequential(
        # 201 x alphasize(71)
        # the input should be (N, C_in, L) = [#batch, 71, 201]
        # which in torch7 the input is (N, L, C_in) -> (N, L, C_out)
        nn.Conv1d(alphasize, 384, 4), # (201-4+1, 384) = (198, 384)
        nn.ReLU(),
        nn.MaxPool1d(3, 3), # ((198-3)/3+1, 384) = (68, 384)
        # 66 x 384
        nn.Conv1d(384, 512, 4),
        nn.ReLU(),
        nn.MaxPool1d(3, 3),
        # 21 x 512
        nn.Conv1d(512, cnn_dim, 4),
        nn.ReLU(),
        nn.MaxPool1d(3, 2)
        # 8 x 256
        )

        # put #batch x 8 x 256 sequence into an rnn model
        # two way to get hidden layer's outputs
        #   1. customize an rnn
        #   2. try to get the intermidite value of rnn

        # length, batch size,
        self.rnn_net = nn.RNNCell(cnn_dim, cnn_dim, nonlinearity = 'relu')


    def forward(self, input):
        """
        Args:
            input (array): onehot text vector, 201 * 71, [doc_length] * [alphabet_size]
        """
        output = self.cnn_net(input)
        n_batch, n_channel, length = output.shape

        # change the shape of in
        output = output.view([length, n_batch, n_channel])
        h0 = torch.zeros(n_batch, n_channel)
        input0 = torch.ones(n_batch, n_channel)
        h = self.rnn_net(input0, h0)

        sum_hiden_layers = h0
        for i in range(length):
            h = self.rnn_net(output[i], h)
            sum_hiden_layers += h

        return sum_hiden_layers / length
        # return #batch x 256, the encoder

if __name__ == '__main__':
    import birds_loader
