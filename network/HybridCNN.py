import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, alphasize, emb_dim, cnn_dim = 256, dropout = 0):

        self.cnn_net = nn.Sequential(
        # 201 x alphasize(71)
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

        self.rnn_net = nn.Sequential(


        )



    def forward(self, input):
        """
        Args:
            input (array): onehot text vector, 201 * 71, [doc_length] * [alphabet_size]
        """
        output = self.cnn_net(input)
        # return #batch x 8 x 256
