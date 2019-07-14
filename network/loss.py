import torch

def JointEmbeddingLoss(fea_txt, fea_img, labels):
    """
    Args:
        input(tensor): the shape of input shold be [batch size, emb_dim]

    """

    batch_size = fea_txt.shape[0]
    num_class = len(labels)
    score = torch.zeros(batch_size, num_class)
    loss = 0
    acc_batch = 0
    '''
    for i in range(batch_size):
        for j in range(num_class):
            score[i, j] = torch.dot(fea_img[i])
    '''
    score = torch.mm(fea_img, torch.transpose(fea_txt, 0, 1))
    for i in range(batch_size):
        label_score = score[i,i]
        for j in range(num_class):
            if j != i:
                # cur_score: the score of i.th image and j.th text
                cur_score = score[i,j]
                thresh = cur_score - label_score + 1
                if thresh > 0:
                    loss += thresh
    return loss / (batch_size * num_class)
