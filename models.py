import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BagOfWords(nn.Module):
    def __init__(self, vocab_size, emb_dim, fc_hidden_size, interaction="cat", embeddings=None):
        super(BagOfWords, self).__init__()

        self.interaction = interaction
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if embeddings is not None:
            self.embed.weight.data.copy_(torch.from_numpy(np.array(embeddings)))
            self.embed.weight.requires_grad = False
        factor = 2 if interaction == "cat" else 1
        self.linear = nn.Sequential(nn.Linear(emb_dim*factor, fc_hidden_size), 
                                    nn.ReLU(),
                                    nn.Linear(fc_hidden_size, 3))
    
    def avg_emb(self, data, length):
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
        return out
        
    def forward(self, data, length):
        out0 = self.avg_emb(data[0], length[0])
        out1 = self.avg_emb(data[1], length[1])
        if self.interaction == "cat":
            out = torch.cat((out0, out1), dim=1)
        elif self.interaction == "minus":
            out = out0 + out1
        out = self.linear(out.float())
        return out    