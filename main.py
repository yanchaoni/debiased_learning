import Constants
from dataloader import *
from models import *
import os
import pickle as pkl
from preprocess import *
import torch
from train import *
from test import *


BATCH_SIZE = 32

processed_data_path = "/scratch/yn811/snli_1.0/train_val.pkl"
if os.path.exists(processed_data_path):
    print("found existing preprocessed data!")
    train_data, val_data, token2id, id2token = pkl.load(open(processed_data_path, "rb"))
else:
    train_data = get_data("/scratch/yn811/snli_1.0/snli_1.0_train.jsonl")
    token2id, id2token = build_vocab(train_data[3])
    val_data = get_data("/scratch/yn811/snli_1.0/snli_1.0_dev.jsonl", False)
    train_data, val_data = token2index_dataset(train_data, token2id), token2index_dataset(val_data, token2id)
    pkl.dump([train_data, val_data, token2id, id2token], open(processed_data_path, "wb"))

pretrained_emb_path = "/scratch/yn811/snli_fasttext_pretrained.pickle"
if os.path.exists(pretrained_emb_path):
    print("found existing loaded pretrained embeddings!")
    pretrained = pkl.load(open("/scratch/yn811/hw2_pretrained.pickle", "rb"))
else:
    pretrained = load_vectors("/scratch/yn811/wiki-news-300d-1M.vec", id2token)
    pkl.dump(pretrained, open(pretrained_emb_path, "wb"))

notPretrained = []
embeddings = [get_pretrain_emb(pretrained, token, notPretrained) for token in id2token]
notPretrained = torch.FloatTensor(np.array(notPretrained)[:, np.newaxis]).to(Constants.DEVICE)


train_dataset = SNLIDataset(train_data)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_func,
                                           shuffle=True)

val_dataset = SNLIDataset(val_data)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                           batch_size=BATCH_SIZE,
                                           collate_fn=collate_func,
                                           shuffle=True)

fc_hidden_size = 300
model = BagOfWords(vocab_size=len(embeddings), emb_dim=300, fc_hidden_size=fc_hidden_size, interaction="cat", embeddings=embeddings).cuda()
# _, val_acc_list = train(model, train_loader, val_loader, 6, label="test")

label = "test"
model.load_state_dict(torch.load('model' + "-" + label + '.ckpt'))
correct_list, wrong_list = get_predicted_examples(val_dataset, model, id2token, label="Bag-of-Word")
# plot_confidence(correct_list, wrong_list, label="test")
# plot_jaccard_sim(train_data)