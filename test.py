import Constants
from dataloader import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def get_predicted_examples(val_dataset, model, id2token, label="", write=False):
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                               batch_size=1,
                                               collate_fn=collate_func,
                                               shuffle=True)
    correct_cnt, wrong_cnt = 0, 0
    correct_list = []
    wrong_list = []

    model.eval()
    for data, lengths, ind_sort, ind_unsort, labels in val_loader:
        outputs = F.softmax(model(data, lengths), dim=1)
        prob, predicted = outputs.topk(k=1, dim=1)


        correct = predicted.eq(labels.view_as(predicted)).sum().item()
        if (correct == 1):
            correct_cnt  += 1
            correct_list.append((data[0][0].cpu().numpy(), data[1][0].cpu().numpy(), predicted.data[0].item(), prob.data[0].item()))
        if (correct == 0):
            wrong_cnt += 1
            wrong_list.append((data[0][0].cpu().numpy(), data[1][0].cpu().numpy(), predicted.data[0].item(), labels.data[0], prob.data[0].item()))
    
    print("{} wrong examples; {} correct examples".format(wrong_cnt, correct_cnt))
    
    if write:
        with open("results/correct_list_%s.txt" % label, "w+") as f:
            for ex in correct_list:
                f.write("---label: {} {:.3}\n -Premise: {}\n -Hypothesis: {}\n".format(
                    Constants.idx2lab[ex[2]], ex[-1], " ".join([id2token[i] for i in ex[0] if i > 0]), 
                    " ".join([id2token[i] for i in ex[1] if i > 0])))
        with open("results/wrong_list_%s.txt" % label, "w+") as f:
            for ex in wrong_list:
                f.write("---label: {}/{} {:.3}\n -Premise: {}\n -Hypothesis: {}\n".format(Constants.idx2lab[ex[3]],
                    Constants.idx2lab[ex[2]], ex[-1], " ".join([id2token[i] for i in ex[0] if i > 0]), 
                    " ".join([id2token[i] for i in ex[1] if i > 0])))
    return correct_list, wrong_list



def plot_confidence(correct_list, wrong_list, label):
    fig, ax = plt.subplots(3,3, figsize=(12,7))
    for i in range(3):
        for j in range(3):
            if i == j:
                ax[i, j].hist([ex[-1] for ex in correct_list if ex[2] == i], bins=30)
                ax[i, j].set_xlabel("prob")
                ax[i, j].set_title("%s: %d" %(Constants.idx2lab[i], sum(1 for ex in correct_list if ex[2] == i)))
            else:
                ax[i, j].hist([ex[-1] for ex in wrong_list if (ex[3]==i) and (ex[2]==j)], bins=30)
                ax[i, j].set_xlabel("prob")
                ax[i, j].set_title("%s/%s: %d" %(Constants.idx2lab[i], Constants.idx2lab[j], 
                                                 sum(1 for ex in wrong_list if (ex[3]==i) and (ex[2]==j))))
    plt.tight_layout()
    plt.savefig("results/confidence_dist_%s" % label)

def get_jaccard_sim(pre, hyp): 
    a = set(pre) 
    b = set(hyp)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def plot_jaccard_sim(train_data):
    jaccard_similarity = [[], [], []]
    cnt = [0, 0, 0]
    for sample in train_data:
        jaccard_similarity[sample[2]].append(get_jaccard_sim(sample[0], sample[1]))
        cnt[sample[2]] += 1
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    for i, ax in enumerate(axs):
        ax.hist(jaccard_similarity[i], bins=30)
        ax.set_title(Constants.idx2lab[i])
    plt.tight_layout()
    plt.savefig("results/jaccard_sim_by_class")
