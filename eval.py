from tqdm import tqdm
from torch import nn
import torch
from train_def import val
from tsav import TwoStreamAuralVisualModel
from loss_functions import CCCEval
from aff2newdataset import Aff2CompDatasetNew
from torch.utils.data.dataloader import DataLoader
import os
from cleaner import clean_dataset


from strings import unpack_tuple

expression_classification_fn = nn.CrossEntropyLoss()
from loss_functions import f1_loss,CCCLoss

def get_accuracy(actual, expected):
    actual = (actual > 0.5).item()
    expected = (expected > 0.5).item()
    # print(expected == actual)
    return actual == expected


def eval(set, device,model_path="Final_Model.pth"):
    model = TwoStreamAuralVisualModel(num_channels=3).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    loop = tqdm(set, leave=False)
    exp_correct = 0
    exp_total = 0
    true_va = []
    predicted_va = []
    true_au = []
    predicted_au = []
    expression_true = []
    expression_pred = []
    for data in loop:
        with torch.no_grad():
            input = data['clip'].to(device)
            # expected
            expressions = data['expressions'].to(device)
            au0 = data['au0'].to(device)
            au1 = data['au1'].to(device)
            au2 = data['au2'].to(device)
            au3 = data['au3'].to(device)
            au4 = data['au4'].to(device)
            au5 = data['au5'].to(device)
            au6 = data['au6'].to(device)
            au7 = data['au7'].to(device)
            au8 = data['au8'].to(device)
            au9 = data['au9'].to(device)
            au10 = data['au10'].to(device)
            au11 = data['au11'].to(device)
            valence = data['valience'].to(device)
            arousal = data['arousal'].to(device)
            # output
            output = model(input).to(device)
            output = output[0]
            expressions_ = torch.sigmoid(output[0:8]).to(device)
            expression_true.append(expressions.item())
            expressions_ = torch.argmax(expressions_).to(device)
            expression_pred.append(expressions_.item())
            exp_total += 1
            if(expressions.item() == expressions_.item()):
                exp_correct += 1
            au0_ = output[8].to(device)
            au1_ = output[9].to(device)
            au2_ = output[10].to(device)
            au3_ = output[11].to(device)
            au4_ = output[12].to(device)
            au5_ = output[13].to(device)
            au6_ = output[14].to(device)
            au7_ = output[15].to(device)
            au8_ = output[16].to(device)
            au9_ = output[17].to(device)
            au10_ = output[18].to(device)
            au11_ = output[19].to(device)
            if(get_accuracy(torch.sigmoid(au0_), au0)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au1_), au1)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au2_), au2)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au3_), au3)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au4_), au4)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au5_), au5)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au6_), au6)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au7_), au7)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au8_), au8)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au9_), au9)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au10_), au10)):
                correct += 1
            if(get_accuracy(torch.sigmoid(au11_), au11)):
                correct += 1
            total += 12
            valence_ = output[20].to(device)
            arousal_ = output[21].to(device)
            predicted_au.append(arousal_.item())
            true_au.append(arousal.item())
            predicted_va.append(valence_.item())
            true_va.append(valence.item())
            loop.set_postfix(acc=correct/total)
    mean_au_accuracy = correct/total
    expresions_accuracy = exp_correct/exp_total
    return (mean_au_accuracy, expresions_accuracy,f1_loss((expression_true),expression_pred,"macro"),f1_loss((expression_true),expression_pred,"weighted"),CCCEval(true_au,predicted_au),CCCEval((true_va),(predicted_va)))

def eval_all(set, device,directory='models'):
    os.remove("results.txt")
    f = open("results.txt", "x")
    f.close()
    f = open("results.txt", "w")
    f.write("File, Accuracy")
    f.close()
    list = os.listdir(directory)
    list.sort()
    max_v = -1
    max_name = ""
    for file in list:
        with open('results.txt', 'a') as filewr:
            acc = eval(set, device, model_path=str(
                os.path.join(directory, file)))
            if(acc[1] >= max_v):
                max_v = acc[1]
                max_name = file
                print(max_v)
            filewr.write("\n " + str(file) + ", " +
                         unpack_tuple(acc))
    f.close()
    print("best acc: " + max_name + " " + str(acc))
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    test_set = Aff2CompDatasetNew(
        root_dir='aff2_processed', mtl_path='mtl_data', dataset_dir='test_set.txt')
    test_set = clean_dataset(test_set)
    test_loader = DataLoader(
        dataset=test_set, num_workers=8, batch_size=1, shuffle=True)
    eval_all(test_loader, device)
