import torch
import wandb
from tqdm import tqdm
def clean_dataset(set):
    i = 0
    removed_for_invalid_csv= 0
    removed_for_data_none = 0
    removed_for_nan =  0
    removed_for_zeros = 0
    removed_for_first = 0
    good = 0
    zeros = torch.zeros((3,set.vid_len,112,112)).cuda()
    for z in tqdm(range(set.__len__()),leave=False):
        data = set.__getitem__(i)
        data['clip'] = data['clip'].cuda()
        if(data['valience']==-5.0 or data['arousal'] == -5.0 or data['expressions']==-1 or data['au0']==None):
            removed_for_invalid_csv+=1 
            set.__remove__(i)
            i = i-1
        elif(data == None):
            removed_for_data_none +=1
            set.__remove__(i)
            i = i-1
        elif(data['clip']==None):
            removed_for_first+=1 
            set.__remove__(i)
            i = i-1
        elif (torch.isnan(data['clip']).any()):
            removed_for_nan+=1
            set.__remove__(i)
            i = i-1
        elif(torch.equal(data['clip'],zeros)):
            removed_for_zeros+=1
            set.__remove__(i)
            i = i-1
        else:
            good +=1 
        try:    
            wandb.log({"total":z,"total good":good, "removed for invalid csv (ex: -5 on arousal)":removed_for_invalid_csv,"removed for none data":removed_for_data_none,"removed_for_nan":removed_for_nan,"removed_for_zeros":removed_for_zeros,"removed_for_first":removed_for_first})    
        except:
            pass
        i+=1
    return set
