from cleaner import clean_dataset
import torch
from train_def import val, train
from loss_functions import *
from torch.utils.data.dataloader import DataLoader
from tsav import TwoStreamAuralVisualModel
import wandb
import torch.optim
from aff2newdataset import Aff2CompDatasetNew
from eval import eval_all
from hyperparams import batch_size, num_workers, epochs, learning_rate
import argparse
parser = argparse.ArgumentParser(description="Train Script")
parser.add_argument("--save_location",help="Save Location of models",default="models")
args = parser.parse_args()

wandb.init(project="full dataset multi task learning baseline")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
train_set = Aff2CompDatasetNew(
    root_dir='aff2_processed', mtl_path='mtl_data', dataset_dir='train_set.txt')

test_set = Aff2CompDatasetNew(
    root_dir='aff2_processed', mtl_path='mtl_data', dataset_dir='test_set.txt')
val_set = Aff2CompDatasetNew(
    root_dir='aff2_processed', mtl_path='mtl_data', test_set=True, dataset_dir='val_set.txt')
train_set = clean_dataset(train_set)
val_set = clean_dataset(val_set)
test_set = clean_dataset(test_set)
print(len(train_set))
print(len(test_set))
print(len(val_set))
train_loader = DataLoader(
    dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(
    dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    dataset=test_set, num_workers=num_workers, batch_size=1, shuffle=True)
model = TwoStreamAuralVisualModel(num_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_workers": num_workers
}
model.train()
wandb.watch(model)
for epoch in range(epochs):
    val(val_loader, model, device, epoch)
    print("validated")
    optimizer.zero_grad()
    print("zeroed gradients and training")
    train(train_loader, model, device, optimizer, epoch)
    optimizer.step()
    torch.save(model.state_dict(), f'{args.save_location}/{epoch+1}_epoch_model.pth')
eval_all(test_loader, device)
