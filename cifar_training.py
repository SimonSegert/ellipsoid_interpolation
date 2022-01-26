import numpy as np
import torch
from torchvision.datasets import MNIST,CIFAR10
from torchvision import transforms
import cifarresnet
import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int,default=100)
parser.add_argument("--lr",type=float,default=.1)
parser.add_argument("--batch_size",type=int,default=256)
parser.add_argument("--ckpt_freq",type=int,default=-1)
parser.add_argument("--save_dir") #will create a directory with this name, and store the saved model+hparams there

args = parser.parse_args()

use_normalize = True
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

n_epochs=args.n_epochs
lr=args.lr
batch_size=args.batch_size
ckpt_freq=args.ckpt_freq
save_dir=args.save_dir
try:
    os.mkdir(save_dir)
except:
    raise ValueError('directory already exists')






device='cuda' if torch.cuda.is_available() else 'cpu'


t = [transforms.ToTensor()]
if use_normalize:
    t.append(normalize)
t = transforms.Compose(t)
ds = CIFAR10(root='/Users/simon/Downloads', train=True, transform=t, download=True)
ds_test = CIFAR10(root='/Users/simon/Downloads', train=False, transform=t, download=True)
trXflat = torch.cat([d[0].ravel().unsqueeze(0) for d in ds], 0).numpy()
testXflat = torch.cat([d[0].ravel().unsqueeze(0) for d in ds_test], 0).numpy()
trX = torch.cat([d[0].unsqueeze(0) for d in ds], 0).numpy()
testX = torch.cat([d[0].unsqueeze(0) for d in ds_test], 0).numpy()

trL = np.array([d[1] for d in ds])
testL = np.array([d[1] for d in ds_test])

dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

model=cifarresnet.load20().to(device)

opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=10e-4)

model.train()
losses = []
accs = []
print_freq=2 #prints progress after this many epochs
for epoch_id in range(n_epochs):
    print(f'epoch={epoch_id}')

    for ii, (a, b) in enumerate(dl):
        opt.zero_grad()
        a=a.to(device)
        b=b.to(device)
        p = model(a)
        l = torch.nn.CrossEntropyLoss()(p, b)
        l.backward()
        opt.step()
        preds = p.detach().argmax(1).numpy()
        acc = np.mean(preds == b.numpy())
        losses.append(l.detach().item())
        accs.append(acc)
        if epoch_id > 0 and epoch_id % print_freq == 0:
            print(np.mean(losses[-print_freq:]), np.mean(accs[-print_freq:]))
        if ckpt_freq>0 and epoch_id%ckpt_freq==0:
            fn=f'{save_dir}/ckpt_{epoch_id}.pth'
            model.eval()
            torch.save(model.state_dict(),open(fn,'wb'))
            model.train()

fn=f'{save_dir}/model.pth'
model.eval()
torch.save(model.state_dict(),open(fn,'wb'))

hparams=vars(args)
with open(f'{save_dir}/hparams.json','w') as f:
    json.dump(hparams,f)