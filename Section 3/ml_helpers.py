from tqdm import tqdm
from rendering import rendering

import torch


def training(model, optimizer, scheduler, tn, tf, nb_bins, nb_epochs, data_loader, device="cpu"):
    
    train_loss=[]
    for epoch in range(nb_epochs):
        for batch in tqdm(data_loader):
            o = batch[:,:3].to(device) #origin
            d = batch[:,3:6].to(device) #density

            target = batch[:,6:].to(device)

            prediction = rendering(model, o, d, tn, tf, nb_bins=nb_bins, device=device) #still works with several rays in a batch?

            loss = ((prediction - target)**2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            train_loss.append(loss.item())
        scheduler.step()
    return train_loss

