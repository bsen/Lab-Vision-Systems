"""
Some functions for training.
"""
import numpy as np
import torch
from my_utils import device, base_path
from torchvision.utils import save_image
import os


def train_epoch(model, train_loader, optimizer, criterion, epoch, device):
    """ Training a model for one epoch """

    loss_list = []
    recons_loss = []
    vae_loss = []

    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass
        recons, (z, mu, log_var) = model(images)

        # Calculate Loss
        loss, (mse, kld) = criterion(recons, images, mu, log_var)
        loss_list.append(loss.item())
        recons_loss.append(mse.item())
        vae_loss.append(kld.item())

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

    mean_loss = np.mean(loss_list)

    return mean_loss, loss_list

@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch=None, savefig=False, savepath=""):
    """ Evaluating the model for either validation or test """
    loss_list = []
    recons_loss = []
    vae_loss = []

    for i, (images, _) in enumerate(eval_loader):
        images = images.to(device)

        # Forward pass
        recons, (z, mu, log_var) = model(images)
        #print(reconds, (z,mu,log_var))

        loss, (mse, kld) = criterion(recons, images, mu, log_var)
        loss_list.append(loss.item())
        recons_loss.append(mse.item())
        vae_loss.append(kld.item())

        if(i==0 and savefig):
            save_image( recons[:64].cpu(), os.path.join(base_path, savepath, f"recons{epoch}.png") )

    # Total correct predictions and loss
    loss = np.mean(loss_list)

    return loss

def train_model(model, optimizer, scheduler, criterion, train_loader,
                valid_loader, num_epochs, savepath, model_path=None):
    """ Training a model for a given number of epochs"""

    train_loss = []
    val_loss =  []
    loss_iters = []

    for epoch in range(num_epochs):
        print('Epoch', epoch)
        # validation epoch
        model.eval()  # important for dropout and batch norms
        log_epoch = (epoch % 5 == 0 or epoch == num_epochs - 1)
        loss = eval_model(
                model=model, eval_loader=valid_loader, criterion=criterion,
                device=device, epoch=epoch, savefig=log_epoch, savepath=savepath
            )
        val_loss.append(loss)

        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model, train_loader=train_loader, optimizer=optimizer,
                criterion=criterion, epoch=epoch, device=device
            )
        scheduler.step(val_loss[-1])
        train_loss.append(mean_loss)
        loss_iters = loss_iters + cur_loss_iters

        if(log_epoch):
            print(f"    Train loss: {round(mean_loss, 5)}")
            print(f"    Valid loss: {round(loss, 5)}")

    print(f"Training completed")

    if model_path != None:
        path = os.path.join(base_path, model_path)
        torch.save(
            {
                'model': model,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'loss_iters': loss_iters,
            },
            f=path)

    return train_loss, val_loss, loss_iters
