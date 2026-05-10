import sys, os
import torch
import torch.nn as nn

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 100
    TRAIN_BATCH_SIZE = batch_size
    loss_fn = torch.nn.MSELoss()

    epoch_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        labels = data_mol.y.view(-1, 1)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        # accumulate for final print
        epoch_loss += loss.item()
        num_batches += 1

        # if batch_idx % LOG_INTERVAL == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch,
        #         batch_idx * TRAIN_BATCH_SIZE,
        #         len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader),
        #         loss.item()
        #     ))

    # NEW: print final loss for the whole epoch
    print("Epoch {} Final Loss: {:.6f}".format(epoch, epoch_loss / num_batches))


# predict
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            labels = data_mol.y.view(-1, 1)
            total_preds = torch.cat((total_preds, output.cpu()), 0)

            total_labels = torch.cat((total_labels, labels.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_benchmark(model, device, train_loader, optimizer, epoch, batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 100
    TRAIN_BATCH_SIZE = batch_size
    loss_fn = torch.nn.MSELoss()
   
    for batch_idx, data in enumerate(train_loader):
        data = cuda(data, device = device)
        labels = data[0].y.view(-1, 1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
def predicting_benchmark(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = cuda(data, device = device)
            labels = data[0].y.view(-1, 1)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, labels.cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))