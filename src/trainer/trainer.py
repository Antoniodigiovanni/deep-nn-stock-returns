import torch


def train_one_step(model, data, optimizer, loss_fn, device):
    optimizer.zero_grad()
    
    for k,v in data.items():
        data[k] = v.to(device)
    
    yhat = model(data['X'])

    #yhat = model(**data)
    loss = loss_fn(yhat.ravel().float(), data['Y'].ravel().float())

    loss.backward()
    optimizer.step()
    return loss

def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer, loss_fn, device)
        #scheduler.step()
        total_loss += loss
        
    # Make loss batch_size-agnostic (correct?)
    total_loss = total_loss / data_loader.batch_size
    return total_loss

##############
# Validation #
##############

def validate_one_step(model, data, optimizer, criterion, device):
    optimizer.zero_grad()
    
    for k,v in data.items():
        data[k] = v.to(device)
    
    with torch.no_grad():
        yhat = model(data['X'])
    
    loss = criterion(yhat.ravel().float(), data['Y'].ravel().float())
    #print(f'Prediction is: {yhat.ravel().float()}')
    #print(f'Original Value is: {data["Y"].ravel().float()}')
    # loss.backward()
    # optimizer.step()
    return loss

def validate_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train(False) # Same as model.eval()
    total_loss = 0

    for batch_index, data in enumerate(data_loader):
        loss = validate_one_step(model, data, optimizer, loss_fn, device)
        total_loss += loss
    # Make loss batch_size-agnostic (correct?)
    total_loss = total_loss / data_loader.batch_size
    return total_loss