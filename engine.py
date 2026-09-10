"""
engine.py contains- train_one_epoch,
                    validate_one_epoch,
                    fit
can be used for training loop for any PyTorch model (LSTM, GRU, CNN, plain linear)
with any DataLoader + loss function.

Not for classification , models returning tuples/multiple outputs.
"""

import torch


def train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch=0):
    """
    Runs 1 full pass over the training data.

    model:          your model object (e.g. LSTM(...).to(device))
    train_loader:   DataLoader wrapping your TRAINING data
    optimizer:      e.g. torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function:  e.g. nn.MSELoss()
    device:         "cuda" or "cpu" — put whatever device your tensors are on
    epoch:          just used for the printed log line, put current epoch number

    Returns: average training loss for this epoch
    """
    model.train()  # training mode

    total_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        # 1) Forward pass
        output = model(x_batch)

        # 2) Compare prediction to the real answer
        loss = loss_function(output, y_batch)

        # 3) Reset old gradients, compute new ones
        optimizer.zero_grad()
        loss.backward()

        # 4) Update the model's weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate_one_epoch(model, test_loader, loss_function, device, epoch=0):
    """
    No weight updates.

    Same parameters as train_one_epoch.

    Returns: average validation loss for this epoch (a float)
    """
    model.eval()  # evaluation mode

    total_loss = 0.0

    with torch.no_grad(): 
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            output = model(x_batch)
            loss = loss_function(output, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    print(f"Epoch {epoch + 1} | Val Loss:   {avg_loss:.4f}")
    return avg_loss


def fit(model, train_loader, test_loader, optimizer, loss_function, device, epochs):
    """
    Runs the FULL training process for a given number of epochs, calling
    train_one_epoch + validate_one_epoch each time, and collecting the
    loss history so you can plot it afterwards.

    Returns: train_losses, val_losses
    """
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_function, device, epoch)
        val_loss = validate_one_epoch(model, test_loader, loss_function, device, epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("-" * 40)

    return train_losses, val_losses