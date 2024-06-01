# Per entrenar el model MLP


# The Loss

criterion = torch.nn.CrossEntropyLoss() # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss



# Optimitzador
learning_rate = 1e-3
lambda_l2 = 1e-5

# we use the optim package to apply gradient descent for our parameter updates
optimizer = torch.optim.SGD(modelMLP.parameters(), lr=learning_rate, momentum=0.9, weight_decay=lambda_l2) # built-in L2


# The training loop

def train(x_train, y_train, x_val, y_val, criterion, model, optimizer, epochs=300, visualize_surface=False):

    losses = {"train": [], "val": []} # Two lists to keep track of the evolution of our losses

    for t in range(epochs):

        # activate training mode
        model.train()

        # Feed forward to get the logits
        y_pred = model(x_train) # x_train is the whole batch, so we are doing batch gradient descent

        # Compute the loss
        loss = criterion(y_pred.squeeze(), y_train)

        # zero the gradients before running the backward pass
        optimizer.zero_grad()

        # Backward pass to compute the gradient of loss w.r.t our learnable params
        loss.backward()

        # Update params
        optimizer.step()

        # Compute the accuracy.
        score, predicted = torch.max(y_pred, 1) # torch.max() returns the maximum value and the argmax (index of the maximum value)
        train_acc = (y_train == predicted).sum().float() / len(y_train)
        losses["train"].append(loss.item()) # keep track of our training loss

        # Run model on validation data
        val_loss, val_acc = calculateLossAcc(criterion, model, x_val, y_val) # Call our helper function on the validation set
        losses["val"].append(val_loss.item()) # keep track of our validation loss

        # Create plots
        display.clear_output(wait=True)
        draw_plots(x_val, y_val, model, losses, visualize_surface, visualize_regressor = False)

        print("Training: [EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, loss.item(), train_acc))
        print("Validation: [EPOCH]: %i, [LOSS]: %.6f, [ACCURACY]: %.3f" % (t, val_loss.item(), val_acc))

    return losses # In case we want to plot them afterwards

# A helper function that calculates our loss and accuracy on a given dataset (by default on our validation set)
def calculateLossAcc(criterion, model, x, y):

    # set model in evaluation mode
    model.eval()
    with torch.no_grad(): # do not compute gradients for validation
        y_pred = model(x)


    # compute loss and accuracy
    _, predicted = torch.max(y_pred, 1)
    loss = criterion(y_pred.squeeze(), y)
    acc = (y == predicted).sum().float() / len(y)

    return loss, acc



# Run training
losses = train(xSpiralTrain, ySpiralTrain, xSpiralVal, ySpiralVal, criterion, modelMLP, optimizer, visualize_surface=True, epochs = 100)

draw_plots(xBlobsVal, yBlobsVal, modelMLP, losses, visualize_surface = True)