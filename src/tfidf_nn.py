from os import path, getcwd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from dataset_loader import AbstractsDataset
from metrics import calculate_metrics


DATA_DIR = path.join(getcwd(), 'data', 'processed')
VOCAB_SIZE = None
DATASET = 'top5'
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
HIDDEN_IN = 1024
HIDDEN_OUT = 512
EPOCHS = 20


def main():
    # load dataset
    fname = path.join(DATA_DIR, DATASET + '.csv')
    df = pd.read_csv(fname, quotechar='"', dtype=str)
    vectorizer = TfidfVectorizer(encoding='ascii', max_features=VOCAB_SIZE)
    X = vectorizer.fit_transform(df['abstract'])
    y = np.array([list(map(int, mask)) for mask in df['keywords']],
                 dtype='uint8')

    # train-validate-test split
    inds = np.random.permutation(df.shape[0])
    i1, i2 = (df.shape[0] * np.array([0.9, 0.95])).astype(int) 
    train_dataset = AbstractsDataset(X[inds[:i1]], y[inds[:i1]], BATCH_SIZE)
    val_dataset = AbstractsDataset(X[inds[i1:i2]], y[inds[i1:i2]], BATCH_SIZE)
    test_dataset = AbstractsDataset(X[inds[i2:]], y[inds[i2:]], BATCH_SIZE)

    # choose device and initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = Model(
        num_features=X.shape[1],
        num_labels=y.shape[1],
        shape_hidden=(HIDDEN_IN, HIDDEN_OUT)
    )
    model = model.to(device)

    # train model
    train_losses, val_losses, train_metrics, val_metrics = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # plot results
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.figure()
    plt.plot(train_metrics[:, 0])
    plt.plot(val_metrics[:, 0])
    plt.xlabel('Epoch')
    plt.ylabel('Mean accuracy')

    plt.figure()
    plt.plot(train_metrics[:, 1])
    plt.plot(val_metrics[:, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Mean precision')

    plt.figure()
    plt.plot(train_metrics[:, 2])
    plt.plot(val_metrics[:, 2])
    plt.xlabel('Epoch')
    plt.ylabel('Mean recall')

    plt.show()


class Model(nn.Module):
    """
    Neural net with a single hidden layer.
    """
    def __init__(self, num_features, num_labels, shape_hidden=(1024, 512)):
        super().__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(num_features, shape_hidden[0]),
            nn.ReLU(),
            nn.Linear(shape_hidden[0], shape_hidden[1]),
            nn.ReLU(),
            nn.Linear(shape_hidden[1], num_labels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)


def train(model, train_dataset, val_dataset, device, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = np.zeros(epochs)
    train_metrics = np.zeros((epochs, 3))
    val_losses = np.zeros_like(train_losses)
    val_metrics = np.zeros_like(train_metrics)
    loss_function = nn.BCELoss()

    for i in range(epochs):
        print(f'Epoch {i}')

        # train loop
        losses = []
        accuracies = []
        precision_values = []
        recall_values = []
        batch_sizes = []
        for X_j, y_j in train_dataset:
            # forward loop
            X_j = torch.tensor(X_j, dtype=torch.float32, device=device)
            y_j = torch.tensor(y_j, dtype=torch.float32, device=device)
            output = model.forward(X_j)
            loss = loss_function(output, y_j)
            # evaluate and save metrics
            losses.append(loss.item())
            predicted = np.round(output.cpu().detach().numpy()).astype('uint8')
            m = calculate_metrics(predicted, y_j.cpu().numpy().astype('uint8'))
            accuracies.append(m.accuracy.mean())
            precision_values.append(m.precision.mean())
            recall_values.append(m.recall.mean())
            batch_sizes.append(X_j.shape[0])
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        weights = np.array(batch_sizes) / sum(batch_sizes)
        train_losses[i] = sum(x * w for x, w in zip(losses, weights))
        train_metrics[i] = (
            sum(x * w for x, w in zip(accuracies, weights)),
            sum(x * w for x, w in zip(precision_values, weights)),
            sum(x * w for x, w in zip(recall_values, weights))
        )
        print(f'Train: loss = {train_losses[i]:.3f},', end=' ')
        print('acc = {:.3f}, prec = {:.3f}, rec = {:.3f}'.format(
            *train_metrics[i]))

        # validation loop
        losses = []
        accuracies = []
        precision_values = []
        recall_values = []
        batch_sizes = []
        with torch.no_grad():
            for X_j, y_j in val_dataset:
                # forward loop
                X_j = torch.tensor(X_j, dtype=torch.float32, device=device)
                y_j = torch.tensor(y_j, dtype=torch.float32, device=device)
                output = model.forward(X_j)
                loss = loss_function(output, y_j)
                # evaluate and save metrics
                losses.append(loss.item())
                predicted = np.round(output.cpu().numpy()).astype('uint8')
                m = calculate_metrics(predicted,
                                      y_j.cpu().numpy().astype('uint8'))
                accuracies.append(m.accuracy.mean())
                precision_values.append(m.precision.mean())
                recall_values.append(m.recall.mean())
                batch_sizes.append(X_j.shape[0])

        weights = np.array(batch_sizes) / sum(batch_sizes)
        val_losses[i] = sum(x * w for x, w in zip(losses, weights))
        val_metrics[i] = (
            sum(x * w for x, w in zip(accuracies, weights)),
            sum(x * w for x, w in zip(precision_values, weights)),
            sum(x * w for x, w in zip(recall_values, weights))
        )
        print(f'Validation: loss = {val_losses[i]:.3f},', end=' ')
        print('acc = {:.3f}, prec = {:.3f}, rec = {:.3f}'.format(
            *val_metrics[i]))
        print()

    return train_losses, val_losses, train_metrics, val_metrics


if __name__ == '__main__':
    main()
