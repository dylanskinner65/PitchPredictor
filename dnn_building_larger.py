import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from adabelief_pytorch import AdaBelief
import torch.optim as optim
import matplotlib.pyplot as plt


def main():
    '''
    Creating Datasets and DataLoaders

    We will create two different dataset classes: one for our training data, and one for our testing data. 
    Once these are done, we will also create a two dataloaders that will be used to create our training
    and testing data. We will use the dataloaders to train our model.
    '''
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset and dataloader.
    class PitchDatasetTrain(Dataset):
        def __init__(self):
            # Read in data
            df = pd.read_parquet(
                '~/PitchPredictor/training_data.parquet', engine='fastparquet')

            # Convert data to PyTorch tensors
            x_data = df.drop(columns=['pitch_type'])
            self.X = torch.tensor(x_data.to_numpy().astype(np.float32))
            self.y = torch.tensor(df['pitch_type'].values, dtype=torch.int8)
            self.n_samples = df.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.n_samples

    class PitchDatasetTest(Dataset):
        def __init__(self):
            # Read in data
            df = pd.read_parquet(
                '~/PitchPredictor/testing_data.parquet', engine='fastparquet')

            # Convert data to PyTorch tensors
            x_data = df.drop(columns=['pitch_type'])
            self.X = torch.tensor(x_data.to_numpy().astype(np.float32))
            self.y = torch.tensor(df['pitch_type'].values, dtype=torch.int8)
            self.n_samples = df.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.n_samples

    # Create the dataset and dataloader.
    train_data = PitchDatasetTrain()
    test_data = PitchDatasetTest()
    batch_size, num_workers = 8192, 1
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, pin_memory=True)

    '''
    Helper Functions

    These are the functions we will use in our model and/or training loop. 
    So far we only have a function to calculate the accuracy of our model, and our custom activation function.
    '''
    # Accuracy Checker Function
    def calculate_accuracy(model, dataloader):
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    # Define new activation function PenalizedTanH
    class PenalizedTanH(nn.Module):
        def __init__(self):
            super(PenalizedTanH, self).__init__()

        def forward(self, x):
            return torch.where(x > 0, torch.tanh(x), 0.25*torch.tanh(x))

    '''
    The Model

    This has our deep learning model. It is bigger than the original. We now use use the <code>AdaBelief</code>
    optimizer and the <code>CrossEntropyLoss</code> loss function.
    '''
    # Build the DNN

    # First, define the hyperparameters
    input_size = 31  # Input size (e.g., number of features)
    hidden_size = 64  # Size of the hidden layer(s)
    output_size = 19  # Output size (e.g., number of classes)
    learning_rate = 0.0001

    # Define the neural network architecture
    class PitchDNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(PitchDNN, self).__init__()
            self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size, hidden_size*2),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size*2, hidden_size*4),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size*4, hidden_size*4),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size*4, hidden_size*2),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size*2, hidden_size),
                                     PenalizedTanH(),
                                     nn.Linear(hidden_size, output_size))

        def forward(self, x):
            return self.net(x)

    class TransformerModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers):
            super(TransformerModel, self).__init__()
            self.transformer = nn.Transformer(
                d_model=input_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
            )
            self.linear = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc = nn.Lnear(hidden_size, output_size)

        def forward(self, x):
            x = x.float()
            x = self.transformer(x, x)
            x = self.fc(x)
            return x

    # Create an instance of the model
    model = TransformerModel(input_size, hidden_size, output_size)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define variables for training loop.
    num_epochs = 12
    total_samples = len(train_data)
    n_iterations = np.ceil(total_samples/batch_size)

    # Define variables for storing our loss and accuracy.
    loss_vals, loss_occ = [], []
    acc_vals, acc_occ = [], []

    # Run training loop.
    for epoch in range(num_epochs):
        # Create our tqdm progress bar.
        tqdm_data_loader = tqdm(train_dataloader, total=n_iterations,
                                desc=f'Epoch [{epoch + 1}/{num_epochs}]', dynamic_ncols=True)

        # Check the model's accuracy (do this for every epoch).
        acc_vals.append(calculate_accuracy(model, test_dataloader))

        # Create a list to hold the current epochs loss values (to average for the tqdm bar).
        curr_loss = []

        # Train the model.
        for i, (inputs, labels) in enumerate(tqdm_data_loader):
            optimizer.zero_grad()  # Clear gradients from previous iteration
            labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            curr_loss.append(loss.item())

            # Update our tqdm loop so we can see what is happening.
            tqdm_data_loader.set_postfix(
                loss=np.mean(curr_loss), acc=acc_vals[-1])

        # Append average of curr_loss to loss_vals
        loss_vals.append(np.mean(curr_loss))

    # After training, you can save the model
    torch.save({'model': model.state_dict(),
                'optim': optimizer.state_dict()}, 'pitch_dnn_L4.pth')

    '''
    Plots

    This is a simple plot of our loss and accuracy. We will use this to see how our model is doing. 
    '''
    # Plot the outputs of our model (the training and the accurary)
    fig, ax = plt.subplots(1, 2, figsize=(8, 6), dpi=100)

    # Plot the loss
    ax[0].plot(np.linspace(0, num_epochs, len(loss_vals)), loss_vals)
    ax[0].set_xlabel('Time Step')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss of the Pitch DNN')

    # Plot the accuracy
    ax[1].plot(np.linspace(0, num_epochs, len(acc_vals)), acc_vals)
    ax[1].set_xlabel('Time Step')
    ax[1].set_ylabel('Accurary (%)')
    ax[1].set_title('Accuracy of the Pitch DNN')

    plt.tight_layout()
    plt.savefig('loss_and_acc4.png')


if __name__ == '__main__':
    main()
