import os
import torch
import csv
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader

# Define hyperparameters and model directory path for testing
class HParams:
    def __init__(self):
        self.model_dir = "test_model"
        self.regression = False  # Assuming this is a classification task
        self.callback_list = ['checkpoint', 'csv_log']

class CustomModelCheckpoint(torch.nn.Module):
    def __init__(self, hparams):
            super(CustomModelCheckpoint, self).__init__()
            self.hparams = hparams
            self.best_val_loss = float("inf")
            self.model_dir = hparams.model_dir
            os.makedirs(self.model_dir, exist_ok=True)

    def forward(self, val_loss, model, epoch):
        # Check if the current validation loss is better than the best seen so far
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            '''checkpoint = {
                'class_name': model.__class__.__name__,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
            }'''
            # Save the model's state dictionary with a filename including the epoch and validation loss
            checkpoint_filename = "checkpoint_{:02d}-{:.2f}.pt".format(epoch, val_loss)
            checkpoint_path = os.path.join(self.model_dir, checkpoint_filename)
            #torch.save(model, checkpoint_path)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                torch.save(model.module, checkpoint_path)
            else:
                torch.save(model, checkpoint_path)


class CustomCSVLogger:
    def __init__(self, hparams):
        self.hparams = hparams
        self.log_file_path = os.path.join(hparams.model_dir, "log.csv")
        self.fieldnames = ['epoch', 'val_loss', 'accuracy']  # Add more fields as needed

        # Write header to the CSV file
        with open(self.log_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, epoch, val_loss, accuracy):
        # Append a new row to the CSV log file with the provided epoch, validation loss, and accuracy
        with open(self.log_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy,
                # Add more fields here and provide corresponding values
            })
            
# Updated train_model function to support the test script
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10, device='cuda'):
    # ... (training setup)
    hparams = HParams()
    hparams.callback_list = ['checkpoint', 'csv_log']

    callbacks = make_callbacks(hparams)  # Instantiate the custom callbacks
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate training loss and accuracy for the current epoch
        train_loss = running_loss / len(dataloaders['train'].dataset)
        train_accuracy = correct_predictions / total_samples

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total_correct = 0
            total_samples = 0
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            val_loss /= len(dataloaders['val'].dataset)
            val_accuracy = total_correct / total_samples

        # Save epoch results to CSV log
        if 'csv_log' in hparams.callback_list:
            for callback in callbacks:
                if isinstance(callback, CustomCSVLogger):
                    callback.log(epoch, val_loss, val_accuracy)

        # Save model checkpoint based on the validation loss
        if 'checkpoint' in hparams.callback_list:
            for callback in callbacks:
                if isinstance(callback, CustomModelCheckpoint):
                    callback(val_loss, model, epoch)

        # Print training and validation statistics for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model

def make_callbacks(hparams):
    callbacks = []
    # Check if 'checkpoint' and/or 'csv_log' are in the list of callbacks requested by hparams
    if 'checkpoint' in hparams.callback_list:
        callbacks.append(CustomModelCheckpoint(hparams))
    if 'csv_log' in hparams.callback_list:
        callbacks.append(CustomCSVLogger(hparams))
    return callbacks

# Simple custom model for testing purposes (Replace with your actual model)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x

class SyntheticDataset(data.Dataset):
    def __init__(self, size, num_classes, transform=None):
        self.size = size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        input_data = torch.rand(784)  # Generate random input data with size 784
        target = torch.randint(0, self.num_classes, (1,)).item()  # Random target label
        if self.transform:
            input_data = self.transform(input_data)
        return input_data, target

if __name__ == "__main__":
    # Load synthetic data for testing
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # Adjust the batch_size to match the input size of the model (784)

    # Create synthetic datasets with the required input size (784)
    train_dataset = SyntheticDataset(size=10000, num_classes=10)
    val_dataset = SyntheticDataset(size=2000, num_classes=10)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    dataloaders = {'train': train_loader, 'val': val_loader}

    # Initialize the model, criterion, and optimizer
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Test the training function with synthetic data
    trained_model = train_model(model, dataloaders, criterion, optimizer, num_epochs=5, device='cpu')

    # Print message indicating successful test
    print("Test completed successfully.")