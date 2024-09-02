#%%
from __future__ import print_function, division
import os
import datetime
from datetime import datetime
import torch
import torchvision.models as models
from torchvision import datasets, transforms

import params
import models_tuned, models_fixed
from callbacks import make_callbacks, CustomCSVLogger, CustomModelCheckpoint

# %%
def print_system_info():
    """Prints information about the system's CPU and GPU resources."""
    CPUs = os.cpu_count()
    GPUs = torch.cuda.device_count()
    print(f"PyTorch Version: {torch.__version__}", flush=True)
    print(f"GPUs available: {GPUs}", flush=True)
    print(f"CPUs available: {CPUs}", flush=True)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA", flush=True)

def get_rank(default_rank=0):
    try:
        rank = int(os.environ['RANK'])
    except KeyError:
        rank = default_rank
    return rank

def setup_DDP(hparams):
    rank = get_rank()
    print("Rank:", rank, flush=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=hparams.world_size, rank=rank)

def cleanup():
    torch.distributed.destroy_process_group()

def setup_data_load():
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'

    # VGG-16, Mobilenetv2 and ResNet50 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        VAL: transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
        ])
    }

    data_dir = os.path.join('/data', 'cs4321', 'HW1')

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=8,
            shuffle=True, num_workers=4
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

    for x in [TRAIN, VAL, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x), flush=True)
        
    print("Created all datasets", flush=True)

    class_names = image_datasets[TRAIN].classes
    print("Classes: ", flush=True)
    print(class_names, flush=True)

    return dataloaders, TRAIN, VAL, TEST


# %%
def main():
    # Begin timer for how long it takes to run through the pipeline
    start = datetime.now()

    # Print system initializer info to verify GPUs
    print_system_info()

    # Get the parameters and save them for future reference
    hparams = params.get_hparams()
    if not os.path.exists(hparams.model_dir):
        # make directories recursively to store the model
        os.mkdir(hparams.model_dir)
    params.save_hparams(hparams)

    # Set up the dataloaders and the transformation of data
    dataloaders, TRAIN, VAL, TEST = setup_data_load()

    # Set up distributed parallel computing
    #setup_DDP(hparams) ## didn't work by calling the function so do it manually
    rank = get_rank()
    print("Rank:", rank, flush=True)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=hparams.world_size, rank=rank)
    
    # Load or build and compile the model
    if hparams.fixed==True:
        model, opt = models_fixed.create_model(hparams)
    elif hparams.fixed==False:
        model, opt = models_tuned.create_model(hparams)


    # If cuda is available, move the model to the GPU
    #if cuda:
    #    model = model.cuda()
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = opt  # <--- here optimizer is passed from the models.create_model bc we pass the weight with it

    callbacks = make_callbacks(hparams)  # Instantiate the custom callbacks
    
    # Training loop
    for epoch in range(hparams.num_epochs):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        for inputs, labels in dataloaders[TRAIN]:
            optimizer.zero_grad()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            outputs = outputs.cuda()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate training loss and accuracy for the current epoch
        train_loss /= len(dataloaders[TRAIN].dataset)
        train_accuracy = correct_predictions / total_samples
        
        #writer.add_scalar('Training Loss', train_loss, epoch)
        #writer.add_scalar('Training Accuracy', 100. * train_accuracy, epoch)

        # Validation phase
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total_correct = 0
            total_samples = 0
            for inputs, labels in dataloaders[VAL]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            val_loss /= len(dataloaders[VAL].dataset)
            val_accuracy = total_correct / total_samples
        
            #writer.add_scalar('Validation Loss', val_loss, epoch)
            #writer.add_scalar('Validation Accuracy', 100. * val_accuracy, epoch)

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
        print(f"Epoch [{epoch+1}/{hparams.num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}", flush=True)

        # After all epochs are completed, evaluate the model on the test set
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            total_correct = 0
            total_samples = 0
            for inputs, labels in dataloaders[TEST]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            test_loss /= len(dataloaders[TEST].dataset)
            test_accuracy = total_correct / total_samples

            #writer.add_scalar('Testing Loss', test_loss, epoch)
            #writer.add_scalar('Testing Accuracy', 100. * test_accuracy, epoch)

        # Print test statistics
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}", flush=True)
   
    # clean up after using DDP
    cleanup()

    # End timer, calculate how long it took to run the pipeline and print the details
    end = datetime.now()
    total = end - start

    print('Start time:', start)
    print('End time:', end)
    print('Total time:', total)


#%%
if __name__ == "__main__":
    main()

# %% ## if using the dataset class then do the below 
    '''
    TRAIN = 'train'
    VAL = 'validation'
    TEST = 'test'

    # VGG-16 Takes 224x224 images as input, so we resize all of them
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        VAL: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    data_dir = os.path.join('/data', 'cs4321', 'HW1')

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), 
            transform=data_transforms[x]
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=8,
            shuffle=True, num_workers=4
        )
        for x in [TRAIN, VAL, TEST]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}

    for x in [TRAIN, VAL, TEST]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x), flush=True)
        
    print("Created all datasets", flush=True)

    class_names = image_datasets[TRAIN].classes
    print("Classes: ", flush=True)
    print(class_names, flush=True)

    '''

    '''
    # import data
    train_dataset = CoastalDataset(hparams.train_dir, transform=data_transforms[TRAIN])
    val_dataset = CoastalDataset(hparams.val_dir, transform=data_transforms[VAL])
    test_dataset = CoastalDataset(hparams.test_dir, transform=data_transforms[TEST])
    print("Created all datasets", flush=True)

    # Assuming dataset consists of a list of image-label pairs in the train_dataset
    # train_dataset = [(image1, label1), (image2, label2), ...]

    # Extract labels from the train_dataset and convert them to integers
    y_train = [label for image, label in train_dataset]
    x_train = [image for image, label in train_dataset]
    # Now y_train should be a list of integers representing the class labels
    # e.g., y_train = [0, 1, 2, 0, 1, ...]

    # Now, can apply one-hot encoding
    y_train = torch.nn.functional.one_hot(torch.tensor(y_train), num_classes=len(train_dataset.classes))

    # Extract labels from test_dataset
    y_test = [label for image, label in test_dataset]
    x_test = [image for image, label in test_dataset]
    # make one_hot_encoded
    y_test = torch.nn.functional.one_hot(torch.tensor(y_test), num_classes=len(test_dataset.classes))
        
    # Extract labels from validation_dataset
    y_val = [label for image, label in val_dataset]
    x_val = [image for image, label in val_dataset]
    # make one_hot_encoded
    y_val = torch.nn.functional.one_hot(torch.tensor(y_val), num_classes=len(val_dataset.classes))

    # Lets look at few data samples
    # Convert to NumPy arrays

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Print shapes
    print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape), flush=True)
    print('Val: X=%s, y=%s' % (x_val.shape, y_val.shape), flush=True)
    print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape), flush=True)

    # Convert to PyTorch tensors
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_val = np.squeeze(x_val)
    y_val = np.squeeze(y_val)
    x_test = np.squeeze(x_test)
    y_test = np.squeeze(y_test)

    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    x_val = torch.tensor(x_val)
    y_val = torch.tensor(y_val)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    '''
    ## if using tensorboard then do below
    '''
    print('tensorboard vis', flush=True)
    # Tensorboard writer
    try:
        os.mkdir('runs/test')
    except FileExistsError:
        pass
    
    # View Data
    writer = SummaryWriter('runs/test') 
    dataiter = iter(image_datasets[TRAIN])
    images, labels = next(dataiter)
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('sample images', img_grid)
    '''
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    images = images.to(device)
    
    print(images.shape)

    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_to_trace = model.module
    else:
        model_to_trace = model
        model_to_trace.eval()


    # Add the Model to TensorBoard:
    writer.add_graph(model_to_trace, images)    
    #writer.add_graph(model, images)

    # Prepare Image and Label Lists:
    #images, labels = image_datasets[TRAIN].data, image_datasets[TRAIN].targets
    images = [image for image, _ in image_datasets[TRAIN]] 
    labels = [label for _, label in image_datasets[TRAIN]] 

    # Print Sample Images:
    print('images', flush=True)
    for i in range(2):
        print(images[i], flush=True)
    # Print Class Labels:
    class_names = image_datasets[TRAIN].classes
    class_labels = [class_names[lab] for lab in labels]
    print('class labels for train set', flush=True)
    print(class_labels, flush=True)

    # Visualize Embeddings:  
    print('Plotting embeddings...', flush=True)
    plot_embeddings(dataloaders[TRAIN], model, writer)
    print('...done.')

    # Visualize Raw Input Data:
    print('Plotting raw data...', flush=True)
    plot_raw_input_data(dataloaders[TRAIN], writer)
    print('...done.')
        
    # Close tensorboard writer
    writer.close()
    '''