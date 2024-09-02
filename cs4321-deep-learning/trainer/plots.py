# 1. Set up the environment:
import os
import torch
import torchvision
import torchinfo
import numpy as np
import seaborn as sns
from matplotlib import rcParams
# autolayout config at runtime
rcParams.update({'figure.autolayout': True})
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from task import setup_data_load
from sklearn.metrics import confusion_matrix, classification_report

# FIXME Modify this for different models
model = sys.argv[1]

RESULTS = {
    'resnet50_fixedfeature':'../best_models/fixedfeature/resnet_checkpoint_29-0.58.pt',
    'vgg16_fixedfeature':'../best_models/fixedfeature/vgg16_checkpoint_39-0.36.pt',
    'mobilenetv2_fixedfeature':'../best_models/fixedfeature/mobilenetv2_checkpoint_39-0.57.pt',

    'resnet50_finetuned':'../best_models/finetune/resnet_checkpoint_21-0.23.pt',
    'vgg16_finetuned': '../best_models/finetune/vgg16_checkpoint_13-0.24.pt',
    'mobilenetv2_finetuned': '../best_models/finetune/mobilenetv2_checkpoint_38-0.36.pt',
}

checkpoint_path = RESULTS[model]
print('Generating plots for:', model)
print('Path:', checkpoint_path)

# model_name = model.__class__.__name__
label_names = {
    0: 'CoastalCliffs',
    1: 'CoastalRocky',
    2: 'CoastalWaterWay',
    3: 'Dunes',
    4: 'ManMadeStructures',
    5: 'SaltMarshes',
    6: 'SandyBeaches',
    7: 'TidalFlats'
}

def print_model_summary(model):
    torchinfo.summary(model=model, 
        input_size=(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    return

def plot_pca(embeddings, labels):
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    plt.figure()

    #plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='rainbow', alpha=0.6, edgecolors='w', linewidth=0.5)
    for i, label in enumerate(set(labels)):
        plt.scatter(pca_result[labels == label, 0], pca_result[labels == label, 1], label=str(label_names[i]), edgecolors='w' )
    
    plt.title('PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    '''
    x_min=-5
    x_max=10
    y_min=-5
    y_max=35

    # Set x-axis limits
    plt.xlim([x_min, x_max])

    # Set y-axis limits
    plt.ylim([y_min, y_max])
    '''
    #x_min=-5
    #x_max=10
    #plt.xlim([x_min, x_max])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.grid(True)
    
    # Save the plot
    save_path = "../plots/"+model+"/pca_plot.png"
    plt.savefig(save_path) #, bbox_inches='tight')
    
    # Optionally display the plot       
    #plt.show()

def plot_tsne(embeddings, labels):
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000) # perplexity=min(60, len(embeddings)-1))
    tsne_result = tsne.fit_transform(embeddings)

    plt.figure()

    #plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='rainbow', alpha=0.6, edgecolors='w', linewidth=0.5)
    for i, label in enumerate(set(labels)):
        plt.scatter(tsne_result[labels == label, 0], tsne_result[labels == label, 1], label=str(label_names[i]), edgecolors='w' )
    plt.title('t-SNE')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    '''
    x_min=-30
    x_max=40
    y_min=-50
    y_max=50

    # Set x-axis limits
    plt.xlim([x_min, x_max])

    # Set y-axis limits
    plt.ylim([y_min, y_max])
    '''
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    # Adjust the layout to make room for the legend
    plt.tight_layout()
    plt.grid(True)
   
    # Save the plot
    save_path = "../plots/"+model+"/tsne_plot.png"
    plt.savefig(save_path)

    # Optionally display the plot     
    #plt.show()

# Do it on raw input data
def plot_raw_input_data(data_loader, writer):
    images, labels = next(iter(data_loader))
    img_grid = torchvision.utils.make_grid(images)
    writer.add_image('Raw Input Data', img_grid)

def plot_confusion_matrix(loaded_model, test_data):
    #cpu = torch.device('cpu')
    dir = '../plots/'+model+'/'
    y_true = []
    y_pred = []
    for inputs, labels in test_data:
        inputs = inputs.cuda()
        outputs = loaded_model(inputs)
        _, pred = torch.max(outputs, 1)
        pred = pred.cpu()
        pred = pred.numpy()
        labels = labels.numpy()
        for i in range(len(pred)):
            y_true.append(labels[i])
            y_pred.append(pred[i])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print('y true:', y_true.shape)
    print('y pred:', y_pred.shape)
    classes = ['CoastalCliffs', 'CoastalRocky', 'CoastalWaterWay', 'Dunes', 'ManMadeStructures', 'SaltMarshes', 'SandyBeaches', 'TidalFlats']
    matrix = confusion_matrix(y_true, y_pred)
    print(classification_report(y_true, y_pred, digits=4))
    rep_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
    heatmap = sns.heatmap(matrix, annot=True, fmt='.2f', cmap=sns.color_palette('Blues', as_cmap=True),
                          xticklabels=classes,
                          yticklabels=classes,)
    figure = heatmap.get_figure()
    class_path = dir + 'classification_matrix.csv'
    print('saved classification matrix to:', class_path, flush=True)
    conf_path = dir + 'confusion_matrix.png'
    print('saved confusion matrix figure to:', conf_path, flush=True)
    try:
        os.makedirs(dir)
    except:
        pass
    #figure.figsize=(8,8)
    figure.savefig(conf_path)
    report = pd.DataFrame(rep_dict).transpose()
    report.to_csv(class_path)

def main():
    # 1. Load the model architecture 
    # 2. Since the entire model was saved at checkpoint can Load the entire model checkpoint back
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/vgg16_2023-08-10_21-47-15/checkpoint_15-0.24.pt'
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/vgg16_2023-08-10_22-12-59/checkpoint_13-0.24.pt'
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/mobilenetv2_2023-08-10_21-34-44/checkpoint_39-0.36.pt'
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/mobilenetv2_2023-08-10_22-42-22/checkpoint_38-0.36.pt'
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/resnet50_2023-08-10_21-25-17/checkpoint_23-0.32.pt'
    #checkpoint_path = '/home/madina.petashvili/team_hp_cs4321_midterm/models/resnet50_2023-08-10_22-33-01/checkpoint_21-0.28.pt'

    #checkpoint_path = '../best_models/finetune/resnet_checkpoint_08-0.33.pt' 

    model = torch.load(checkpoint_path)
    print_model_summary(model)
    # Assuming you've set up your dataloaders and constants correctly in the function `setup_data_load()`
    dataloaders, TRAIN, VAL, TEST = setup_data_load()
 
    plot_confusion_matrix(model, dataloaders[TEST])
    # print(model)
    # 3. Remove the last classifier layer
    #model.module.classifier = torch.nn.Sequential(*list(model.module.classifier.children())[:-7])
    
    # Replace the avgpool layer with an identity operation
    #model.avgpool = torch.nn.Identity()
    # Replace the classifier with an identity operation
    #model.classifier = torch.nn.Identity()

    # Remove the classifier and the avgpool layer to get embeddings
    #model = model.out_features
    # Removing the last fully connected layer to get embeddings
    modules = list(model.children())[:-1]  # all layers except the last one
    model = torch.nn.Sequential(*modules)
    #model = model.module.features
    print(model)
    
    print("Starting model evaluation..")
    model.eval()  # Set the model to evaluation mode

    # 3. Feature extraction
    embeddings_list = []  # Store the embeddings for each batch
    labels_list = []  # Store the labels for each batch
    
    with torch.no_grad():
        for inputs, batch_labels in dataloaders[TEST]:
            inputs = inputs.cuda()     ## for each image input
            output = model(inputs)   ## have an output feature
            
            # Flatten the output to go from a 3D input tensor per image (channels x height x width)
            # to a 2D (batch_size x features) output tensor
            output = output.view(output.size(0), -1)  
            
            embeddings_list.append(output.cpu())
            labels_list.append(batch_labels.cpu())
    
    print("Starting feature extraction..")
    # Concatenate embeddings and labels
    embeddings = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)
    
    embeddings_np = embeddings.detach().numpy()
    labels_np = labels.numpy()

    print("Starting plots..")
    # call the functions on training data pass labels to color-code points based on class
    plot_pca(embeddings_np, labels_np)
    plot_tsne(embeddings_np, labels_np)
    
    print("plots done")


if __name__ == "__main__":
    main()

# For t-SNE and PCA visualization
def plot_embeddings(data_loader, model, writer):
    embeddings = []
    labels = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    all_embeddings = []
    all_labels = []
    all_inputs = []

    for inputs, labels in data_loader:
        with torch.no_grad():
            if inputs.shape[0] == 0:
                continue
            inputs = inputs.to(device)
            outputs = model(inputs)
        #embeddings.append(outputs)
        #labels.extend(labels)
        all_embeddings.append(outputs)
        all_labels.append(labels)
        all_inputs.append(inputs)
    #embeddings = torch.cat(embeddings).cpu().numpy()
    all_embeddings = torch.cat(all_embeddings, 0)
    all_labels = torch.cat(all_labels, 0)
    all_inputs = torch.cat(all_inputs, 0)

    #tsne = TSNE(n_components=8).fit_transform(all_embeddings)
    #pca = PCA(n_components=8).fit_transform(all_embeddings)
    
    tsne = TSNE(n_components=2).fit_transform(all_embeddings.cpu().detach().numpy())
    pca = PCA(n_components=2).fit_transform(all_embeddings.cpu().detach().numpy())

    for i, (x, y) in enumerate(tsne):
        #print(embeddings[i:i+1].shape, inputs[i:i+1].shape)
        writer.add_embedding(all_embeddings, metadata=all_labels, label_img=all_inputs, tag='TSNE')
        
        #writer.add_embedding(embeddings[i:i+1], metadata=[labels[i]], label_img=inputs[i:i+1], global_step=i, tag='TSNE')
    for i, (x, y) in enumerate(pca):
        writer.add_embedding(all_embeddings, metadata=all_labels, label_img=all_inputs, tag='PCA')
        #writer.add_embedding(embeddings[i:i+1], metadata=[labels[i]], label_img=inputs[i:i+1], global_step=i, tag='PCA')
