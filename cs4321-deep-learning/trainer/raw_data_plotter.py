import torch
import numpy as np
import torchvision.transforms as transforms
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from data_class import CoastalDataset
from joblib import dump, load

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

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Assuming dataset consists of a list of image-label pairs in the train_dataset
# train_dataset = [(image1, label1), (image2, label2), ...]
start = datetime.now()
train_dir = "/data/cs4321/HW1/train"
train_dataset = CoastalDataset(train_dir, transform=transform)

# 1. Load your dataset using dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=500,
                                          shuffle=True, num_workers=2)
print("Loaded the training dataset", flush=True)

# 2. Convert image data to a format suitable for t-SNE (e.g., N x D array)
data_list = []
labels_list = []
for data in trainloader:
    images, labels = data
    data_list.append(images.view(images.size(0), -1).numpy())
    labels_list.append(labels.numpy())


#print("data list", data_list)
#print("labels list", labels_list)

data_np = np.vstack(data_list)
labels_np = np.hstack(labels_list)

# 3. Apply t-SNE
print("applying t-SNE..")
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(data_np)
dump(tsne_results, 'tsne_results.joblib')
#tsne_results= load('tsne_results.joblib')
print("got t-SNE plot results")
print(tsne_results)

# 4. Plot the results
print("plotting t-SNE results")
#for i, label in enumerate(set(labels)):
#    scatter = plt.scatter(tsne_results[labels == label, 0], tsne_results[labels == label, 1], label=str(label_names[i]), edgecolors='w' )

# Create a custom legend with label names
#handles, labels = scatter.legend_elements()
# Extract the integer value from the formatted string labels
#new_labels = [label_names[int(label.get_text().replace('$\\mathdefault{', '').replace('}$', ''))] for label in handles]

#plt.legend(handles, new_labels, title="Classes")

'''
#legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("t-SNE of Raw Image Features")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
# Adjust the layout to make room for the legend
plt.tight_layout()
plt.grid(True)
#plt.colorbar()
#scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_np, cmap='rainbow', alpha=0.6, edgecolors='w', linewidth=0.5)
'''
#fig, ax = plt.subplots()
for i in range(len(tsne_results)):
    point = tsne_results[i]
    label = labels_np[i]
    #print('point:', point, ' - label:', label, 'label_name: ', label_names[label])
    plt.scatter(point[0], point[1], c=label, cmap='rainbow', norm='linear', alpha=0.6, edgecolors='w', linewidth=0.5, label=label_names[label])
#legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
#plt.title("t-SNE of Raw Image Features")
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
# Adjust the layout to make room for the legend
plt.grid(True)
print("t-SNE results plotted")

# 5. Save plot to a file
print("saving the t-SNE plot...")
save_path = "../team_hp_cs4321_midterm/plots/raw/test/tsne_plot.png"
plt.savefig(save_path, bbox_inches='tight')

print("t-SNE plot saved!")

end = datetime.now()
total = end - start
print('Start time:', start)
print('End time:', end)
print('Total time:', total)