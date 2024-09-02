import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchinfo

def print_model_summary(hparams, model):
    torchinfo.summary(model=model, 
        input_size=(hparams.batch_size, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
    return

def create_fully_connected_model(hparams):
    model = nn.Sequential(
        nn.Flatten(input_shape=hparams.input_shape),
        nn.Linear(hparams.input_shape[1], 200),
        nn.Sigmoid(),
        nn.Linear(200, 60),
        nn.Sigmoid(),
        nn.Linear(60, 10),
        nn.Softmax()
    )
    return model

def create_vgg16_model(hparams):
    """Creates a VGG16 model using transfer learning."""
    print('Creating fixed-feature VGG16 model')
    # Load the pre-trained model from pytorch
    model = models.vgg16(pretrained=True)
    # Set requires_grad to True for model parameters
    # Freeze the weights of the pre-trained model
    # Freeze training for all layers
    for param in model.features.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 8)]) # Add our layer with 8 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier

    print_model_summary(hparams, model)

    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)
    return model, opt

def create_resnet50_model(hparams):
    """Creates a ResNet50 model using transfer learning."""
    # Load the pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze the weights of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.fc.in_features
    clf = list(model.fc.children())[:-1] # Remove last layer
    clf.extend([nn.Linear(num_features, 8)]) # Add our layer with 8 outputs
    model.fc = nn.Sequential(*clf) # Replace the model classifier

    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    print_model_summary(hparams, model) 
    return model, opt
    
def create_mobilenetv2_model(hparams):#, cuda=True):
    """Creates a MobileNetV2 model using transfer learning."""
    # Load the pre-trained model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze the weights of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[1].in_features
    features = list(model.classifier.children())[:-1] # Remove last layer
    features.extend([nn.Linear(num_features, 8)]) # Add our layer with 8 outputs
    model.classifier = nn.Sequential(*features) # Replace the model classifier
    print_model_summary(hparams, model)

    # If cuda is available, move the model to the GPU
    #if cuda:
    #    model = model.cuda()
    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9)

    return model, opt

def create_model(hparams):
    model_type = hparams.model_type.lower()
    print('Creating fixed-feature model:', model_type)
    if model_type == 'fully_connected':
        return create_fully_connected_model(hparams)
    elif model_type == 'vgg16':
        return create_vgg16_model(hparams)
    elif model_type == 'resnet50':
        return create_resnet50_model(hparams)
    elif model_type == 'mobilenetv2':
        return create_mobilenetv2_model(hparams)
    else:
        print('unsupported model type %s' % (model_type))
        return None
    
