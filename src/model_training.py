from pathlib import Path
import os
import copy
import sys
import numpy as np
import pandas as pd
import random
import torch
import torchvision
import matplotlib as mpl
mpl.use("Agg")  # save img on the remote cluster
import matplotlib.pyplot as plt
import seaborn as sns

from data_classes import VOPADataset
from models import Dense161_model, VGG16_model, ResNet18_model
from utils import plot_accuracies, move_nohup_file

from functools import wraps
from time import time
from tqdm import tqdm
from datetime import date

from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='...'
    )

    parser.add_argument(
        '-1',
        '--data',
        type=str,
        help='training set folder',
    )

    parser.add_argument(
        '-2',
        '--metadata',
        type=str,
        help='metadata file'
    )

    parser.add_argument(
        '-3',
        '--epochs',
        type=int,
        default=1,
        help='training epochs'
    )

    parser.add_argument(
        '-4',
        '--batch',
        type=int,
        help='batch size'
    )

    parser.add_argument(
        '-5',
        '--reagent',
        type=str,
        help='either tap, purified or all'
    )

    parser.add_argument(
        '-6',
        '--model',
        type=str,
        help='either densnet(Densnet161) or vgg(VGG16)'
    )

    parser.add_argument(
        '-7',
        '--optimizer',
        type=str,
        help='optimizer sgd, adam, ...'
    )

    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='output folder'
    )

    args = parser.parse_args()
    return args

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        el = time() - ts
        ttime = f'{el//(3600*24):.0f}d {(el//3600)%24:.0f}h {(el//60)%60:.0f}m {el%60:.0f}s'
        print(f'## {f.__name__} elapsed:\t{ttime}')
        return result
    return wrap

def controls_mask_tap(vopa_csv):

    negative_controls = (vopa_csv.treatment == 'NEGATIVE') & \
        (vopa_csv.reagents == 'tap')
    positive_controls = (vopa_csv.treatment == '2C7') & \
        (vopa_csv.reagents == 'tap')

    return negative_controls, positive_controls

def controls_mask_pur(vopa_csv):

    negative_controls = (vopa_csv.treatment == 'NEGATIVE') & \
        (vopa_csv.reagents == 'purified')
    positive_controls = (vopa_csv.treatment == '2C7') & (vopa_csv.treatment_conc >= 5.) \
        (vopa_csv.reagents == 'purified')

    return negative_controls, positive_controls

def controls_mask(vopa_csv, reagent):
    '''
    reagent: 'tap', 'purified', 'all'
    '''
    negative_controls = (vopa_csv.treatment == 'NEGATIVE') & \
        (vopa_csv.reagents == reagent)
    if reagent == 'tap':
        positive_controls = (vopa_csv.treatment == '2C7') & \
            (vopa_csv.reagents == reagent)
    elif reagent == 'purified':
        positive_controls = (vopa_csv.treatment == '2C7') & (vopa_csv.treatment_conc >= 5.) & \
            (vopa_csv.reagents == 'purified')
    elif reagent == 'all':
        p1 = (vopa_csv.treatment == '2C7') & \
            (vopa_csv.reagents == 'tap')
        p2 = (vopa_csv.treatment == '2C7') & (vopa_csv.treatment_conc >= 5.) & \
            (vopa_csv.reagents == 'purified')
        positive_controls = p1+p2
        n1 = (vopa_csv.treatment == 'NEGATIVE') & (vopa_csv.reagents == 'tap')
        n2 = (vopa_csv.treatment == 'NEGATIVE') & (vopa_csv.reagents == 'purified')
        negative_controls = n1 + n2
    else:
        print(f'{reagent} is unknown. Plaese provide a valid reagent')

    return negative_controls, positive_controls

def labelled_dataset(path_data, path_protocols, reagent):

    vopa = VOPADataset(path_data=path_data, path_protocols=path_protocols)
    negative_controls, positive_controls = controls_mask(vopa.csv, reagent)
    df_pos = vopa.csv[positive_controls].copy().reset_index(drop=True)
    df_pos['label'] = [1]*len(df_pos)
    df_neg = vopa.csv[negative_controls].copy().reset_index(drop=True)
    df_neg['label'] = [0]*len(df_neg)
    labelled_csv = pd.concat([df_pos,df_neg]).reset_index(drop=True)
    lab_dataset = VOPADataset(
        path_data=path_data, path_protocols=path_protocols, csv=labelled_csv
    )

    return lab_dataset

def train_split_wells(data_csv=None, verbose=True):

    train_ratio,test_ratio = (0,1)
    while abs(train_ratio-test_ratio) > 5e-2:
        index_list = (data_csv[["experiment", "row", "column"]].drop_duplicates(keep="first").index.tolist())
        train_index = random.sample(index_list, int(0.8 * len(index_list)))
        test_index = list(set(index_list) - set(train_index))
        if verbose:
            train_res = data_csv.iloc[sorted(train_index)].label.value_counts()
            train_ratio = train_res.loc[1] / (train_res.loc[1] + train_res.loc[0])
            test_res = data_csv.iloc[sorted(test_index)].label.value_counts()
            test_ratio = test_res.loc[1] / (test_res.loc[1] + test_res.loc[0])
            print(f"Train/Test ratio difference: {abs(train_ratio-test_ratio)*100:0.3}%")
            print(f"# Train ratio: {train_ratio:0.3}")
            print(f"# Test ratio: {test_ratio:0.3}")

    return train_index, test_index

# for the training index get all the images in the wells
def train_split_images(data_csv):
    train_index,_ = train_split_wells(data_csv=data_csv)
    train_img_index = []
    for ii in train_index:
        tmp_obj = data_csv.iloc[ii]
        exp = data_csv["experiment"] == tmp_obj.experiment
        row = data_csv["row"] == tmp_obj.row
        col = data_csv["column"] == tmp_obj.column
        mask = exp & row & col
        train_img_index.extend(data_csv[mask].index.tolist())

    train_img_index = sorted(train_img_index)
    # Get all the images for the test wells
    test_img_index = sorted(list(set(data_csv.index.tolist()) - set(train_img_index)))
    return train_img_index, test_img_index

@timing
def train_model(model, criterion, optimizer, scheduler, batch, num_epochs=1, out_folder=None):

    # Initialize the Gradient Scaler
    grad_scaler = torch.cuda.amp.GradScaler()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        sseparator = "#" * 10
        print(f"{sseparator}\tEpoch\t{epoch+1}/{num_epochs}")
        # Epoch train and validation phase
        for phase in ["train", "test"]:
            # print("## " + f"{phase}".capitalize())
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_correct = 0
            # Iterate over the Data
            # for inputs, labels in tqdm(dataloaders[phase],mininterval=100):
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.type(torch.float16)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                # Stats
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels)  #labels.data
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct.double() / dataset_sizes[phase]
            print(f"{phase.upper()}\tLoss:\t{epoch_loss:.4f}\tAcc:\t{epoch_acc:.4f}")
            
            # # Write 2 tensorboard
            # try:
            #     writer.add_scalar(f'{phase} loss', epoch_loss, epoch)
            #     writer.add_scalar(f'{phase} accuracy', epoch_acc, epoch)
            # except:
            #     pass

            # Deep copy the model & save checkpoint to file
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                # Save checkpoint
                if out_folder:
                    SAVING_PATH = Path(os.path.abspath(out_folder))
                    # SAVING_PATH = Path(os.getcwd())/ Path("src/saved_checkpoints")/ Path(f"{out_folder}")
                else:
                    SAVING_PATH = Path(os.path.abspath("src/saved_checkpoints"))
                    # SAVING_PATH = Path(os.getcwd()) / Path("src/saved_checkpoints")
                if not os.path.isdir(SAVING_PATH):
                    os.mkdir(SAVING_PATH)
                checkpoint = SAVING_PATH / Path(f"checkpoint_{epoch+1}_{epoch_acc:.4f}.tar")
                
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "batch_size": batch,
                    },
                    checkpoint,
                )
    print(f"Best test Acc: {best_acc:.4f}")
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

if __name__=='__main__':

    args = parse_args()
    path_data = Path(os.path.abspath(args.data))
    path_protocols = Path(os.path.abspath(args.metadata))
    out_folder = Path(os.path.abspath(args.output))
    reagent = args.reagent

    print('Labelled dataset')
    # labelled dataset for supervised training
    vopa = labelled_dataset(path_data, path_protocols, reagent)
    # training/test index splitted by well
    # train_index, test_index = train_split_wells()

    print('Train test split')
    # training/test index for all the images (splitted by well)
    train_img_index, test_img_index = train_split_images(data_csv=vopa.csv)

    print('\tdone')
    # Set the transform to resize the input to (512x512)
    resize = torchvision.transforms.Resize(size=512)

    # Train Dataset
    train_csv = vopa.csv.iloc[train_img_index].copy().reset_index(drop=True)
    vopa_train = VOPADataset(
        path_data=path_data,
        path_protocols=path_protocols,
        csv=train_csv,
        transform=resize,
    )
    # Test Dataset
    test_csv = vopa.csv.iloc[test_img_index].copy().reset_index(drop=True)
    vopa_test = VOPADataset(
        path_data=path_data,
        path_protocols=path_protocols,
        csv=test_csv,
        transform=resize,
    )
    dataset_sizes = {"train": len(vopa_train), "test": len(vopa_test)}

    # Save Dataset for later evaluation
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Tensorboard writer init
    today = date.today().strftime("%d%b%y")
    tb_writer_folder = out_folder.parents[0] / Path(f'runs/{today}_{args.epochs}_{args.batch}_{args.reagent}_{args.model}')
    ii=1
    while os.path.exists(tb_writer_folder):
        tb_writer_folder = Path(str(tb_writer_folder)+f'_{ii}')
        ii+=1
    writer = SummaryWriter(tb_writer_folder)

    # Save the dataset splits
    # torch.save(vopa_train, out_folder / Path("vopa_train.pt"))
    # torch.save(vopa_test, out_folder / Path("vopa_test.pt"))

    # Train/Test dataloaders
    train_loader = torch.utils.data.DataLoader(
        vopa_train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        vopa_test, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True
    )
    dataloaders = {"train": train_loader, "test": test_loader}

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f'Model initialization:\t{args.model}')
    # model, criterion, optimizer, lr scheduler
    if args.model == 'densenet':
        model = Dense161_model()
    elif args.model == 'vgg':
        model = VGG16_model()
    elif args.model == 'resnet':
        model = ResNet18_model()
    else:
        sys.exit('Please provide a valid model: densnet or vgg or resnet')
    
    model = model.to(torch.float16) #model.to(torch.float16)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # WD = 0.1
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.99), weight_decay = WD)
    
    print(f'Optimizer selection:\t{args.optimizer}')
    if args.optimizer == 'sgd':
        # lr = 0.0005
        lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9) # this in now working for both VGG and Densenet
        step_size = max(5, args.epochs//4) 
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif args.optimizer == 'adam':
        step_size = args.epochs -1
        lr = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=1)
    else:
        sys.exit('Please provide a valid optimizer: sgd or adam')

    print('Model training')

    # training
    best_model = train_model(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        args.batch,
        args.epochs,
        out_folder
    )

    # Flush to write everything on the nohup.out file
    sys.stdout.flush()
    # # Move the nohup.out file
    # move_nohup_file(out_folder)
    # # Plot the accuracies
    # plot_accuracies(out_folder, model=args.model)

    # #Plot confusion matrix
    # y_pred = []
    # y_true = []
    # # Set device
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # for inputs, labels in dataloaders['test']:        
    #     inputs = inputs.type(torch.float16).to(device)
    #     labels = labels.type(torch.float16)
    #     with torch.set_grad_enabled(False):
    #         outputs = best_model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #     y_pred.extend(preds.cpu().numpy())
    #     y_true.extend(labels.cpu().numpy().astype(int))

    # classes = ("Positive Control", "Negative Control")
    # cf_matrix = confusion_matrix(y_true, y_pred, normalize="all")
    # df_cm = pd.DataFrame(
    #     np.round(cf_matrix, 3),
    #     index=[i for i in classes],
    #     columns=[i for i in classes],
    # )
    # plt.figure(figsize=(12, 7))
    # sns.heatmap(df_cm, annot=True)
    # plt.title(f"Accuracy: {df_cm.iloc[0,0] + df_cm.iloc[1,1]} %")
    # plt.xlabel("Predicted condition")
    # plt.ylabel("Actual condition")
    # plt.savefig(out_folder /  Path(f"confusion_matrix_best_model.png"))


