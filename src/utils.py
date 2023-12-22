import os
import re
import sys
from pathlib import Path
import pandas as pd
from data_classes import VOPADataset,VOPADataset_img
from puzzlecrop import Puzzlecrop
from models import Dense161_model,VGG16_model,ResNet18_model
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

from sklearn.svm import LinearSVC
import numpy as np
from datetime import date

from time import time
from functools import wraps

import matplotlib.pyplot as plt




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

# create plates metadata files
def do_plate_metadata(in_path, out_path):
    """
    Example of file_name: r07c05f16-ch4.png
    in_path = "../../renamed_images/"
    out_path = "../../modified_images/"
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # list files in img directory
    files = sorted([_ for _ in os.listdir(in_path) if _.endswith((".jpg", ".png", ".jpeg", ".tiff"))])
    # files = sorted(files)
    # plate_metadata_header = ["row", "column", "site", "plane", "channel"]
    plate_metadata_header = ["row", "column", "site", "channel"]
    # plate_metadata_regex = "^r(\d+)c(\d+)f(\d+)p(\d+)-ch(\d+)"
    plate_metadata_regex = "^r(\d+)c(\d+)f(\d+)-ch(\d+)"
    p = re.compile(plate_metadata_regex)
    
    plate_metadata = list()
    for file in files:
        row, column, site, channel = p.findall(file)[0]
        # convert the column into letter
        ROW_code = ["A", "B", "C", "D", "E", "F", "G", "H"]
        ROW_name = ["01", "02", "03", "04", "05", "06", "07", "08"]
        row_converter = {_: x for (_, x) in zip(ROW_name, ROW_code)}
        code_row = row_converter[row]
        plate_metadata.append([code_row, int(column), int(site), int(channel)])

    plate_metadata_df = pd.DataFrame(plate_metadata, columns=plate_metadata_header)
    if str(out_path).endswith(".csv"):
        plate_metadata_df.to_csv(out_path, index=False)
    else:
        out_path = os.path.join(
            out_path, str(in_path).split("/")[-1] + "_plate_metadata.csv"
        )
        plate_metadata_df.to_csv(out_path, index=False)

    return None

def tap_purified(experiment):
    mapping = {
        '010722_1' : 'purified',
        '010722_2' : 'purified',
        '010722_3' : 'purified',
        '010722_4' : 'purified',
        '150722_1' : 'purified',
        '150722_2' : 'purified',
        '250422_1' : 'tap',
        '260422_1' : 'tap',
        '260422_2' : 'tap',
        '280422_1' : 'tap',
        '280422_2' : 'tap',
        '310322_1' : 'purified',
        '310322_2' : 'purified',
    }
    
    return mapping[experiment]

# create long protocol format
def do_long_protocols(in_path, out_path):
    
    long_format = list()
    for filename in sorted(os.listdir(in_path)):
        # Skipp hidden files and folders
        if filename.startswith("."):
            continue
        # Skipp all non-.csv files
        if not (filename.endswith(".csv")):
            continue
        # Do recursive for nested folders
        if os.path.isdir(os.path.join(in_path, filename)):
            do_long_protocols(os.path.join(in_path, filename), out_path)
            continue
        ROW_LIST = ["A", "B", "C", "D", "E", "F", "G", "H"]
        COL_LIST = [str(_) for _ in range(1, 13)]
        FILE = filename.split("/")[-1] #TODO redundant - no file will have the path attached
        if FILE.split(".")[-1] == "xlsx":
            df = pd.read_excel(os.path.join(in_path, filename))
        elif FILE.split(".")[-1] == "csv":
            # df = pd.read_csv(os.path.join(in_path, filename), sep=";")
            df = pd.read_csv(os.path.join(in_path, filename), sep=",") # 280422_2.csv
        else:
            print(
                "ERROR: Unknown file extension - {}".format(
                    os.path.join(in_path, filename)
                )
            )
            continue
        experiment = FILE.split(".")[0]
        tap_pure = tap_purified(experiment)
        data = df[COL_LIST]
        long_format_header = [
            "experiment",
            "row",
            "column",
            "treatment",
            "treatment_conc",
            "pathogen",
            "cell_type",
            'MOI',
            'reagents'
        ]
        
        for ii in range(len(ROW_LIST)):
            for jj in range(len(COL_LIST)):
                tmp_data = data.iloc[ii, jj]
                if not (tmp_data != tmp_data):  # TODO correggo
                    if len(tmp_data.split(";")) == 1:
                        treatment = tmp_data.split(";")[0]
                        treatment_conc = None
                    elif len(tmp_data.split(";")) == 2:
                        treatment, treatment_conc = tmp_data.split(";")
                    else:
                        pass
                    
                    pathogen = "FA1090"
                    cell_type = 'THP1'
                    moi = 40
                    # remove the treatment for "control conditions"
                    control_condition_list = [
                        'MOI 20',
                        'MOI 40',
                        'MOI 80',
                        'CELL ONLY',
                        'CELLS ONLY',
                        'BACTERIA ONLY',
                        # 'NEGATIVE', #TODO to include here or not?
                    ]
                    if treatment in control_condition_list:
                        if treatment == 'BACTERIA ONLY':
                            cell_type = None
                        elif treatment.startswith('MOI'):
                            moi = int(re.findall(r'\d+', treatment)[0])
                        else:
                            pass
                        treatment = None
                    sample = [
                        experiment,
                        ROW_LIST[ii],
                        COL_LIST[jj],
                        treatment,
                        treatment_conc,
                        pathogen,
                        cell_type,
                        moi,
                        tap_pure,
                    ]
                    long_format.append(sample)
        print("DONE exp {}".format(os.path.join(in_path, filename)))

    # write the dataframe to the output
    long_format_df = pd.DataFrame(long_format, columns=long_format_header)

    if str(out_path).endswith(".csv"):
        long_format_df.to_csv(out_path, index=False)
    else:
        out_path = os.path.join(out_path, "protocols_long.csv")
        long_format_df.to_csv(out_path, index=False)

    return None

# create vopa image metadata .csv file
def do_img_metadata(
    BASE='src/metadata/plates_metadata', 
    protocols='src/metadata/protocols_long.csv', 
    output='src/metadata'
):
    BASE = Path(os.path.abspath(BASE))
    PROTOCOL_FILE = Path(os.path.abspath(protocols))
    file_list = [_ for _ in os.listdir(BASE) if _.endswith("_plate_metadata.csv")]
    df_protocols = pd.read_csv(PROTOCOL_FILE)
    frames = []
    for tmp_file in sorted(file_list):
        # get the experiment number
        tmp_experiment = "_".join(tmp_file.split('_')[2:4])        
        tmp_metadata = pd.read_csv(BASE / Path(tmp_file))
        tmp_metadata.insert(0, "experiment", [tmp_experiment] * len(tmp_metadata))
        tmp_metadata = pd.merge(tmp_metadata, df_protocols, on=["experiment", "row", "column"], how="inner")
        if len(tmp_metadata) > 0:
            frames.append(tmp_metadata)
            print(tmp_experiment)
        else:
            print("Issue at experiment {}".format(tmp_file))
    vopa_df = pd.concat(frames, axis=0, ignore_index=True)
    if output:
        vopa_df.to_csv(Path(os.path.abspath(output)) / Path("vOPA_image_metadata.csv"), index=False)
    else:
        vopa_df.to_csv(BASE / Path("vOPA_image_metadata.csv"), index=False)

# image normalization, puzzlecrop augmentation and tensor dataset storage
def save_dataset(
    path_img,
    path_protocols,
    comp_normalization=False,
    tensor_folder="src/tensor_dataset",
    metadata_folder="src/metadata",
    save_tensor_csv = True,
):
    '''
    Performing puzzlecrop augmentation
    '''
    tensor_folder = Path(tensor_folder)
    metadata_folder = Path(metadata_folder)
    vopa_img = VOPADataset_img(path_img=path_img, path_protocols=path_protocols)

    # compute statistics for image normalization from the dataset
    if comp_normalization:
        STOP = 1000
        CAP = 10000
        print(f'Started normalizing with {STOP} images')
        
        # check std and mean have no NaN values
        std_data,mean_data = torch.tensor([1, float('nan'), 2]),torch.tensor([1, float('nan'), 2])
        while (torch.isnan(std_data).any() and torch.isnan(mean_data).any()):
            data_list = list()
            idx_list = random.sample(range(len(vopa_img.csv)), CAP)
            ii=0
            while len(data_list) < STOP:
                data, target = vopa_img[idx_list[ii]]
                # print(ii, "--", idx_list[ii], target)
                if target.split("_")[0] == "240222":
                    pass
                    print("# Not loaded: ", target)
                else:
                    data_list.append(data)
                    print(len(data_list), "--", idx_list[ii], target)
                ii+=1
                    
            batch = torch.stack(data_list, dim=0)
            print()
            print(f"# Sample for the normalization: {batch.shape}")
            std_data, mean_data = torch.std_mean(batch.type(torch.float16), [0, 2, 3], keepdim=True)
            print(
                f"Normalize with\n mean: \n{mean_data.ravel().tolist()}\n std: \n{std_data.ravel().tolist()}"
            )

        SIZE = batch.shape[-1] // 2
        transform = transforms.Compose(
            [
                transforms.Normalize(mean_data.ravel().tolist(), std_data.ravel().tolist()),
                Puzzlecrop(size=SIZE, stride=SIZE // 2),
            ]
        )
    # not normalise or use training-set image normalization statistics
    else:
        SIZE = 2160 // 2
        # not normalise
        if False:
            MEAN = [4.8125, 18.625, 1.8955078125, 5.4921875]
            STD = [15.7734375, 26.109375, 8.234375, 6.45703125]
            transform = transforms.Compose([
                transforms.Normalize(MEAN, STD),
                Puzzlecrop(size=SIZE, stride=SIZE // 2),
            ])
        transform = transforms.Compose([
            Puzzlecrop(size=SIZE, stride=SIZE // 2),
        ])

    # Generate the folders to store the tensors
    for exp in sorted(vopa_img.csv.experiment.unique().tolist()):
        out_folder = tensor_folder / Path(exp)
        # Generate the output directory if not exists
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(out_folder)

    for idx, (data, label) in tqdm(enumerate(vopa_img)):
        data = data.type(torch.float16)
        patches = transform(data)
        sv_folder = "_".join(label.split("_")[:-1])

        if False:
            # Create the sv_folder
            if not os.path.exists(tensor_folder / Path(sv_folder)):
                os.makedirs(tensor_folder / Path(sv_folder))

        for id_patch, patch in enumerate(patches):
            name = label + "_{:02d}.pt".format(id_patch + 1)
            saving_path = tensor_folder / Path(sv_folder) / Path(name)
            # uint8 conversion to save storage
            patch = patch.type(torch.uint8)
            torch.save(patch, saving_path)

    # Save the tensors_dataset metadata .csv file
    if save_tensor_csv:
        df_repeated_with_index = (
            vopa_img.csv.drop("channel", axis=1)
            .copy()
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_repeated_with_index = pd.concat([df_repeated_with_index] * 9)
        df_repeated_with_index = df_repeated_with_index.sort_index().reset_index(drop=True)
        new_col = list(range(1, 10)) * (len(df_repeated_with_index) // 9)
        df_repeated_with_index.insert(4, "patch", new_col)
        # save the .csv
        # metadata_folder = Path("/home/dcardamone/vopa/vOPA_tensors_metadata.csv")
        metadata_folder = Path(f"{metadata_folder}/vOPA_tensors_metadata.csv")
        df_repeated_with_index.to_csv(metadata_folder, index=False)

def load_model_checkpoint(
    checkpoint,
    model_name,
):

    # checkpoint = Path(os.path.abspath(checkpoint))
    assert str(checkpoint).endswith('.tar')
    state_dict = torch.load(checkpoint)

    if model_name == 'densenet':
        model = Dense161_model()
    elif model_name == 'vgg':
        model = VGG16_model()
    elif model_name == 'resnet':
        model = ResNet18_model()
    else:
        sys.exit('Please provide a valid model: densnet or vgg or resnet')

    model = model.to(torch.float16)
    model.load_state_dict(state_dict['model_state_dict'])

    return model

def store_features(
    checkpoint,
    model_name,
    path_data='src/tensor_dataset',
    path_protocols='src/metadata/vOPA_tensors_metadata.csv',
    batch=16,
):
    '''
    Stores the embedding from the network last layers.
    checkpoint = src/saved_checkpoint/<...>
    '''
    
    path_data = Path(os.path.abspath(path_data))
    path_protocols = Path(os.path.abspath(path_protocols))
    checkpoint = Path(os.path.abspath(checkpoint))
    
    # best_model = sorted([_ for _ in os.listdir(checkpoint) if 'checkpoint' in _])[-1]
    keysort = lambda x: float(x.split('_')[2].replace('.tar',''))
    best_model = sorted([_ for _ in os.listdir(checkpoint) if 'checkpoint' in _ and _.endswith('.tar')], key=keysort)[-1]

    saving_folder = checkpoint / Path(f"embeddings_{best_model.replace('.tar','')}")
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    best_model = checkpoint / Path(best_model)    
    model = load_model_checkpoint(best_model,model_name)
    
    feature_model = torch.nn.Sequential()
    feature_model.add_module('feature_extractor',model.feature_extractor)
    if isinstance(model.head, nn.Sequential) and len(model.head)>1:
        feature_model.add_module('head',model.head[:-1])
    else:
        # feature_model.add_module('head',model.head)
        pass
    feature_model.eval()
    assert not feature_model.training
    del model

    # Set the resize for the inputs
    resize = transforms.Resize(size=512)

    # tensor dataset
    vopa = VOPADataset(path_data=path_data, path_protocols=path_protocols,transform=resize,)
    vopa_loader = torch.utils.data.DataLoader(vopa, batch_size=batch)

    # set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_model = feature_model.to(device)
    
    print(f"Embedding saving started")
    for exp in sorted(vopa.csv.experiment.unique().tolist()):
        out_folder = saving_folder / Path(exp)
        # Generate the output directory if not exists
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
    
    iii=1
    for inputs, labels in vopa_loader:
        inputs = inputs.type(torch.float16)
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            embedding = feature_model(inputs)
        for ii,el in enumerate(labels):
            # Format the label as the standard saving for tensor data
            el = el.split('_r')[0]+'_r'+el.split('_r')[1].replace('_','',2)
            saving_path = saving_folder / Path(el.split('_r')[0]) / Path(f'{el}.pt')
            # Force the tensor to be on CPU
            torch.save(embedding[ii].cpu(), saving_path)
        iii += 1
        if (iii % 100 == 0):
            print(f"Saved\t{iii}-{vopa.csv.shape[0]//batch}\tfeature embeddings")
            sys.stdout.flush()

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

def labelled_dataset(path_data, path_protocols, reagent, **kwargs):

    vopa = VOPADataset(path_data=path_data, path_protocols=path_protocols, **kwargs)
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

def move_nohup_file(folder):
    folder = Path(os.path.abspath(folder))
    fnohup = folder.parents[0]/Path(folder.name+'_nohup.out')
    if os.path.isfile(fnohup):
        # Move the nohup file
        new_path = folder / fnohup.name
        fnohup.rename(new_path)

def plot_accuracies(path_nohup,model=None):
    if model is None:
        model = 'DensNet' if 'densenet' in str(path_nohup) else None
        model = 'ResNet' if 'resnet' in str(path_nohup) else None
        assert(model is not None)
        
    path_nohup = Path(os.path.abspath(path_nohup))
    nohup = [_ for _ in os.listdir(path_nohup) if _.endswith('nohup.out')][0]
    nohup = Path(path_nohup) / Path(nohup)

    # Read the nohup file
    nohup_text = open(nohup, "r")
    nohup_text = nohup_text.read()

    # epoch_regex = "Epoch\t(.+?)/(.+)"
    train_regex = "TRAIN\tLoss:\t(.{6})\tAcc:\t(.{6})"
    test_regex = "TEST\tLoss:\t(.{6})\tAcc:\t(.{6})"

    # epochs = pd.DataFrame(re.findall(epoch_regex,nohup_text),columns=['epoch','tot_epoch']).astype(int)
    training = pd.DataFrame(re.findall(train_regex,nohup_text),columns=['loss','accuracy']).astype(float)
    test = pd.DataFrame(re.findall(test_regex,nohup_text),columns=['loss','accuracy']).astype(float)

    for metric in list(training):
        plt.figure(figsize=(10,4))
        plt.plot(training[metric], label='Train')
        plt.plot(test[metric], label='Validation')
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.title(f'{model} Controls Classification Loss', pad=13)
        plt.legend(loc='upper right')
        plt.savefig(path_nohup/Path(f'metric_plot_{metric}.png'))

def save_projection_axes(path_data, path_protocols, reagent, nsample=2e3, **kwargs):
    SEED = 123
    print(f'Seed: {SEED}')
    nsample = int(nsample)
    
    controls_dataset = labelled_dataset(path_data, path_protocols, reagent, **kwargs)
    
    pos_csv = controls_dataset.csv[controls_dataset.csv.label==1]
    neg_csv = controls_dataset.csv[controls_dataset.csv.label==0]
    if len(pos_csv) > nsample:
        pos_csv = pos_csv.sample(n=nsample,random_state=SEED).sort_index()
    positive_dataset = labelled_dataset(path_data, path_protocols, reagent, csv=pos_csv)
    if len(neg_csv) > nsample:
        neg_csv = neg_csv.sample(n=nsample,random_state=SEED).sort_index()
    negative_dataset = labelled_dataset(path_data, path_protocols, reagent, csv=neg_csv)

    print(f'Positive Controls:{len(pos_csv)}\nNegative Controls:{len(neg_csv)}')
    pos_dl = torch.utils.data.DataLoader(dataset=positive_dataset, batch_size=positive_dataset.__len__())
    positive_controls, positive_labels = next(iter(pos_dl))

    neg_dl = torch.utils.data.DataLoader(dataset=negative_dataset, batch_size=negative_dataset.__len__())
    negative_controls, negative_labels = next(iter(neg_dl))

    kwargs = {'verbose':1, 'max_iter':int(10e4), 'C':.5, 'tol':1e-4}
    svc = LinearSVC(class_weight="balanced", **kwargs)
    x = torch.cat((negative_controls, positive_controls)).numpy()
    y = torch.cat((negative_labels, positive_labels)).numpy()

    sys.stdout.flush()
    svc.fit(x,y)
    sys.stdout.flush()

    print("## Training linear svm DONE")
    projectionAxis = torch.tensor(svc.coef_, dtype=torch.float16)
    projectionAxis = projectionAxis.view(-1)
    # today = date.today().strftime("%d%b%y")
    axis_destination = path_data.parent / Path(path_data.name.replace('embeddings','projection_axis')+'.pt')
    torch.save(projectionAxis, Path(os.path.abspath(axis_destination)))
    print('# Projection Axis saved')


def get_projections(in_tensor, projectionAxis):
    """
    # Compute on/off perturbation using wiki formulas -- https://en.wikipedia.org/wiki/Vector_projection
    """
    in_tensor = in_tensor.to(torch.float32)
    projectionAxis = projectionAxis.to(torch.float32)
    # projectionAxis = projectionAxis * torch.norm(projectionAxis) ** -1
    OFF_Perturbation = (
        torch.matmul(in_tensor, projectionAxis) * torch.norm(projectionAxis) ** -1
    )
    # ON_Perturbation = OFF_Perturbation * torch.norm(in_tensor - projectionAxis) ** -1
    ON_Perturbation = (torch.norm(projectionAxis) ** -1) * torch.mul(
        OFF_Perturbation, torch.norm(in_tensor - projectionAxis, dim=1)
    )

    onoff_tensor = torch.stack((ON_Perturbation, OFF_Perturbation),1)

    return onoff_tensor


def normalize(xx, MM, mm, meanOff, stdOff):
    """
    Inputs:
        xx columns are [ON, OFF]
        MM = mean On_pos
        mm = mean ON_neg
        meanOff = men(OFF_pos, OFF_neg)
        stdOff = std(OFF_pos, OFF_neg)
    """

    traslation = torch.tensor(((MM + mm) / 2, meanOff))
    scaling = torch.tensor((((MM - mm) / 2) ** -1, (stdOff) ** -1))

    return torch.mul(xx - traslation, scaling)


def store_on_off_features(
    axis_destination,
    path2embeddings=None,
    path_protocols='src/metadata/vOPA_tensors_metadata.csv',
):
    axis_destination = os.path.abspath(axis_destination)
    axis_destination = Path(axis_destination) if axis_destination.endswith(".pt") else Path(axis_destination+'.pt')
    
    path_protocols = Path(os.path.abspath(path_protocols))

    if path2embeddings is None:
        path2embeddings = axis_destination.parent / Path(axis_destination.name.replace('projection_axis','embeddings'))
        path2embeddings = Path(str(path2embeddings).replace('.pt',''))
    else:
        path2embeddings = Path(os.path.abspath(path2embeddings))

    path2onoff = axis_destination.parent / Path(axis_destination.name.replace('projection_axis','onoff'))
    path2onoff = Path(str(path2onoff).replace('.pt',''))
    kwargs = {
        'path_data' : path2embeddings, 'path_protocols' : path_protocols,
    }
    vopa = VOPADataset(**kwargs)
    batch_size = 16
    vopa_loader = torch.utils.data.DataLoader(vopa, batch_size=batch_size)

    # Store the onoff embeddings
    print(f"OnOff features saving started")
    for exp in sorted(vopa.csv.experiment.unique().tolist()):
        out_folder = path2onoff / Path(exp)
        # Generate the output directory if not exists
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
            print(out_folder)

    projectionAxis = torch.load(axis_destination)
    iii=1
    for inputs,labels in vopa_loader:
        onoff_embedding = get_projections(inputs, projectionAxis)
        for ii,el in enumerate(labels):
            el = el.split('_r')[0]+'_r'+el.split('_r')[1].replace('_','',2)
            saving_path = path2onoff / Path(el.split('_r')[0]) / Path(f'{el}.pt')
            # Note the onoff embedding are alredy on CPU
            torch.save(onoff_embedding[ii], saving_path)

        iii+=1
        if (iii%100==0):
            print(f"Saved\t{iii}-{vopa.csv.shape[0]//batch_size}\tfeature embeddings")
            sys.stdout.flush()


