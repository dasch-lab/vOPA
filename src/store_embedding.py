from pathlib import Path
import os

from utils import (
    timing, 
    save_projection_axes, 
    store_features, 
    store_on_off_features
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='...'
    )

    parser.add_argument(
        '-1',
        '--data',
        type=str,
        help='...'
    )

    parser.add_argument(
        '-2',
        '--metadata',
        type=str,
        help='metadata file'
    )

    parser.add_argument(
        '-3',
        '--reagent',
        type=str,
        help='either tap, purified or all'
    )

    parser.add_argument(
        '-4',
        '--folder',
        type=str,
        help='folder with stored model checkpoints'
    )

    parser.add_argument(
        '-5',
        '--model',
        type=str,
        help='either densnet(Densnet161), vgg(VGG16) or resnet(ResNet18)'
    )

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = parse_args()

    path_data = Path(os.path.abspath(args.data))
    path_protocols = Path(os.path.abspath(args.metadata))
    reagent = args.reagent
    model_folder = Path(os.path.abspath(args.folder))
    model = args.model

    # Do the decoration manually
    store_features = timing(store_features)
    store_on_off_features = timing(store_on_off_features)
    
    # Store model embeddings
    # Redo only if the embedding folder do not exist
    if not any(['embedding' in _ for _ in os.listdir(model_folder)]):
        store_features(model_folder,model)
    else:
        print(f'Embedding alrady present in:\t{model_folder.name}')
    
    # Compute and save SVM on/off projection axis
    best_embeddings = sorted([_ for _ in os.listdir(model_folder) if 'embeddings' in _ and 'OLD' not in _],key=lambda x:float(x.split('_')[-1]))[0]
    path2embeddings = Path(os.path.abspath(model_folder / Path(best_embeddings)))
    save_projection_axes(path2embeddings, path_protocols, reagent, nsample=5e3)
    
    # Store on/off embeddings
    axis_destination = Path(path2embeddings.parent / Path(path2embeddings.name.replace('embeddings','projection_axis')))
    store_on_off_features(axis_destination)

    print('## DONE')



