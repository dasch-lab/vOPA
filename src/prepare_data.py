import os 
from pathlib import Path
from utils import (
    do_plate_metadata,
    do_long_protocols,
    do_img_metadata,
    save_dataset,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='...'
    )

    parser.add_argument(
        '-1',
        '--img_folder',
        type=str,
        default='src/images',
        help='plate metadata input',
    )

    parser.add_argument(
        '-2',
        '--meta_folder',
        type=str,
        default='src/metadata',
        help='plate metadata output',
    )

    parser.add_argument(
        '-3',
        '--tensor_folder',
        type=str,
        default='src/tensor_dataset',
        help='tensor dataset folder',
    )

    parser.add_argument(
        '-4',
        '--normalize',
        action='store_true',
        help='normalize the dataset computing mean and std'
    )


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img_folder = Path(os.path.abspath(args.img_folder))
    meta_folder = Path(os.path.abspath(args.meta_folder))
    tensor_folder = Path(os.path.abspath(args.tensor_folder))

    normalize = args.normalize

    print('## Do plate metadata')
    # plate metadata
    plates_meta = meta_folder / Path('plates_metadata')
    for plate in os.listdir(img_folder):
        if plate.startswith('.'):
            continue
        do_plate_metadata(img_folder/Path(plate), plates_meta)
    
    print('## Do long protocols')
    # protocols in long format
    raw_protocols = meta_folder / Path('raw_protocols')
    do_long_protocols(raw_protocols, meta_folder)


    # images metadata
    print('## Do images metadata')
    plate_meta = meta_folder / Path('plates_metadata')
    protocols = meta_folder/Path('protocols_long.csv')
    do_img_metadata(plate_meta, protocols, meta_folder)

    print('## Save tensor dataset')
    # save tensor dataset
    image_protocols = meta_folder/Path('vOPA_image_metadata.csv')
    save_dataset(
        path_img = img_folder,
        path_protocols = image_protocols,
        tensor_folder = tensor_folder,
        metadata_folder = meta_folder,
        comp_normalization=normalize,
        save_tensor_csv=True,
    )

    print('#Done , data prepared')

'''
# Usage

python src/prepare_data.py --img_folder src/images --meta_folder src/metadata --tensor_folder src/tensor_dataset --normalize

'''

