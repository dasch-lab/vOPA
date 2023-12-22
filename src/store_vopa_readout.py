from pathlib import Path
import os
import pandas as pd
import torch
import copy
from utils import save_projection_axes
from data_classes import VOPADataset
import numpy as np


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

    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()

    data = Path(os.path.abspath(args.data))
    metadata = Path(os.path.abspath(args.metadata))

    # Dataset with on/off embeddings
    onoff_vopa = VOPADataset(path_data=data, path_protocols=metadata)

    # experiment_list = ['250422_1', '280422_1', '280422_2', '150722_1', '150722_2', '310322_1', '310322_2']
    experiment_list = ['250422_1', '280422_1', '280422_2','150722_1']
    # experiment_list = ['150722_1']
    for experiment in experiment_list:
        try:
            experiment_csv = onoff_vopa.csv[onoff_vopa.csv['experiment']==experiment].reset_index(drop=True)

            exp_vopa = VOPADataset(path_data=data, path_protocols=metadata, csv=experiment_csv)
            exp_vopa_loader = torch.utils.data.DataLoader(exp_vopa, batch_size=len(exp_vopa.csv))
            experiment_data, _ = next(iter(exp_vopa_loader)) # get all the data

            experiment_data = copy.copy(exp_vopa.csv.assign(on_pert=experiment_data[:,0], off_pert=experiment_data[:,1]))

            if experiment == '250422_1':
                # experiment_data[experiment_data.treatment == '2C7'].groupby(['row','column']).size()
                mask_2c7 = experiment_data.treatment == '2C7'
                mask_unrelated = experiment_data.treatment == 'NEGATIVE'
                mask_nomab = experiment_data.row.isin(['F','G']) & (experiment_data.column != 11)

                # get only the masked part of the data
                experiment_data = experiment_data[mask_2c7+mask_unrelated+mask_nomab]

                # add the 'sample' column for plot pourpouses
                experiment_data['sample'] = None
                experiment_data.loc[:,'sample'][mask_2c7] = '2C7'
                experiment_data.loc[:,'sample'][mask_unrelated] = 'Unrelated mAb'
                experiment_data.loc[:,'sample'][mask_nomab] = 'No mAb'
                
                aaa = experiment_data.groupby(['row','column','sample'], as_index=False)[['on_pert', 'off_pert']].agg(['max','mean','median'])
                aaa.columns = ["_".join(col_name).rstrip('_') for col_name in aaa.columns]
                aaa.reset_index(inplace=True)

                # Save read-out .xlsx , .csv
                export_excel = copy.copy(aaa)
                export_excel = export_excel.round(5)
                export_excel.to_csv(data.parent / Path(experiment+'_readouts'+'.csv'), index=False)
                export_excel.to_excel(data.parent / Path(experiment+'_readouts'+'.xlsx'), index=False)

            elif experiment == '280422_1':
                
                aaa = experiment_data.groupby(['row','column','treatment'], as_index=False)[['on_pert', 'off_pert']].agg(['max','mean','median'])
                aaa.columns = ["_".join(col_name).rstrip('_') for col_name in aaa.columns]
                aaa.reset_index(inplace=True)

                # Save read-out .xlsx , .csv
                export_excel = copy.copy(aaa)
                export_excel = export_excel.round(5)
                export_excel.to_csv(data.parent / Path(experiment+'_readouts'+'.csv'), index=False)
                export_excel.to_excel(data.parent / Path(experiment+'_readouts'+'.xlsx'), index=False)

            elif experiment == '280422_2':
                bbb = experiment_data.groupby(['row','column','treatment'], as_index=False)[['on_pert', 'off_pert']].agg(['max','mean','median'])
                bbb.columns = ["_".join(col_name).rstrip('_') for col_name in bbb.columns]
                bbb.reset_index(inplace=True)

                # Save read-out .xlsx , .csv
                export_excel = copy.copy(bbb)
                export_excel = export_excel.round(5)
                export_excel.to_csv(data.parent / Path(experiment+'_readouts'+'.csv'), index=False)
                export_excel.to_excel(data.parent / Path(experiment+'_readouts'+'.xlsx'), index=False)

            elif experiment == '150722_1':

                experiment_data = experiment_data[experiment_data['treatment'].notna()]

                # Patch for the column 11 of negative concentration values
                experiment_data.loc[(experiment_data['treatment']=='NEGATIVE'),('treatment_conc')] = experiment_data.loc[(experiment_data['treatment']=='NEGATIVE'),('treatment_conc')].fillna(value=10.)

                aaa = experiment_data.groupby(['row','column','treatment','treatment_conc'], as_index=False)[['on_pert', 'off_pert']].agg(['max','mean','median'])
                aaa.columns = ["_".join(col_name).rstrip('_') for col_name in aaa.columns]
                aaa.reset_index(inplace=True)

                # FIX THE CONTROL CONCENTRATIONs FOR NEGATIVE
                conc_patch = [(('B',9),50.), (('C',9),50.), (('B',10),5.), (('C',10),5.), (('B',11),0.5), (('C',11),0.5)]
                conc_patch += [(('D',9),0.05), (('E',9),0.05), (('D',10),0.005), (('E',10),0.005), (('D',11),0.0005), (('E',11),0.0005)]
                conc_patch += [(('F',9),0.00005), (('G',9),0.00005)]
                conc_patch += [(('B',8),0.00005), (('C',8),0.00005), (('D',8),0.00005), (('E',8),0.00005), (('F',8),0.00005), (('G',8),0.00005),]
                conc_patch += [(('B',11),0.5), (('C',11),0.5), (('D',11),0.0005), (('E',11),0.0005),]

                for ((row,col),conc) in conc_patch:
                    mask = (aaa['row']==row) & (aaa['column']==col)
                    aaa.loc[mask,'treatment_conc'] = conc
                print(f'\tDone concentration patching for protocol: {experiment}')

                aaa['treatment_conc'] = aaa['treatment_conc'].astype(float).map(np.log)
                aaa.sort_values('treatment',inplace=True)

                shift = aaa['treatment_conc'].min()
                if shift < 0:
                    aaa['treatment_conc'] = aaa['treatment_conc'].map(lambda x:x-shift)

                # Save read-out .xlsx , .csv
                export_excel = copy.copy(aaa)
                export_excel['treatment_conc'] = export_excel['treatment_conc'].map(lambda x:x+shift).map(np.exp)
                export_excel = export_excel.round(5)
                export_excel.to_csv(data.parent / Path(experiment+'_readouts'+'.csv'), index=False)
                export_excel.to_excel(data.parent / Path(experiment+'_readouts'+'.xlsx'), index=False, engine='xlsxwriter', float_format="%.5f")

            elif experiment == '':
                pass

        except:
            print(f'Exception for experiment: {experiment}')



'''
# Example command to get the readout for the experiment in the list
python src/store_vopa_readout.py --data src/saved_checkpoints/06Feb23_15_8_all_densenet_1e3/onoff_checkpoint_6_0.9974 --metadata src/metadata/vOPA_tensors_metadata.csv
'''
