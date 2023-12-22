# Deep-learning image analysis for high-throughput screening of opsono-phagocytosis-promoting monoclonal antibodies against Neisseria gonorrhoeae

Scripts for running the analysis described in
<!-- the paper <link_to_paper>. -->

## Environment setup
```
conda create -p ./.env --file ./requirements.txt
```

## Model training
```
RUN="Run01_15_8_tap"
python src/model_training.py -o src/saved_checkpoints/$RUN --reagent all --data src/tensor_dataset --metadata src/metadata/vOPA_tensors_metadata.csv --epochs 15 --batch 8
```

## Store the embeddings
```
MODEL_CHECKPOINT=06Feb23_15_8_all_densenet_1e3
# -- densenet
python src/store_embedding.py --data src/tensor_dataset_pure --metadata src/metadata/vOPA_tensors_metadata.csv --reagent all --folder src/saved_checkpoints/$MODEL_CHECKPOINT --model densenet
```

## Store the vOPA readouts
```
python python src/store_vopa_readout.py --data src/saved_checkpoints/06Feb23_15_8_all_densenet_1e4/onoff_checkpoint_6_0.9974 --metadata src/metadata/vOPA_tensors_metadata.csv
```

## Generate the figures
```
python src/figures.py
```
