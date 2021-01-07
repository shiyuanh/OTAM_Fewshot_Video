## Multi-modal Few Shot Video Recognition via Temporal Alignment


#### Training
Training command is as follows:
```
python main.py --modality [MODALITY] --data_dir [DATA_DIR] --fusion_mode [FUSION_MODE]
```
`DATA_DIR` is the directory where you put the RGB and Optical Flow frames. `MODALITY` is one of `RGB`, `Flow`, `Joint` to train models for different modalities. `FUSION_MODE` is only used for Joint modality, it is one of `cat`, `avg`, `mlp` that corresponds to fusion by feautre concatenation, averaging or multi-layer perception learning. The model weights would be saved under `logs/[MODALITY]_[NUM_SEGMENTS]_[FUSION_MODE]`

#### Testing
Testing commad is as follows:
```
python test.py --modality [MODALITY] --data_dir [DATA_DIR] --fusion_mode [FUSION_MODE]
```
where it will output the mean and std for the classficiation accuracies of 600 randomly selected tasks.
