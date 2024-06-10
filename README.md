# paper-augmentation
## Repository Structure
- nnunet: Contains modified / additional files for the nnunet segmentation model. Pull nnunet from https://github.com/MIC-DKFZ/nnUNet (guaranteed to work with version 2.3.1), newer versions might work but may also break things. Place the files in the corresponding nnunet subdirectories.
- scripts: Contains scripts for statistical analysis of computed torsion angles and dice coefficients on tes data, as well as python scripts for torsional alignment post-processing.
- torsion: Contains all relevant code necessary for torsional alignment post-processing.

## nnunet modifications
- nnunet/training/data_augmentation/custom_transforms/motion_augments contains all MRI-specific augmentations, in batchgenerators-like format so they can seamlessly integrate into nnunet.
- nnunet/training/nnUNetTrainerCustom contains a custom trainer class that incorporates the different augmentation schemes. 
- nnunet/training/nnUNetTrainer is the nnUNetTrainer base class and has been modified to allow incorportation of the augmentation customisations. Now allows additional arguments to set augmentation type and number of epochs to train.
- nnunet/run/run_training is the training entry point and has been modified with additional arguments to specific augmentation scheme and number of epochs to train.

## Data
A dataset containing all data used in the study is available at https://huggingface.co/datasets/westfechtel/augmentation-paper