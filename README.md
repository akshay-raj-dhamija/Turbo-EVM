# Turbo-EVM
A highly optimized version of EVM

### Details Coming soon

Sample command to train an EVM Model with different combinations of hyper parameters.
If more than one parameter is provided a list of possible parameters, EVM's are trained for all possible combinations

```python -m torch.distributed.launch main.py --feature_files /scratch/adhamija/4mohsen/Features/feature_train_efficient_b3_center_loss_fp16.hdf5 --layer_names features --output_path /scratch/adhamija/4mohsen/Features/EVM_Models/DM/ --gpus 10 --tailsizes 33998 --cover_thresholds 0.8 --distance_multipliers 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --distance_metric euclidean --cls_per_chunk 25```

Sample command to test an EVM Model.

```python -m torch.distributed.launch main.py --feature_files /scratch/adhamija/4mohsen/Features/feature_train_efficient_b3_center_loss_fp16.hdf5 --layer_names features --output_path /scratch/adhamija/4mohsen/Features/EVM_Models/DM/ --gpus 10 --tailsizes 33998 --cover_thresholds 0.8 --distance_multipliers 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --distance_metric euclidean --cls_per_chunk 25 --run_tests --test_feature_files /scratch/adhamija/4mohsen/Features/feature_val_efficient_b3_center_loss_fp16.hdf5```
