# Turbo-EVM
This is a highly optimized version of EVM, which is able to bring down the running time from a few days to an hour or less.

Please note the optimizations have been performed with the ImageNet dataset in mind, which has `~1300` samples per class.
If your dataset has a different distribution of samples you might run into different kinds of issues resulting in gpu running out of the memory.
In such cases, it is advised to either port that specific part of the computation to the CPU which will increase the running time or 
considering using smaller chunks of data. 
A very dirty implementation of such a case has been provided at `https://github.com/akshay-raj-dhamija/Turbo-EVM/blob/16f17c8a2b21eaa7f069afb14768035978f64d6a/EVM/EVMtrainer.py#L117` 
but it should be noted it will highly slow down the processing.

### Structuring your input features

This code base uses HDF5 files both for excepting input and providing outputs.
You may use the `conversion_np_2_hdf5.py` script to convert an existing numpy feature file into the required HDF5 format.
The numpy file should have the first column indicating the class the sample belongs to followed by the actual features.

### Running Training

Sample command to train an EVM Model with different combinations of hyper parameters.
If more than one parameter is provided a list of possible parameters, EVM's are trained for all possible combinations

```python -m torch.distributed.launch main.py --feature_files /scratch/adhamija/4mohsen/Features/feature_train_efficient_b3_center_loss_fp16.hdf5 --layer_names features --output_path /scratch/adhamija/4mohsen/Features/EVM_Models/DM/ --gpus 10 --tailsizes 33998 --cover_thresholds 0.8 --distance_multipliers 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --distance_metric euclidean --cls_per_chunk 25```

In case of gpu OOM error please try and lower the `cls_per_chunk` parameter.


### Running Testing

The commands for running the testing are very similar to the training commands above but also need two additional parameters `--run_tests` and `--test_feature_files`
Please note the `--output_path` should be the same where the models were saved, this will also be the destination of all the resulting scoring files.

```python -m torch.distributed.launch main.py --feature_files /scratch/adhamija/4mohsen/Features/feature_train_efficient_b3_center_loss_fp16.hdf5 --layer_names features --output_path /scratch/adhamija/4mohsen/Features/EVM_Models/DM/ --gpus 10 --tailsizes 33998 --cover_thresholds 0.8 --distance_multipliers 0.4 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0 --distance_metric euclidean --cls_per_chunk 25 --run_tests --test_feature_files /scratch/adhamija/4mohsen/Features/feature_val_efficient_b3_center_loss_fp16.hdf5```
