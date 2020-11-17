import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description="This script trains an EVM",
)
parser.add_argument("--feature_file_names",
                    nargs="+",
                    default=["/scratch/mjafarzadeh/feature_train_efficientnet_b3_fp16_imagenet.npy",
                             "/scratch/mjafarzadeh/feature_unknown_efficientnet_b3_fp16_imagenet.npy",
                             "/scratch/mjafarzadeh/feature_val_efficientnet_b3_fp16_imagenet.npy"],
                    help="numpy files containing features, the first column should be class label")
parser.add_argument("--output_file_path",
                    default="/net/patriot/scratch/adhamija/SAILON_FEATURES/",
                    help="The directory in which output hdf5 files will be saved")
args = parser.parse_args()

for file_name in args.feature_file_names:
    data = np.load(file_name,allow_pickle=True)
    classes = set(data[:,0].tolist())
    output_file_path = f"{args.output_file_path}/{(file_name.split('/')[-1]).split('.')[0]}.hdf5"
    with h5py.File(output_file_path, "w") as hf:
        for i,c in enumerate(classes):
            print(f"{i}/{len(classes)} {data[data[:,0]==c,1:].shape}")
            g = hf.create_group(str(c))
            g.create_dataset('features', data=data[data[:,0]==c,1:])