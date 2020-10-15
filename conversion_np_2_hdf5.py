import numpy as np
import h5py
# Paths relative to Jarvis
file_names = ["/scratch/mjafarzadeh/umd_extracted_features/feature_train_efficient_b3_center_loss_fp16.npy",
              "/scratch/mjafarzadeh/umd_extracted_features/feature_val_efficient_b3_center_loss_fp16.npy",
              "/scratch/mjafarzadeh/umd_extracted_features/feature_unknown_train_efficient_b3_center_loss_fp16.npy",
              "/scratch/mjafarzadeh/umd_extracted_features/feature_unknown_val_efficient_b3_center_loss_fp16.npy"]
root_output_file_path = "/net/patriot/scratch/adhamija/4mohsen/Features/"
for file_name in file_names:
    data = np.load(file_name,allow_pickle=True)
    classes = set(data[:,0].tolist())
    output_file_path = root_output_file_path+(file_name.split('/')[-1]).split('.')[0]+'.hdf5'
    with h5py.File(output_file_path, "w") as hf:
        for i,c in enumerate(classes):
            print(f"{i}/{len(classes)} {data[data[:,0]==c,1:].shape}")
            g = hf.create_group(str(c))
            g.create_dataset('features', data=data[data[:,0]==c,1:])