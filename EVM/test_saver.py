import h5py
from tqdm import tqdm
import pathlib
global hf
global pbar
global count

def initializer(args,test_file_name, combination_dict):
    global hf
    global pbar
    global count
    count = 0
    output_file_path = pathlib.Path(f"{args.output_path}"
                                    f"/EVM_model_tail_{combination_dict['tailsize']}_ct_{combination_dict['cover_threshold']}_dm_{combination_dict['distance_multiplier']}"
                                    f"/{test_file_name.split('/')[-1]}")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print("SAVING FILE TO",output_file_path)
    hf = h5py.File(output_file_path, "w")
    pbar = tqdm(total=args.total_no_of_classes)
    return

def close():
    global hf
    global pbar
    hf.close()
    pbar.close()
    return

def save_cls_results(cls_name, results):
    global hf
    global pbar
    global count
    g = hf.create_group(cls_name)
    g.create_dataset('probs', data=results)
    pbar.update(1)
    count+=1

def cls_counter():
    return count