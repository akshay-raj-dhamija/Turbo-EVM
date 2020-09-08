import h5py
from tqdm import tqdm


global hf
global pbar
global count
count = 0

def initializer(args):
    global hf
    global pbar
    output_file_path=f'{args.output_path}/EVM_model_tail_{args.tailsize}_ct_{args.cover_threshold}_dm_{args.distance_multiplier}.hdf5'
    print("SAVING FILE TO",output_file_path)
    hf = h5py.File(output_file_path, "w")
    pbar = tqdm(total=len(args.all_classes))
    return

def close():
    global hf
    global pbar
    hf.close()
    pbar.close()
    return

def save_cls_evs(cls_name, extreme_vectors, extreme_vectors_indexes, covered_vectors):
    global hf
    global pbar
    global count

    g = hf.create_group(cls_name)
    w = g.create_group("weibull")
    w.create_dataset('Scale', data=extreme_vectors['Scale'])
    w.create_dataset('Shape', data=extreme_vectors['Shape'])
    w.create_dataset('signTensor', data=extreme_vectors['signTensor'])
    w.create_dataset('translateAmountTensor', data=extreme_vectors['translateAmountTensor'])
    w.create_dataset('smallScoreTensor', data=extreme_vectors['smallScoreTensor'])
    g.create_dataset('extreme_vectors_indexes', data=extreme_vectors_indexes)
    c = g.create_group("covered_vectors")
    for ind, covered_vector in enumerate(covered_vectors):
        c.create_dataset(str(ind), data=covered_vector)
    pbar.update(1)
    count+=1

def save_multi_cls_evs(to_save):
    print("MULTI")
    for _ in to_save:
        print("_",_)
        cls_name, extreme_vectors, extreme_vectors_indexes, covered_vectors = _
        print("L",cls_name, extreme_vectors, extreme_vectors_indexes, covered_vectors)
        save_cls_evs(cls_name, extreme_vectors, extreme_vectors_indexes, covered_vectors)

def cls_counter():
    return count