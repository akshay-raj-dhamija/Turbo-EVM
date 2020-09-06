import h5py
from tqdm import tqdm


global hf
global pbar


def initializer(args):
    global hf
    global pbar
    pbar = tqdm(total=len(args.all_classes))
    hf = h5py.File(f'{args.output_path}/EVM_model_tail_{args.tailsize}_ct_{args.cover_threshold}_dm_{args.distance_multiplier}.hdf5', "w")
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
