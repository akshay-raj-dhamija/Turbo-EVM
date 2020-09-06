import argparse
import h5py
import time
import torch
import torch.multiprocessing as mp
import torch.distributed.rpc as rpc
import EVM

def command_line_options():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script trains an EVM",
    )
    parser.add_argument("--feature_files",
                        nargs="+",
                        default=["/scratch/Features/ImageNetPretrained_ResNet50/ARD_TESTING_Train_ImageNet2012_1000.hdf5"],
                        help="HDF5 feature files")
    parser.add_argument("--layer_names",
                        nargs="+",
                        help="Layer names to train EVM on",
                        default=["avgpool"])
    parser.add_argument("--tailsize", help="EVM tail size", type=int, default=33998)
    parser.add_argument("--cover_threshold", help="EVM cover threshold", type=float, default=0.7)
    parser.add_argument("--distance_multiplier", help="distance multiplier", type=float, default=0.55)
    parser.add_argument("--output_path", help="output directory path", default="", required=True)
    parser.add_argument("-g", "--gpus", nargs="+", default=[1], type=int, help="number of gpus per node")
    parser.add_argument("-w", "--world-size", default=1, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--debug", help="debugging flag", action="store_true", default=False)
    parser.add_argument('--dist-url', default='env://', type=str, #tcp://128.198.147.47:7979
                        help='url used to set up distributed training')
    args = parser.parse_args()
    args.world_size = sum(args.gpus)
    return args

def main(args):
    h5_objs = [h5py.File(file_name, "r") for file_name in args.feature_files]
    file_layer_comb = list(zip(h5_objs, args.layer_names))
    args.all_classes = []
    for cls in h5_objs[0].keys():
        temp=[None,0]
        for hf, layer_name in file_layer_comb:
            if temp[0] is None:
                temp[0] = hf[cls][layer_name].shape[0]
            temp[1] += hf[cls][layer_name].shape[1]
        args.all_classes.append((cls,temp))
    for h in h5_objs: h.close()

    if args.debug:
        args.all_classes = args.all_classes[: args.world_size * 3]

    to_save_queue = mp.Queue()
    if args.world_size == 1:
        EVM.trainer(0, args, to_save_queue)
    else:
        processes = []
        for rank in range(args.gpus[args.local_rank]):
            p = mp.Process(target=EVM.trainer, args=(rank, args, to_save_queue))
            p.start()
            processes.append(p)

    EVM.model_saver.initializer(args)
    for _ in range(len(args.all_classes)):
        try:
            to_save = to_save_queue.get()
        except:
            to_save = None
        if to_save is not None:
            current_cls_being_processed, extreme_vectors, extreme_vectors_indexes, covered_vectors = to_save
            EVM.model_saver.save_cls_evs(current_cls_being_processed, extreme_vectors, extreme_vectors_indexes, covered_vectors)

    EVM.model_saver.close()

    if args.world_size > 1:
        for p in processes:
            p.join()


if __name__ == "__main__":
    args = command_line_options()
    start_time = time.time()
    main(args)
    print(f"Finished Training in {time.time() - start_time} seconds")
