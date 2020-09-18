import argparse
import h5py
import time
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
    parser.add_argument("--tailsizes", nargs="+", help="All EVM tail sizes to process",
                        type=int, default=[33998])
    parser.add_argument("--cover_thresholds", nargs="+", help="All EVM cover thresholds to process",
                        type=float, default=[0.7])
    parser.add_argument("--distance_multipliers", nargs="+", help="All EVM distance multipliers",
                        type=float, default=[0.55])
    parser.add_argument("--output_path", help="output directory path", default="", required=True)
    parser.add_argument("-g", "--gpus", nargs="+", default=[1], type=int, help="number of gpus per node")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--test_feature_files",
                        nargs="+",
                        default=["/scratch/Features/ImageNetPretrained_ResNet50/Val_ImageNet2012_1000.hdf5",
                                 "/scratch/Features/ImageNetPretrained_ResNet50/ImageNet_360.hdf5"],
                        help="HDF5 feature files")
    parser.add_argument("--run_tests", help="Run Tests", action="store_true", default=False)
    parser.add_argument("--debug", help="debugging flag", action="store_true", default=False)
    parser.add_argument("--cls_per_chunk", help="Number of Classes per distance computation chunk", default=100, type=int)
    parser.add_argument('--dist-url', default='localhost', type=str,
                        help='Masters IP address')
    args = parser.parse_args()
    args.world_size = sum(args.gpus)
    return args

def main(args):
    if not args.run_tests:
        EVM.trainer(args)
    else:
        EVM.inference(args)


if __name__ == "__main__":
    args = command_line_options()
    start_time = time.time()
    main(args)
    print(f"Finished all processing in {time.time() - start_time} seconds")