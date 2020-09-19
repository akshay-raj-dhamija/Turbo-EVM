import argparse
import collections
import torch
from EVM import EVMtrainer
from EVM import EVMinference
from .utils.debugger import ForkedPdb


# Check pytorch version
assert float('.'.join(torch.__version__.split('.')[:-1]))>=1.6, "Pytorch version should be at minimum 1.6"

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


def get_all_evm_combinations(args):
    combinations = []
    saver_process_mapping = {}
    saver_process_no = 0
    evm_params = collections.namedtuple('EVM_params', 'tailsize cover_threshold distance_multiplier')
    for tailsize in sorted(args.tailsizes):
        for cover_threshold in args.cover_thresholds:
            for distance_multiplier in args.distance_multipliers:
                params_named_tuple = evm_params(tailsize=tailsize,
                                                cover_threshold=cover_threshold,
                                                distance_multiplier=distance_multiplier)
                combinations.append(params_named_tuple)
                saver_process_mapping[params_named_tuple.__str__()] = args.world_size + saver_process_no
                saver_process_no+=1
    return combinations, saver_process_mapping


def trainer(args):
    EVMtrainer.main(args)

def inference(args):
    EVMinference.main(args)
