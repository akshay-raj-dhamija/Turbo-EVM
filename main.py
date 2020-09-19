import time
import EVM


def main(args):
    if not args.run_tests:
        EVM.trainer(args)
    else:
        EVM.inference(args)

if __name__ == "__main__":
    args = EVM.command_line_options()
    start_time = time.time()
    main(args)
    print(f"Finished all processing in {time.time() - start_time} seconds")