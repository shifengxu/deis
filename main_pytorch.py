import argparse
import os
import yaml
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from deis_runner import DeisRunner, get_eps_fn
from deis_vubo_helper import DeisVuboHelper
from schedule.schedule_batch import ScheduleBatch
from utils import dict2namespace, log_info

log_fn = log_info

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--todo", type=str, default='sample_scheduled')
    parser.add_argument("--config", type=str, default='./configs/cifar10.yml')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[7])
    parser.add_argument("--seed", type=int, default=0, help="Random seed. 0 means ignore")
    parser.add_argument("--repeat_times", type=int, default=1, help='run XX times to get avg FID')
    parser.add_argument("--ss_plan_file", type=str,     default="./output1_cifar10/vubo_ss_plan.txt")
    parser.add_argument("--ab_original_dir", type=str,  default='./output1_cifar10/phase1_ab_original')
    parser.add_argument("--ab_scheduled_dir", type=str, default='./output1_cifar10/phase2_ab_scheduled')
    parser.add_argument("--ab_summary_dir", type=str,   default='./output1_cifar10/phase3_ab_summary')
    parser.add_argument("--sample_count", type=int, default=10, help="sample image count")
    parser.add_argument("--sample_batch_size", type=int, default=5, help="0 mean from config file")
    parser.add_argument("--sample_output_dir", type=str, default="./output1_cifar10/generated")
    # parser.add_argument("--sample_ckpt_path", type=str, default="./exp/ema-lsun-bedroom-model-2388000.ckpt")
    # parser.add_argument("--fid_input1", type=str, default="./exp/datasets/lsun/bedroom_train/")
    parser.add_argument("--sample_ckpt_path", type=str, default="./exp/ema-cifar10-model-790000.ckpt")
    parser.add_argument("--fid_input1", type=str, default="cifar10-train")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--noise_schedule", type=str, default="discrete", help="for NoiseScheduleV2")

    parser.add_argument("--ts_phase_arr", nargs='+', type=str, default=['t'])
    parser.add_argument("--ts_order_arr", nargs='+', type=float, default=[1.])
    parser.add_argument("--num_step_arr", nargs='+', type=int, default=[10, 20])
    parser.add_argument("--method_arr", nargs='+', type=str, default=['t_ab'])
    parser.add_argument("--ab_order_arr", nargs='+', type=int, default=[3])
    parser.add_argument("--rk_method_arr", nargs='+', type=str, default=['3kutta'])

    # arguments for schedule_batch
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--lp', type=float, default=0.01, help='learning_portion')
    parser.add_argument('--aa_low', type=float, default=0.0001, help="Alpha Accum lower bound")
    parser.add_argument("--aa_low_lambda", type=float, default=10000000)
    parser.add_argument("--weight_file", type=str, default='./output1_cifar10/res_mse_avg_list.txt')
    parser.add_argument("--weight_power", type=float, default=1.0, help='change the weight value')

    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    gpu_ids = args.gpu_ids
    log_fn(f"gpu_ids : {gpu_ids}")
    args.device = torch.device(f"cuda:{gpu_ids[0]}") if torch.cuda.is_available() and gpu_ids else torch.device("cpu")
    new_config.device = args.device

    # set random seed
    seed = args.seed  # if seed is 0. then ignore it.
    log_fn(f"args.seed : {seed}")
    if seed:
        log_fn(f"  torch.manual_seed({seed})")
        log_fn(f"  np.random.seed({seed})")
        log_fn(f"  random.seed({seed})")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    if seed and torch.cuda.is_available():
        log_fn(f"  torch.cuda.manual_seed({seed})")
        log_fn(f"  torch.cuda.manual_seed_all({seed})")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    log_fn(f"final seed: torch.initial_seed(): {torch.initial_seed()}")
    cudnn.benchmark = True
    return args, new_config

def main():
    args, config = parse_args_and_config()
    log_fn(f"pid : {os.getpid()}")
    log_fn(f"cwd : {os.getcwd()}")
    log_fn(f"args: {args}")
    eps_fn = get_eps_fn(args, config)
    runner = DeisRunner(args, config, eps_fn, ts_phase_arr=args.ts_phase_arr,
                        ts_order_arr=args.ts_order_arr, num_step_arr=args.num_step_arr,
                        method_arr=args.method_arr, ab_order_arr=args.ab_order_arr,
                        rk_method_arr=args.rk_method_arr)
    arr = args.todo.split(',')
    arr = [a.strip() for a in arr]
    for a in arr:
        if a == 'sample_baseline' or a == 'sample_all':
            log_fn(f"{a} ======================================================================")
            runner.sample_baseline()
        elif a == 'alpha_bar_all':
            log_fn(f"{a} ======================================================================")
            runner.alpha_bar_all()
        elif a == 'schedule' or a == 'schedule_only':
            log_fn(f"{a} ======================================================================")
            sb = ScheduleBatch(args)
            sb.schedule_batch()
        elif a == 'sample_scheduled':
            log_fn(f"{a} ======================================================================")
            helper = DeisVuboHelper(args, runner)
            helper.sample_scheduled()
        elif a == 'schedule_sample':
            log_fn(f"{a} ======================================================================")
            helper = DeisVuboHelper(args, runner)
            helper.schedule_sample()
        else:
            raise Exception(f"Invalid todo: {a}")
    # for
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
