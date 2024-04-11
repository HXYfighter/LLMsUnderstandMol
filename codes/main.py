import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import argparse
import datetime

from torch.utils.tensorboard import SummaryWriter

# rdkit
from rdkit import Chem, DataStructs

# guacamol
from guacamol.benchmark_suites import goal_directed_suite_v2

from generator_guacamol import generator_guacamol



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # generation model
    parser.add_argument('--model_type', type=str, default="gpt", choices=["gpt", "rnn"])
    parser.add_argument('--prior_path', type=str, default="ckpt/chembl.pt")
    # task
    parser.add_argument('--task', type=str, default="guacamol", choices=["guacamol", "docking"])
    parser.add_argument('--guacamol_id', type=int, default=0, choices=list(range(20)))
    # RL hyper-parameters
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--sigma', type=float, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=1000)

    parser.add_argument('--vocab_path', type=str, default="data/vocab.txt")
    parser.add_argument('--logger_path', type=str, required=True)
    args = parser.parse_args()
    

    # GuacaMol benchmark
    if args.task == "guacamol":
        args.logger_path = f"{args.guacamol_id}/{args.logger_path}/"
        writer = SummaryWriter(f"log_guacamol/{args.logger_path}/")
        writer.add_text("configs", str(args))

        benchmark = goal_directed_suite_v2()[args.guacamol_id]
        print(f'Task: {args.guacamol_id} - {benchmark.name}')
        generator = generator_guacamol(logger=writer, configs=args)
        result = benchmark.assess_model(generator)
        print(f'Results for the benchmark {result.benchmark_name}:')
        print(f'  Score: {result.score:.6f}')
        print(f'  Execution time: {str(datetime.timedelta(seconds=int(result.execution_time)))}')
        print(f'  Metadata: {result.metadata}')
