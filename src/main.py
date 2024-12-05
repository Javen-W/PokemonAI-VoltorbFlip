import os
import logging
from eval_server import EvaluationServer
import argparse


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    args = parser.parse_args()
    print(args.mode)

    # init evaluation server & run
    eval_server = EvaluationServer(mode=args.mode)
    eval_server.run()
