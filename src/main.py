import os
import logging
from eval_server import EvaluationServer


if __name__ == "__main__":
    # init evaluation server & run
    eval_server = EvaluationServer()
    eval_server.run()
