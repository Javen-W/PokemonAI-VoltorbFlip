import json
import logging
import platform
import os
import io
import signal
import socket
import subprocess
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from visible_cnn import VoltorbFlipCNN
from hidden_hybrid import HybridModel


class EvaluationServer:
    """
    Emulator config constants.
    """
    HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
    PORT = 0  # Port to listen on (non-privileged ports are > 1023)
    EMU_PATH = './emu/BizHawk-2.9.1/'
    N_CLIENTS = 1  # number of concurrent clients to evaluate genomes
    SCRIPT_PATH = './src/eval_client.lua'

    """
    Image processing constants.
    """
    # Edge Detection Kernel
    KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    TRAINING_PATH = './training_data/'
    IMAGE_DIMS = (256, 384)  # width x height
    CROP_DIMS = (5, 197, 5 + 190, 197 + 187)
    TILE_ORDER = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 3, 4, 5, 6, 7, 8, 9]

    """
    Network packet header constants.
    """
    PNG_HEADER = (b"\x89PNG", 4)
    VISIBLE_STATE_HEADER = (b"VISIBLE_STATE:", 14)
    HIDDEN_STATE_HEADER = (b"HIDDEN_STATE:", 13)
    FITNESS_HEADER = (b"FITNESS:", 8)
    LOG_HEADER = (b"LOG:", 4)
    READY_STATE_HEADER = b"5 READY"

    REQUEST_MODE_HEADER = (b"REQUEST_MODE", 12)
    REQUEST_SEED_HEADER = (b"REQUEST_SEED", 12)

    READY_STATE = "READY"
    SUCCESS_STATE = "SUCCESS"
    FINISH_STATE = "FINISHED"
    EVAL_MODE = "MODE_EVAL"
    TRAIN_MODE = "MODE_TRAIN"

    def __init__(self, mode: str):
        # default socket timeout
        socket.setdefaulttimeout(300)

        # evaluation vars
        self.mode = self.EVAL_MODE
        if mode == "train":
            self.mode = self.TRAIN_MODE

        self.client_ps = []  # emulator client process ID(s)
        self.logger = self._init_logger()  # init the logger
        self.state_index = self.init_state_index(self.TRAINING_PATH)

        # load pretrained models
        self.visible_model, self.hidden_model = None, None
        if self.mode == self.EVAL_MODE:
            self.load_models()

        # emulator path
        if platform.system() == 'Windows':
            self.EMU_PATH = os.path.join(self.EMU_PATH, 'EmuHawk.exe')
        else:
            self.EMU_PATH = os.path.join(self.EMU_PATH, 'EmuHawkMono.sh')

    def run(self):
        # init network server
        server = self.init_server()

        # create client process and record pid
        self.spawn_client()

        # wait for client process to connect to socket
        client, addr = server.accept()
        self.logger.debug(f"Connected by {addr}.")

        # evaluate client(s)
        fitness = self.evaluate_client(client)
        self.logger.info(f"Evaluation fitness: {fitness}")

        # send finish state to client
        self.send_response(client, self.FINISH_STATE)

        # successful generation evaluation
        self.close_server(server)
        return True

    def init_server(self) -> socket.socket:
        # init socket server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # TODO move to init()
        self.logger.debug("Initializing socket server...")

        # bind socket server
        self.PORT = 0  # reset port
        server.bind((self.HOST, self.PORT))
        self.PORT = server.getsockname()[1]

        # listen for incoming connections
        self.logger.debug(f'Socket server: listening on port {self.PORT}')
        server.listen()

        # return server object
        return server

    def load_models(self) -> None:
        # visible model (predicts visual features)
        self.visible_model = VoltorbFlipCNN()
        self.visible_model.load_weights()

        # hidden model (predicts hidden features)
        self.hidden_model = HybridModel(
            tabular_input_size=45,
            image_output_size=64,
            num_classes=4,
        )
        self.hidden_model.load_weights()

    def evaluate_client(self, client):
        # wait for client to be ready
        while True:
            data = client.recv(1024)
            if not data:
                raise ConnectionClosedException
            if data == self.READY_STATE_HEADER:
                self.logger.debug("Client is ready to evaluate.")
                self.send_response(client, self.READY_STATE)
                break

        # repeat game loop
        fitness = None
        while fitness is None:
            # receive client buffered message
            data = client.recv(16000)

            # client finished sending data
            if not data:
                raise ConnectionClosedException

            # parse received data into individual message(s) and process
            for msg in self._parse_msgs(data):

                # is message a log?
                if msg[:self.LOG_HEADER[1]] == self.LOG_HEADER[0]:
                    self.logger.debug(msg[self.LOG_HEADER[1]:])

                # is message a state screenshot?
                elif msg[:self.PNG_HEADER[1]] == self.PNG_HEADER[0]:
                    im = self.process_screenshot(msg)
                    # respond to client with success
                    if self.mode == self.TRAIN_MODE:
                        self.send_response(client, self.SUCCESS_STATE)
                    # make decision
                    if self.mode == self.EVAL_MODE:
                        decision = self.process_decision(im)
                        self.send_response(client, decision)

                # is message a visible state struct?
                elif msg[:self.VISIBLE_STATE_HEADER[1]] == self.VISIBLE_STATE_HEADER[0]:
                    trimmed_msg = msg[self.VISIBLE_STATE_HEADER[1]:]
                    csv_path = os.path.join(self.TRAINING_PATH, f"visible_states.csv")
                    self.process_gamestate(trimmed_msg, csv_path)
                    # respond to client with success
                    self.send_response(client, self.SUCCESS_STATE)

                # is message a hidden state struct?
                elif msg[:self.HIDDEN_STATE_HEADER[1]] == self.HIDDEN_STATE_HEADER[0]:
                    trimmed_msg = msg[self.HIDDEN_STATE_HEADER[1]:]
                    csv_path = os.path.join(self.TRAINING_PATH, f"hidden_states.csv")
                    self.process_gamestate(trimmed_msg, csv_path)
                    # respond to client with success
                    self.send_response(client, self.SUCCESS_STATE)

                # is message a mode request?
                elif msg[:self.REQUEST_MODE_HEADER[1]] == self.REQUEST_MODE_HEADER[0]:
                    # respond to client with evaluation mode
                    self.send_response(client, self.mode)

                # is message a fitness score?
                elif msg[:self.FITNESS_HEADER[1]] == self.FITNESS_HEADER[0]:
                    fitness = int(msg[self.FITNESS_HEADER[1]:])
                    # respond to client with success
                    self.send_response(client, self.SUCCESS_STATE)

        # finished evaluating client
        return fitness

    @classmethod
    def _parse_msgs(cls, _msg) -> [bytes]:
        """
        Parse received client data into individual message(s).
        """
        if not _msg:
            return []
        split = _msg.split(b" ", 1)
        size = int(split[0])
        return [split[1][:size]] + cls._parse_msgs(split[1][size:])

    def process_decision(self, image: np.array) -> str:
        """
        Processes evaluation image state through pre-trained models and prepares a decision map for client.
        """
        image = Image.fromarray(image)
        # predict visible and hidden state features
        visible_hat = self.visible_model.predict(image)
        self.logger.debug(f"state({self.state_index}) visible_hat={visible_hat}")
        scores, hidden_hat = self.hidden_model.predict(visible_hat, image)
        self.logger.debug(f"state({self.state_index}) hidden_hat={hidden_hat}")

        # sort decision scores by tile index
        tile_weights = np.zeros(25)
        for i, tile_idx in enumerate(self.TILE_ORDER):
            score = scores[i]
            tile_val = hidden_hat[i].item()
            if tile_val == 4:
                score = 1.0 - score  # bomb
            tile_weights[tile_idx] = score

        # create decision map for client
        decision_msg = "{ " + ", ".join(["{:.32f}".format(x) for x in tile_weights]) + " }"
        self.logger.debug(f"state({self.state_index}) decision_msg={decision_msg}")
        return decision_msg

    def process_screenshot(self, png: bytes) -> np.array:
        """
        Converts a Bytes format PNG image into a numpy matrix and does basic preprocessing.
        If in training mode, the processed image will also be saved to disk.
        """
        self.logger.debug("Processing game screenshot...")
        # read image and convert to grayscale
        img = PIL.Image.open(io.BytesIO(png)).convert('L')
        img = img.crop(self.CROP_DIMS)
        im = np.array(img)

        # save training image
        if self.mode == self.TRAIN_MODE:
            path = os.path.join(self.TRAINING_PATH, f"screenshots/{self.state_index}.png")
            self.save_img(im, path)

        # advance state index
        self.state_index += 1

        # return image
        return im

    def process_gamestate(self, state: bytes, csv_path: str) -> pd.DataFrame:
        """
        Converts a Bytes format gamestate struct to dataframe form.
        If in training mode, the gamestate dataframe will also be appended to disk.
        """
        self.logger.debug("Evaluating game state...")
        # read and sort input state
        json_state = self.flatten_dict(self.sort_dict(json.loads(state)))
        json_state['state_index'] = self.state_index
        self.logger.debug(json_state)

        # append data frame to CSV file
        df = pd.DataFrame.from_records([json_state]).set_index('state_index')
        if self.mode == self.TRAIN_MODE:
            df.to_csv(csv_path, mode='a', index=True, header=self.state_index == 0)

        # return gamestate dataframe
        return df

    def init_state_index(self, training_path):
        screenshot_path = os.path.join(training_path, "screenshots")
        if os.path.exists(screenshot_path) and self.mode == self.TRAIN_MODE:
            n_images = len(os.listdir(screenshot_path))
            print(f"Existing training images: {n_images}")
            return n_images
        return 0

    @classmethod
    def send_response(cls, client, msg):
        """
        Send response message to client.
        """
        client.sendall(b'' + bytes(f"{len(msg)} {msg}", 'utf-8'))

    @classmethod
    def flatten_dict(cls, x: dict):
        """
        Converts a 2D dictionary into 1D with combined key names.
        """
        output = {}
        for k1 in x:
            for k2 in x[k1]:
                k3 = f"{k1}_{k2}"
                output[k3] = x[k1][k2]
        return output

    @classmethod
    def sort_dict(cls, item: dict):
        """
        Recursively sorts a nested dictionary.
        """
        for k, v in sorted(item.items()):
            item[k] = sorted(v) if isinstance(v, list) else v
        return {k: cls.sort_dict(v) if isinstance(v, dict) else v for k, v in sorted(item.items())}

    @classmethod
    def _init_logger(cls):
        """
        Initializes the EvaluationServer logger.
        Logs training results to logs/trainer.log and debug logs to console.
        """
        logger = logging.getLogger("eval_server")
        logger.setLevel(logging.DEBUG)
        log_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')

        # remove any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()

        # create and add handlers
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_format)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)

        info_handler = logging.FileHandler(f"./logs/eval_server.log")
        info_handler.setFormatter(log_format)
        info_handler.setLevel(logging.DEBUG)
        logger.addHandler(info_handler)

        return logger

    @staticmethod
    def calculate_mindex(data):
        """
        Calculates the message index for the received data.
        """
        return data.find(b" ") + 1

    def spawn_client(self):
        """
        Spawns the emulator process and starts the eval_client.lua script.
        :return: Process object
        """
        self.logger.debug("Spawning emulator client process...")
        ps = subprocess.Popen([
                self.EMU_PATH,
                # f'--chromeless',
                f'--socket_port={self.PORT}',
                f'--socket_ip={self.HOST}',
                f'--lua={os.path.abspath(self.SCRIPT_PATH)}'
            ],
            preexec_fn=os.setsid if platform.system() != 'Windows' else None,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == 'Windows' else None,
        )
        self.client_ps.append(ps)
        return ps

    def kill_client(self, ps):
        """
        Kills the emulator client process group.
        :return: None
        """
        # Send the signal to all the process groups
        if platform.system() != 'Windows':
            os.killpg(os.getpgid(ps.pid), signal.SIGTERM)
        else:
            # ps.send_signal(signal.CTRL_BREAK_EVENT)
            ps.send_signal(signal.SIGTERM)

    def close_server(self, s):
        """
        Forcibly closes the socket server and client process.
        """
        self.logger.debug("Closing server...")
        try:
            for pid in self.client_ps:
                self.kill_client(pid)
            s.shutdown(socket.SHUT_RDWR)
            s.close()
        except Exception:
            # self.logger.error("close_server() failed.")
            return

    def save_img(self, img, path):
        # img = Image.fromarray(img.astype(np.uint8))
        PIL.Image.fromarray(np.uint8(img * 255)).save(path)
        # plt.imsave(path, im, cmap='gray')
        self.logger.debug(f"Saved image: {path}")


class ConnectionClosedException(Exception):
    def __init__(self, message="Client connection closed before finishing evaluation."):
        self.message = message
        super().__init__(self.message)
