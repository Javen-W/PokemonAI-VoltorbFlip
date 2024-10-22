import logging
import platform
import os
import io
import signal
import socket
import subprocess
import numpy as np
from PIL import Image


class EvaluationServer:
    """
    Emulator config constants.
    """
    HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
    PORT = 0  # Port to listen on (non-privileged ports are > 1023)
    EMU_PATH = './emu/BizHawk-2.9.1/'
    N_CLIENTS = 1  # number of concurrent clients to evaluate genomes

    """
    Image processing constants.
    """
    # Edge Detection Kernel
    KERNEL = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    """
    Network packet header constants.
    """
    PNG_HEADER = (b"\x89PNG", 7)
    GAMESTATE_HEADER = (b"STATE", 5)
    READY_STATE = b"5 READY"
    ACK_HEADER = b"ACK"
    SEED_STATE = (b"SEED", 4)
    FINISH_STATE = b"8 FINISHED"
    FITNESS_HEADER = (b"FITNESS:", 8)
    LOG_HEADER = (b"LOG:", 4)

    def __init__(self):
        # default socket timeout
        socket.setdefaulttimeout(300)

        # evaluation vars
        self.client_ps = []  # emulator client process ID(s)
        self.logger = self._init_logger()  # init the logger

        # evaluation mode parameters
        self.SCRIPT_PATH = './src/eval_client.lua'

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
        self.evaluate_client(client)

        # send finish state to client
        client.sendall(self.FINISH_STATE)

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

    def evaluate_client(self, client):
        # wait for client to be ready
        while True:
            data = client.recv(1024)
            if not data:
                raise ConnectionClosedException
            if data == self.READY_STATE:
                self.logger.debug("Client is ready to evaluate next genome.")
                client.sendall(self.READY_STATE)
                break

        # repeat game loop
        finished = False
        while not finished:
            # receive client buffered message
            data = client.recv(8192)

            # client finished sending data
            if not data:
                raise ConnectionClosedException

            # Parse received data into individual message(s) and process
            for msg in self._parse_msgs(data):

                # is msg a fitness score?
                if msg[:self.FITNESS_HEADER[1]] == self.FITNESS_HEADER[0]:
                    self.logger.debug("Client is finished evaluating...")
                    fitness = float(msg[self.FITNESS_HEADER[1]:])
                    finished = True

                # is msg a log?
                elif msg[:self.LOG_HEADER[1]] == self.LOG_HEADER[0]:
                    self.logger.debug(msg[self.LOG_HEADER[1]:])

                # is msg a state screenshot?
                elif msg[:self.PNG_HEADER[1]] == self.PNG_HEADER[0]:
                    self.process_screenshot(msg)
                    # respond to client with acknowledgement
                    self.send_response(client, self.ACK_HEADER)

        # finished evaluating client
        return None

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

    def process_screenshot(self, png: bytes):
        """
        TODO
        """
        self.logger.debug("Processing game screenshot...")
        # read image and convert to grayscale
        img = Image.open(io.BytesIO(png)).convert('L')
        # img.show()
        img = np.array(img)

    @classmethod
    def send_response(cls, client, msg):
        """
        Send response message to client.
        """
        client.sendall(b'' + bytes(f"{len(msg)} {msg}", 'utf-8'))

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
        stream_handler.setLevel(logging.INFO)
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


class ConnectionClosedException(Exception):
    def __init__(self, message="Client connection closed before finishing evaluation."):
        self.message = message
        super().__init__(self.message)
