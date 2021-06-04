from util import logger
import threading
from subprocess import call


class NeuralVerifier:
    def __init__(self):
        self.__init_gpt_model()

    def __init_gpt_model(self):
        self.gpt_server_thread = threading.Thread(
            target=lambda: call(["python", "reliability_assessment.gpt_detector.server.py"]))
        self.gpt_server_thread.start()

    def __stop_gpt_model(self):
        self.gpt_server_thread.raise_exception()





class ReliabilityAssessment:
    def __init__(self):
        self.default_logger = logger.get_logger('reliability_assessment')
