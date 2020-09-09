from TFL_manager import TFLManager
from frame import Frame


def load_play_list(file_name):
    with open(file_name, "r") as pls_file:
        return pls_file.readlines()


class Controller:
    def __init__(self, pls_file_name):
        self.play_list = load_play_list(pls_file_name)
        self.tfl_manager = TFLManager(self.play_list[0][:-1])

    def run(self):
        current_frame = Frame()
        prev_frame = Frame()

        for index, path_frame in enumerate(self.play_list[1:]):
            current_frame.frame_path = path_frame[:-1]
            current_frame.frame_id = index

            prev_frame, current_frame = self.tfl_manager.on_frame(prev_frame, current_frame)

        return current_frame.tfl_distance
