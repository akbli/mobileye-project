import pickle


def load_model():
    return


def load_pkl_file(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file, encoding='latin1')

    focal = data['flx']
    pp = data['principle_point']

    return pp, focal


def load_frames_from_pls(file_name):
    with open(file_name, "r") as pls_file:
        return pls_file.readlines()


class Controller:
    def __init__(self, pls_file_name):
        self.path_frames = load_frames_from_pls(pls_file_name)
        self.tfl_manager = TFLManager(self.path_frames[0][:-1])

    def run(self):
        distance_of_lights = list()
        current_frame = Frame()
        prev_frame = Frame()

        for index, path_frame in enumerate(self.path_frames[1:]):
            current_frame.img_path = path_frame[:-1]
            current_frame.frame_id = index

            current_frame.candidates, current_frame.auxiliary = self.tfl_manager.detect_candidates(current_frame.img_path)
            current_frame.candidates, current_frame.auxiliary = self.tfl_manager.get_tfl_lights(current_frame)

            distance_of_lights = self.tfl_manager.find_dist_of_tfl(current_frame, prev_frame)

            prev_frame = current_frame

        return distance_of_lights


class Frame:
    def __init__(self):
        self.frame_id = 0
        self.img_path = ""
        self.candidates = []
        self.auxiliary = []


class TFLManager:
    def __init__(self, pkl_path):
        principal_point, focal_length = load_pkl_file(pkl_path)
        self.principal_point = principal_point
        self.focal_length = focal_length
        self.EM = []
        self.net = load_model()# todo

    def detect_candidates(self, path_frame):
        candidates = [(928, 157), (516, 623), (1127, 327), (865, 404), (104, 443)]
        auxiliary = ['r', 'r', 'r', 'r', 'g', 'g']
        return candidates, auxiliary

    def get_tfl_lights(self, frame):
        # todo use self.net-neroul net
        return frame.candidates[:3], frame.auxiliary[:3]

    def find_dist_of_tfl(self,  current_frame, prev_frame):
        # todo init self.EM
        # todo: use self.principal_point, self.focal_length to find dist

        return [42.1, 39.5, 46.5]


def main(argv=None):
    controller = Controller('frames.pls')
    print(controller.run())


if __name__ == '__main__':
    main()




