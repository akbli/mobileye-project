import pickle
from PIL import Image
import matplotlib.pyplot as plt


def load_model():
    return


def get_data_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        return pickle.load(pkl_file, encoding='latin1')


class TFLManager:
    def __init__(self, pkl_path):
        self.pkl_data = get_data_pkl(pkl_path)
        self.principal_point = self.pkl_data['principle_point']
        self.focal_length = self.pkl_data['flx']
        self.EM = []
        self.net = load_model()

    def on_frame(self,  prev_frame, current_frame):

        current_frame.light_candidates, current_frame.light_auxiliary = \
            self.detect_candidates(current_frame.frame_path)
        assert len(current_frame.light_candidates) == len(current_frame.light_auxiliary)

        current_frame.tfl_candidates, current_frame.tfl_auxiliary = self.get_tfl_lights(current_frame)

        assert len(current_frame.tfl_candidates) == len(current_frame.tfl_auxiliary)
        assert len(current_frame.tfl_candidates) <= len(current_frame.light_candidates)

        current_frame.tfl_distance = self.find_dist_of_tfl(current_frame, prev_frame)

        self.visualize(prev_frame, current_frame)

        prev_frame = current_frame

        return prev_frame, current_frame

    def get_EM(self, index):
        return self.pkl_data['egomotion_' + str(index) + '-' + str(index + 1)]

    def visualize(self, prev_frame, curr_frame):
        fig, (candidate, traffic_light, dis) = plt.subplots(1, 3, figsize=(12, 6))
        candidate.set_title('candidates')
        candidate.imshow(Image.open(curr_frame.frame_path))
        for i in range(len(curr_frame.light_candidates)):
            candidate.plot(curr_frame.light_candidates[i][0], curr_frame.light_candidates[i][1],
                           curr_frame.light_auxiliary[i] + "+")

        traffic_light.set_title('traffic_lights')
        traffic_light.imshow(Image.open(curr_frame.frame_path))

        for i in range(len(curr_frame.tfl_candidates)):
            traffic_light.plot(curr_frame.tfl_candidates[i][0], curr_frame.tfl_candidates[i][1],
                               curr_frame.tfl_auxiliary[i] + "*")
        dis.set_title('distance')
        dis.imshow(Image.open(curr_frame.frame_path))

        for i in range(len(curr_frame.tfl_distance)):
            dis.text(curr_frame.tfl_candidates[i][0], curr_frame.tfl_candidates[i][1],
                     r'{0:.1f}'.format(curr_frame.tfl_distance[i]), color='y')
        plt.show()

    def detect_candidates(self, path_frame):
        candidates = [(928, 157), (516, 623), (536, 331), (1127, 327), (865, 404), (104, 443)]
        auxiliary = ['r', 'r', 'r', 'r', 'g', 'g']
        return candidates, auxiliary

    def get_tfl_lights(self, frame):
        # todo use self.net-neroul net
        return frame.light_candidates[:3], frame.light_auxiliary[:3]

    def find_dist_of_tfl(self,  current_frame, prev_frame):
        # todo init self.EM
        # todo: use self.principal_point, self.focal_length to find dist

        return [42.1, 39.5, 46.5]
