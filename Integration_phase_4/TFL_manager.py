import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

from TFL_attantion_phase_1.run_attention import find_tfl_lights
from TFL_detection_phase_2.init_data_set import pad_image, crop
from TFL_distance_phase_3.SFM import calc_TFL_dist
from TFL_distance_phase_3.SFM_standAlone import FrameContainer


def get_data_pkl(pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        return pickle.load(pkl_file, encoding='latin1')


class TFLManager:
    def __init__(self, pkl_path):
        self.pkl_data = get_data_pkl(pkl_path)
        self.principal_point = self.pkl_data['principle_point']
        self.focal_length = self.pkl_data['flx']
        self.EM = []
        self.net = tf.keras.models.load_model("../TFL_detection_phase_2/data/model.h5")

    def on_frame(self, prev_frame, current_frame):
        self.detect_candidates(current_frame)

        assert len(current_frame.light_candidates) == len(current_frame.light_auxiliary)

        current_frame = self.get_tfl_lights(current_frame)
        print(current_frame.frame_id)
        assert len(current_frame.tfl_candidates) == len(current_frame.tfl_auxiliary)
        assert len(current_frame.tfl_candidates) <= len(current_frame.light_candidates)

        if prev_frame.img is not None:
            current_frame.tfl_distance = self.find_dist_of_tfl(current_frame, prev_frame)

        self.visualize(prev_frame, current_frame)

        return current_frame

    def get_EM(self, index):
        return self.pkl_data['egomotion_' + str(index) + '-' + str(index + 1)]

    def detect_candidates(self, frame):
        frame.img = np.array(Image.open(frame.path_frame))
        x_red, y_red, x_green, y_green = find_tfl_lights(frame.img)

        red_len = len(x_red)
        green_len = len(x_green)
        frame.light_candidates = [(x_red[i], y_red[i]) for i in range(red_len)]
        frame.light_auxiliary = ['r'] * red_len
        frame.light_candidates += [(x_green[i], y_green[i]) for i in range(green_len)]

        frame.light_auxiliary += ['g'] * green_len

    def get_tfl_lights(self, frame):
        size = 81
        crop_shape = (size, size)
        padding_img = pad_image(frame.img, size // 2)

        for index, candidate in enumerate(frame.light_candidates):
            x = candidate[0] + size // 2
            y = candidate[1] + size // 2

            cropped_img = crop(padding_img, (y, x), size // 2)
            l_predictions = self.net.predict(cropped_img.reshape([-1] + list(crop_shape) + [3]))

            if l_predictions[0][0] > 0.97:
                frame.tfl_candidates.append(candidate)
                frame.tfl_auxiliary.append(frame.light_auxiliary[index])

        return frame

    def find_dist_of_tfl(self,  current_frame, prev_frame):
        prev_container = FrameContainer(prev_frame.path_frame)
        current_container = FrameContainer(current_frame.path_frame)
        prev_container.traffic_light = prev_frame.tfl_candidates
        current_container.traffic_light = current_frame.tfl_candidates

        current_container.EM = self.pkl_data['egomotion_' + str(current_frame.frame_id - 1) + '-' + str(current_frame.frame_id)]
        current_container = calc_TFL_dist(prev_container, current_container, self.focal_length, self.principal_point)

        return np.array(current_container.traffic_lights_3d_location)

    def visualize(self, prev_frame, curr_frame):
        fig, (candidate, traffic_light, dis) = plt.subplots(1, 3, figsize=(12, 6))
        candidate.set_title('candidates')
        candidate.imshow(Image.open(curr_frame.path_frame))
        for i in range(len(curr_frame.light_candidates)):
            candidate.plot(curr_frame.light_candidates[i][0], curr_frame.light_candidates[i][1],
                           curr_frame.light_auxiliary[i] + '+')
        traffic_light.set_title('traffic_lights')
        traffic_light.imshow(Image.open(curr_frame.path_frame))
        for i in range(len(curr_frame.tfl_candidates)):
            traffic_light.plot(curr_frame.tfl_candidates[i][0], curr_frame.tfl_candidates[i][1],
                               curr_frame.tfl_auxiliary[i] + '*')
        dis.set_title('distance')

        if prev_frame.img is not None:
            dis.imshow(Image.open(curr_frame.path_frame))

            if len(curr_frame.tfl_distance):
                for idx, point in enumerate(curr_frame.tfl_candidates):
                    dis.text(point[0], point[1], r'{0:.1f}'.format(curr_frame.tfl_distance[idx, 2]), color='b')

        plt.show()
