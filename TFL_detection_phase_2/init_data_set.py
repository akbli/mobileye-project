import glob
import os
import random

import numpy as np
from PIL import Image

from TFL_attantion_phase_1.run_attention import find_tfl_lights


def load_binary_file(data_dir: str, crop_shape=(81, 81)) -> dict:
    images = np.memmap(data_dir + '\\data.bin', mode='r', dtype=np.uint8).reshape(
        [-1] + list(crop_shape) + [3])

    labels = np.memmap(data_dir + '\\labels.bin', mode='r', dtype=np.uint8)

    return {'images': images, 'labels': labels}


def load_tfl_data(url: str, suffix: str) -> list:
    data_list = []

    for subdir, dirs, files in os.walk(url):
        for directory in dirs:
            data_list += glob.glob(os.path.join(url + '\\' + directory, suffix))

    return data_list


def load_data(dir_set: str) -> dict:
    image_suffix = '*_leftImg8bit.png'
    label_suffix = '*_labelIds.png'

    url_image = "data\\leftImg8bit\\"
    url_label = "data\\gtFine\\"

    return {'images': load_tfl_data(url_image + dir_set, image_suffix),
            'labels': load_tfl_data(url_label + dir_set, label_suffix)}


def crop(image: np.ndarray, candidate, size: int):
    x, y = candidate
    x = int(x)
    y = int(y)
    x += size
    y += size
    return image[x - size:x + size + 1, y - size:y + size + 1]


def get_rand_pixel(pixels):
    rand_p = random.choice(pixels)
    index_rand_p = pixels.index(rand_p)

    return rand_p, index_rand_p


def save_image(dir_name: str, image, label):
    data_root_path = "data/data_set"

    with open(f"{data_root_path}\\{dir_name}\\data.bin", "ab") as data_file:
        np.array(image, dtype=np.uint8).tofile(data_file)

    with open(f"{data_root_path}\\{dir_name}\\labels.bin", "ab") as labels_file:
        np.asarray([label], dtype=np.uint8).tofile(labels_file)


def pad_image(image: np.ndarray, padding_size: int) -> np.ndarray:
    height, width, dim = image.shape

    v_padding = np.zeros((padding_size, width, dim), int)
    h_padding = np.zeros((height + padding_size * 2, padding_size, dim), int)

    image = np.vstack([v_padding, image, v_padding])
    image = np.hstack([h_padding, image, h_padding])

    return image


def insert_data_set(x_coord, y_coord, label, image, dir_name):
    size = 81

    padding_image = pad_image(image, size // 2)

    pixels_of_tfl = [p for p in zip(y_coord, x_coord) if label[int(p[0]), int(p[1])] == 19]
    pixels_of_not_tfl = [p for p in zip(y_coord, x_coord) if label[int(p[0]), int(p[1])] != 19]

    len_tfl = len(pixels_of_tfl)
    len_not_tfl = len(pixels_of_not_tfl)
    for i in range(min(len_tfl, len_not_tfl)):
        rand_tfl_pixel, rand_tfl_index = get_rand_pixel(pixels_of_tfl)
        pixels_of_tfl = pixels_of_tfl[:rand_tfl_index] + pixels_of_tfl[rand_tfl_index + 1:]
        cropped_image = crop(padding_image, rand_tfl_pixel, size // 2)
        save_image(dir_name, cropped_image, 1)

        rand_not_tfl_pixel, rand_not_tfl_index = get_rand_pixel(pixels_of_not_tfl)
        pixels_of_not_tfl = pixels_of_not_tfl[:rand_not_tfl_index] + pixels_of_not_tfl[
                                                                     rand_not_tfl_index + 1:]
        cropped_image = crop(padding_image, rand_not_tfl_pixel, size // 2)
        save_image(dir_name, cropped_image, 0)


def create_data_set(dir_name, t_img_label):
    image = np.array(Image.open(t_img_label[0]))
    label = np.array(Image.open(t_img_label[1]))

    x_red, y_red, x_green, y_green = find_tfl_lights(image)

    x_coord = np.concatenate([x_red, x_green])

    y_coord = np.concatenate([y_red, y_green])
    insert_data_set(x_coord, y_coord, label, image, dir_name)


def main():
    dir_name_t = 'train'
    tfl_data_t = load_data(dir_name_t)

    for item in zip(*tfl_data_t.values()):
        create_data_set(dir_name_t, item)

    dir_name_v = 'val'
    tfl_data_v = load_data(dir_name_v)

    for item in zip(*tfl_data_v.values()):
        create_data_set(dir_name_v, item)

    train_path = "data/data_set/train"
    val_path = "data/data_set/val"

    data_set = {'train': load_binary_file(train_path), 'val': load_binary_file(val_path)}

    for k, v in data_set.items():
        print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape,
                                                   np.mean(v['labels'] == 1) * 100))


if __name__ == '__main__':
    main()
