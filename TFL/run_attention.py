try:
    print("Elementary imports: ")
    import os
    import json
    import glob
    import argparse

    print("numpy/scipy imports:")
    import numpy as np
    from scipy import signal as sg
    import scipy.ndimage as ndimage
    from scipy.ndimage.filters import maximum_filter
    from scipy import misc

    print("PIL imports:")
    from PIL import Image

    print("matplotlib imports:")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

except ImportError:
    print("Need to fix the installation")
    raise
print("All imports okay. Yay!")


def get_kernel():
    return np.array([[-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324],
                     [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -2 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, 11 / 324,
                      11 / 324, 11 / 324, 11 / 324, 11 / 324, 11 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -2 / 324, -2 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -2 / 324],
                     [-2 / 324, -2 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324, -1 / 324,
                      -1 / 324, -1 / 324, -2 / 324, -2 / 324],
                     [-2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324, -2 / 324,
                      -2 / 324, -2 / 324, -2 / 324, -2 / 324]])


def get_max_candidates(max_lights_image, image, RGB):
    x_color = []
    y_color = []
    frame_width = 14
    image_height, image_width = max_lights_image.shape[:2]

    for coordX in range(0, image_height - frame_width, frame_width):
        for coordY in range(0, image_width - frame_width, frame_width):
            currImage = max_lights_image[coordX:coordX + frame_width, coordY:coordY + frame_width]
            localMax = np.amax(currImage)
            maxCurr = np.argmax(currImage)

            if localMax > 100:
                image[coordX + maxCurr // frame_width, coordY + maxCurr % frame_width] = RGB
                x_color.append(coordX + maxCurr // frame_width)
                y_color.append(coordY + maxCurr % frame_width)
            currImage[:] = localMax

    return x_color, y_color


def convolution(image, color_image, RGB):
    kernel = get_kernel()
    image_after_con = sg.convolve2d(color_image, kernel, boundary='symm', mode='same')

    max_lights_image = ndimage.maximum_filter(image_after_con, size=5)

    return get_max_candidates(max_lights_image, image, RGB)


def find_tfl_lights(image):
    red_image = image[:, :, 0]
    green_image = image[:, :, 1]
    y_red, x_red = convolution(image, red_image, [255, 0, 0])
    y_green, x_green = convolution(image, green_image, [0, 255, 0])

    return x_red, y_red, x_green, y_green


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    image = np.array(Image.open(image_path))

    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]
    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def show_image_and_gt(image, objects, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objects is not None:
        for o in objects:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'data'
    if args.dir is None:
        args.dir = default_base

    f_list = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))
    for image in f_list:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(f_list):
        print(
            "You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
