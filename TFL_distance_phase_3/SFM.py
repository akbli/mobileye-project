from cmath import sqrt
import numpy as np


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if abs(tZ) < 10e-6:
        print('tz = ', tZ)
    elif 0 == norm_prev_pts.size:
        print('no prev points')
    elif 0 == norm_prev_pts.size:
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []

    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    return (pts-pp)/focal


def unnormalize(pts, focal, pp):
    return pts * focal + pp


def decompose(EM):
    R = EM[:3, :3]
    tZ = EM[2, 3]
    x_foe = EM[0, 3]/tZ
    y_foe = EM[1, 3]/tZ

    foe = np.array([x_foe, y_foe])
    return R, foe, tZ


def rotate(pts, R):
    pts_rotated = []

    for p in pts:
        p = np.append(p, np.array(1))
        result = R.dot(p)
        pts_rotated.append([(result[0] / result[2]), (result[1] / result[2])])

    return np.array(pts_rotated)


def find_corresponding_points(p, norm_pts_rot, foe):
    e_x = foe[0]
    e_y = foe[1]
    m = (e_y - p[1]) / (e_x - p[0])
    n = (p[1] * e_x - e_y * p[0]) / (e_x - p[0])

    def epipolar_line(x):
        return m * x + n

    def distance(x1, x2):
        return abs((x1 - x2) / sqrt(m*m + 1))

    min_index = 0
    min_dist = 2

    for i, pt in enumerate(norm_pts_rot):
        dist = distance(pt[1], epipolar_line(pt[0]))

        if min_dist > dist:
            min_dist = dist
            min_index = i

    return min_index, norm_pts_rot[min_index]


def calc_dist(p_curr, p_rot, foe, tZ):
    dis_x = tZ*(foe[0]-p_rot[0]) / (p_curr[0]-p_rot[0])
    dis_y = tZ*(foe[1]-p_rot[1]) / (p_curr[1]-p_rot[1])

    dX = abs(foe[0] - p_curr[0])
    dY = abs(foe[1] - p_curr[1])

    ratio = dX/(dY + dX)
    return dis_x*ratio + dis_y*(1-ratio)
