import cv2
import numpy as np


path = 'result/pixel_wise/'


def l1_distance(x, y):
    return abs(x-y)


def l2_distance(x, y):
    return (x-y)**2


def pixel_wise_matching(left_img, right_img, disparity_range, distance_method, save_result=True):
    # read left, right images then convert to grayscale
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    # convert to float32
    left = left.astype(np.float32)
    right = right.astype(np.float32)

    # initial depth matrix
    h, w = left.shape[:2]
    if distance_method == l1_distance:
        max_value = 255
    elif distance_method == l2_distance:
        max_value = 255 ** 2
    else:
        max_value = 0
    scale = 16

    # calculate cost using broadcasting
    cost = np.full((h, w, disparity_range), max_value, dtype=np.float32)
    for d in range(disparity_range):
        left_d = left[:, d:]
        right_d = right[:, 0:w-d]
        # using distance_method: l1, l2
        cost[:, d:, d] = distance_method(left_d, right_d)

    min_cost = np.argmin(cost, axis=2)
    depth = min_cost * scale
    depth = depth.astype(np.uint8)

    if save_result == True:
        print('Saving result...')
        # Save results
        end_name = "l1" if distance_method == l1_distance else "l2"

        cv2.imwrite(f'{path}pixel_wise_{end_name}.png', depth)
        cv2.imwrite(f'{path}pixel_wise_{end_name}_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth


left_img_path = 'assets/tsukuba/left.png'
right_img_path = 'assets/tsukuba/right.png'
disparity_range = 16

pixel_wise_result_l1 = pixel_wise_matching(left_img_path,
                                           right_img_path,
                                           disparity_range,
                                           l1_distance,
                                           save_result=True)

pixel_wise_result_l2 = pixel_wise_matching(left_img_path,
                                           right_img_path,
                                           disparity_range,
                                           l2_distance,
                                           save_result=True)
