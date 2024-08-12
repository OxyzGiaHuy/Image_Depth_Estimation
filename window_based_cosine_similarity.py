import cv2
import numpy as np

path = 'result/window_based/'


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def window_based_matching(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    # read images
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    # define depth matrix and max value
    height, width = left.shape[:2]
    depth = np.zeros((height, width), dtype=np.uint8)
    scale = 3

    # kernel half: distance from center to the edge
    kernel_half = int((kernel_size - 1)/2)
    for y in range(kernel_half, height - kernel_half):
        for x in range(kernel_half, width - kernel_half):
            disparity = 0
            # cost = cosine => optinmal -> 1
            cost_optimal = -1
            for d in range(disparity_range):
                cost = -1
                if (x-d-kernel_half) > 0:
                    window_left = left[(
                        y-kernel_half):(y+kernel_half)+1, (x-kernel_half):(x+kernel_half)+1]
                    window_right = right[(
                        y-kernel_half):(y+kernel_half)+1, (x-d-kernel_half):(x-d+kernel_half)+1]

                    wl_flatten = window_left.flatten()
                    wr_flatten = window_right.flatten()
                    cost = cosine_similarity(wl_flatten, wr_flatten)
                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = d
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        cv2.imwrite(f'{path}window_based_cosine_similarity.png', depth)
        cv2.imwrite(f'{path}window_based_cosine_similarity_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth


left_img_path = 'assets/Aloe/Aloe_left_1.png'
right_img_path = 'assets/Aloe/Aloe_right_2.png'
disparity_range = 64
kernel_size = 3

window_based_result_cosine_similarity = window_based_matching(left_img_path,
                                                              right_img_path, disparity_range, kernel_size, save_result=True)
