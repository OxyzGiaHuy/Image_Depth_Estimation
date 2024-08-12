import cv2
import numpy as np

path = 'result/window_based/'


def l1_distance(x, y):
    return abs(x-y)


def l2_distance(x, y):
    return (x-y)**2


def window_based_matching(left_img, right_img, disparity_range, distance_method, kernel_size=5, save_result=True):
    # read images
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    #define depth matrix and max value
    height, width = left.shape[:2]
    depth = np.zeros((height, width), dtype=np.uint8)
    if distance_method == l1_distance:
        max_value = 255*9
    elif distance_method == l2_distance:
        max_value = 255**2*9
    else:
        max_value = 1e7
    scale = 3

    # kernel half: distance from center to the edge
    kernel_half = int((kernel_size - 1)/2)
    for y in range(kernel_half, height - kernel_half + 1):
        for x in range(kernel_half, width - kernel_half + 1):
            disparity = 0
            cost_min = max_value
            for d in range(disparity_range):
                window_cost = 0
                # calculate 1 cost
                for v in range(-kernel_half, kernel_half):
                    for u in range(-kernel_half, kernel_half):
                        if (x+u-d) < 0:
                            cost = max_value
                        else:
                            cost = distance_method(
                                left[y+v, x+u], right[y+v, x+u-d])
                        window_cost += cost
                if window_cost < cost_min:
                    cost_min = window_cost
                    disparity = d
            depth[y, x] = disparity * scale

    if save_result == True:
        print('Saving result...')
        # Save results
        end_name = "l1" if distance_method == l1_distance else "l2"

        cv2.imwrite(f'{path}window_based_{end_name}.png', depth)
        cv2.imwrite(f'{path}window_based_{end_name}_color.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')
    return depth


left_img_path = 'assets/Aloe/Aloe_left_1.png'
right_img_path = 'assets/Aloe/Aloe_right_1.png'
disparity_range = 64
kernel_size = 3

window_based_result_l1 = window_based_matching(left_img_path,
                                               right_img_path,
                                               disparity_range,
                                               l1_distance,
                                               kernel_size,
                                               save_result=True)

window_based_result_l2 = window_based_matching(left_img_path,
                                               right_img_path,
                                               disparity_range,
                                               l2_distance,
                                               kernel_size,
                                               save_result=True)
