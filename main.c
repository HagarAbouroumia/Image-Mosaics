import numpy as np
import cv2
import functions as f
from Image import Image
import os
import glob
import time
import imutils

start = 0
folder_name = "images"
if __name__ == "__main__":
    images = os.listdir(str(folder_name))
    images.sort()
    print(images)
    original_img = Image(cv2.imread(str(folder_name) + "/" + str(images[0])))
    start = time.time()
    j = len(images)

    for i in range(1, j):
        projected_img = Image(cv2.imread(str(folder_name) + "/" + str(images[i])))
        print(images[i])
        print(i)
        matches, kp1, kp2 = f.features_detection(projected_img.get_image(), original_img.get_image())

        # projected_img.set_resize(500,500)
        img1_coor, img2_coor = f.get_correspondence_points(matches, kp1, kp2)
        p = projected_img.get_image()
        H, _ = cv2.findHomography(np.array(img1_coor, np.float32), np.array(img2_coor, np.float32), cv2.RANSAC, 5.0)
        # f.check_H(H, img1_coor[1], projected_img.get_image(), original_img.get_image())
        temp1 = f.plot_points_on_image(original_img.get_image(), img2_coor)
        temp2 = f.plot_points_on_image(projected_img.get_image(), img1_coor)

        cv2.imshow("image1", temp1)
        cv2.imshow("image2", temp2)
        cv2.waitKey(0)
        projected_img_corners = cv2.perspectiveTransform(projected_img.get_corners(), H)
        xmin, ymin, xmax, ymax = f.get_min_max(projected_img_corners)
        x_translation = np.abs(np.min([0, xmin]))
        y_translation = np.abs(np.min([0, ymin]))
        corners_tran = original_img.get_corners()
        corners_tran[:, 0, 0] += x_translation + np.max([0, xmax - original_img.shape(1)])
        corners_tran[:, 0, 1] += y_translation + np.max([0, ymax - original_img.shape(0)])
        _, _, xmax, ymax = f.get_min_max(corners_tran)

        output_img = Image(np.empty((ymax, xmax, 3), np.uint8))
        output_img.set_pixels(original_img.shape(0), original_img.shape(1), original_img.get_image())
        T = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
        output_img.set_image(cv2.warpAffine(output_img.get_image(), T, (output_img.shape(1), output_img.shape(0))))
        translate = np.eye(3)
        translate[0, 2] = x_translation
        translate[1, 2] = y_translation
        H = np.matmul(translate, H)

        output_img.set_image(f.forward_warping(projected_img.get_image(), H, output_img.get_image()))
        output_img.set_image(f.inverse_warping(output_img.get_image(), projected_img.get_image(), H))
        output_img.write_image("out.jpg")
        del original_img
        del projected_img
        del output_img

        original_img = Image(cv2.imread("out.jpg"))
print(start - time.time())
f.get_final_img(cv2.imread("out.jpg"))
