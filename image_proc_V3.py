import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
import pandas as pd

def circle_intensity(y_c,x_c,R):
    sum_int = [0,0,0]
    for x_r in range(x_c-R, x_c+R):
        for y_r in range(y_c-R, y_c+R):
            if(((x_r-x_c)**2+(y_r-y_c)**2) < R**2):
                sum_int[0] = sum_int[0] + img[x_r, y_r][0]
                sum_int[1] = sum_int[1] + img[x_r, y_r][1]
                sum_int[2] = sum_int[2] + img[x_r, y_r][2]
    return np.array(sum_int)/1000000
	
# def rotate_image(image, angle):
#     """
#     Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
#     (in degrees). The returned image will be large enough to hold the entire
#     new image, with a black background
#     """

#     # Get the image size
#     # No that's not an error - NumPy stores image matricies backwards
#     image_size = (image.shape[1], image.shape[0])
#     image_center = tuple(np.array(image_size) / 2)

#     # Convert the OpenCV 3x2 rotation matrix to 3x3
#     rot_mat = np.vstack(
#         [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
#     )

#     rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

#     # Shorthand for below calcs
#     image_w2 = image_size[0] * 0.5
#     image_h2 = image_size[1] * 0.5

#     # Obtain the rotated coordinates of the image corners
#     rotated_coords = [
#         (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
#         (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
#     ]

#     # Find the size of the new image
#     x_coords = [pt[0] for pt in rotated_coords]
#     x_pos = [x for x in x_coords if x > 0]
#     x_neg = [x for x in x_coords if x < 0]

#     y_coords = [pt[1] for pt in rotated_coords]
#     y_pos = [y for y in y_coords if y > 0]
#     y_neg = [y for y in y_coords if y < 0]

#     right_bound = max(x_pos)
#     left_bound = min(x_neg)
#     top_bound = max(y_pos)
#     bot_bound = min(y_neg)

#     new_w = int(abs(right_bound - left_bound))
#     new_h = int(abs(top_bound - bot_bound))

#     # We require a translation matrix to keep the image centred
#     trans_mat = np.matrix([
#         [1, 0, int(new_w * 0.5 - image_w2)],
#         [0, 1, int(new_h * 0.5 - image_h2)],
#         [0, 0, 1]
#     ])

#     # Compute the tranform for the combined rotation and translation
#     affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

#     # Apply the transform
#     result = cv2.warpAffine(
#         image,
#         affine_mat,
#         (new_w, new_h),
#         flags=cv2.INTER_LINEAR
#     )

#     return result
    
# def crop_pic(image, width, height , x_scale, y_scale):
#     """
#     Given a NumPy / OpenCV 2 image, crops it to the given width and height,
#     around it's centre point
#     """

#     image_size = (image.shape[1], image.shape[0])
#     image_center = (int(image_size[0] * 0.5*x_scale), int(image_size[1] * 0.5*y_scale))

#     if(width > image_size[0]):
#         width = image_size[0]

#     if(height > image_size[1]):
#         height = image_size[1]

#     x1 = int(image_center[0] - width * 0.5)
#     x2 = int(image_center[0] + width * 0.5)
#     y1 = int(image_center[1] - height * 0.5)
#     y2 = int(image_center[1] + height * 0.5)

#     return image[y1:y2, x1:x2]
    
min_excitation = 0
max_excitation = 40*255/1_000_000
img1 = cv2.imread("test.jpeg")

img_scale = 0.5
rot_angle = -27
rot_cntr_sc_x = 1.145
rot_cntr_sc_y = 1.022
crop_w = int(0.32*img1.shape[1]*img_scale)
crop_h = int(0.428*img1.shape[0]*img_scale)
hole_dist = 148*img_scale 
hole_diam = int(68*img_scale) 
hole_diam = 200
hole_dist = 100
data=[0]*49
for i in range(0, 48):
	data[i+1]='tube'+str(i+1)
out_file = open("result_graph.csv", "w")
writer = csv.writer(out_file)
writer.writerow(data)

with open("centers.txt", "r") as f:
    centers = f.readlines()
centers = [x.strip() for x in centers]
centers = [x.split(' ') for x in centers]
centers = [[int(float(y)) for y in x] for x in centers]
# sort circles in a row from left to right and then from top to bottom
centers.sort(key=lambda x: (x[1], x[0]))
centers = pd.read_csv("sorted_centers.csv", header=None, index_col=None).values.tolist()

intensity_df = pd.DataFrame()
circle_x=[]
circle_y=[]
circle_cen=[]
for i in centers:
    circle_x.append(i[0])
for j in centers:
    circle_y.append(j[1])
for z in centers:
    circle_cen.append(z[2])

    
        
for x in range(2, 41):
    img2 = cv2.imread("test2.jpeg")

    # compute difference
    img= cv2.subtract(img2, img1)

    img = img[410:2285, 800:3330]

    # read the circle coordinates
    for idx, circle in enumerate(centers):
        tube_name = f"tube_{idx+1}"
        circle_center = (int(circle[0]), int(circle[1]))
        circle_radius = int(circle[2])
        cv2.circle(img, circle_center, circle_radius, (255, 0, 0), 10)
        aa = circle_intensity(*circle_center, circle_radius)
        intensity_df = intensity_df.append({'tube': tube_name, 'cycle_0': aa[1]}, ignore_index=True)
        # write tube number inside the circle
        cv2.putText(img, str(idx+1), circle_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        		
    intensity_df.to_csv('result_graph.csv', index=False)
    cv2.imwrite("dst_"+str(x)+".jpg", img)
    print(circle_center[0])
    cv2.imshow('image',img)
    cv2.waitKey(0)
   

