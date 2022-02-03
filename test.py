import os
import cv2

CURRENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

image = cv2.imread('input_images/image1.jpg')
cv2.imwrite('output_images/new_image.jpg', image)

for x in os.listdir('input_images'):
    path_input_image = "input_images/" + x
    output_image_name = x.split('.')[0] + "_fin." + x.split('.')[1]
    path_output_image = "output_images/" + output_image_name
    
    image = cv2.imread(path_input_image)
    orig = image.copy()
    #do something here
    cv2.imwrite(path_output_image, orig)

    


