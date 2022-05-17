import cv2
import LBP

texture_path = "D:/python_pro/Datasets/anno_data_v220415/Floor_seg/"
img = cv2.imread(texture_path + "{}.jpg".format(2), 0)
img = LBP.LBP(img)   # circular_
cv2.imwrite("wenli4.png", img)

