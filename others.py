import cv2
import numpy as np

# 读取左右两幅图像
imgL = cv2.imread('Q3_Image/imL.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('Q3_Image/imR.png', cv2.IMREAD_GRAYSCALE)
imgL = cv2.resize(imgL, None, fx=0.5, fy=0.5)
imgR = cv2.resize(imgR, None, fx=0.5, fy=0.5)
stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
disparity = stereo.compute(imgL, imgR)
disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disparity = disparity.astype(np.uint8)
# disparity = cv2.resize(disparity, None, fx=0.5, fy=0.5)
# disparity = cv2.resize(disparity, (800,600))
# cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
cv2.imshow("disparity",disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
