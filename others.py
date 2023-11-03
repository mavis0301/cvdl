import cv2
import numpy as np

# # 读取左右两幅图像
# imgL = cv2.imread('Q3_Image/imL.png', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('Q3_Image/imR.png', cv2.IMREAD_GRAYSCALE)
# imgL = cv2.resize(imgL, None, fx=0.6, fy=0.6)
# imgR = cv2.resize(imgR, None, fx=0.6, fy=0.6)
# stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
# disparity = stereo.compute(imgL, imgR)
# disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
# disparity = disparity.astype(np.uint8)
# disparity = cv2.resize(disparity, (800,600))
# cv2.imshow("disparity",disparity)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


left_image = cv2.imread('Q4_Image/Left.jpg')

left_image = cv2.resize(left_image, None, fx=0.3, fy=0.3)
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
# 创建SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和计算描述符
kp, des = sift.detectAndCompute(gray_left, None)

# 绘制关键点在左图像上
left_image_with_keypoints = cv2.drawKeypoints(left_image, kp, None, color=(0, 255, 0))

# 显示带有关键点的左图像
cv2.imshow('Left Image with Keypoints', left_image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()





