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


import cv2

# 加载左图像（Image 1）
left_image = cv2.imread('Q4_Image/Left.jpg')
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)

# 加载右图像（Image 2）
right_image = cv2.imread('Q4_Image/Right.jpg')
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

# 创建SIFT检测器
sift = cv2.SIFT_create()

# 在左图像和右图像上检测关键点和计算描述符
kp1, des1 = sift.detectAndCompute(gray_left, None)
kp2, des2 = sift.detectAndCompute(gray_right, None)

# 使用BFMatcher.knnMatch来匹配关键点
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 提取好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配关键点
matched_image = cv2.drawMatches(left_image, kp1, right_image, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
matched_image = cv2.resize(matched_image,None,fx=0.3,fy=0.3)
# 显示匹配关键点的图像
cv2.imshow('Matched Keypoints', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





