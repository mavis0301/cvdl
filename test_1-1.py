import cv2
import numpy as np

# 指定圖像路徑
image_path = "Q1_Image/1.bmp"

# 讀取圖像
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 指定棋盤格寬度和高度
width, height = 11, 8

# 使用cv2.findChessboardCorners找到角落
ret, corners = cv2.findChessboardCorners(gray_image, (width, height), None)

if ret:
    # 如果找到角落，進一步優化角落位置
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)

    # 繪製角落
    image_with_corners = cv2.drawChessboardCorners(image, (width, height), corners, ret)
    scaled_image = cv2.resize(image_with_corners, (800, 600))

    # 顯示圖像
    cv2.imshow("Chessboard Corners", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No corners found in the image.")