import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox

class ChessboardCornerFinderApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Chessboard Corner Finder")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.load_images_button = QPushButton("Load Images")
        self.load_images_button.clicked.connect(self.load_images)
        layout.addWidget(self.load_images_button)

        self.find_corner_button = QPushButton("1.1 Find Corner")
        self.find_corner_button.clicked.connect(self.find_and_display_corners)
        layout.addWidget(self.find_corner_button)
        self.find_corner_button.setEnabled(False)

        self.find_intrinsic_button = QPushButton("1.2 Find Intrinsic")
        self.find_intrinsic_button.clicked.connect(self.find_intrinsic_matrix)
        layout.addWidget(self.find_intrinsic_button)
        self.find_intrinsic_button.setEnabled(False)

        self.image_paths = []
        self.current_image_index = 0
        self.object_points = []  # 物體三維點
        self.image_points = []   # 圖像二維點
        self.width = 11
        self.height = 8

    def load_images(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Images (*.bmp)")
        file_dialog.setViewMode(QFileDialog.List)
        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            if self.image_paths:
                self.find_corner_button.setEnabled(True)
                self.find_intrinsic_button.setEnabled(True)
                self.current_image_index = 0
                self.object_points = []
                self.image_points = []
                QMessageBox.information(self, "Info", "Images loaded successfully.")
    
    def find_and_display_corners(self):
        if self.image_paths:
            if self.current_image_index < len(self.image_paths):
                image_path = self.image_paths[self.current_image_index]
                image = cv2.imread(image_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray_image, (self.width, self.height), None)

                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                    image_with_corners = cv2.drawChessboardCorners(image, (self.width, self.height), corners, ret)
                    self.show_corners_image(image_with_corners)
                    self.add_object_and_image_points(corners)
                else:
                    QMessageBox.warning(self, "Warning", f"No corners found in {image_path}.")
                
                self.current_image_index += 1
                if self.current_image_index == len(self.image_paths):
                    self.find_corner_button.setEnabled(False)
            else:
                QMessageBox.information(self, "Info", "All images processed.")
        else:
            QMessageBox.warning(self, "Warning", "Please load images first.")
    
    def show_corners_image(self, image):
        if image is not None:
            cv2.imshow("Chessboard Corners", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def add_object_and_image_points(self, corners):
        # 生成物體三維點
        object_point = np.zeros((self.width * self.height, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        self.object_points.append(object_point)
        
        self.image_points.append(corners)
    
    def find_intrinsic_matrix(self):
        if len(self.object_points) > 0 and len(self.image_points) > 0:
            # 將物體三維點轉換為陣列
            object_points = np.array(self.object_points)
            image_points = np.array(self.image_points)

            # 計算內部矩陣
            ret, intrinsic_matrix, distortion_coefficients, _, _ = cv2.calibrateCamera(
                object_points, image_points, (self.width, self.height), None, None)

            if ret:
                print("Intrinsic Matrix:")
                print(intrinsic_matrix)
            else:
                QMessageBox.warning(self, "Warning", "Failed to compute Intrinsic Matrix.")
        else:
            QMessageBox.warning(self, "Warning", "No calibration data available. Please load and find corners in images first.")

def main():
    app = QApplication(sys.argv)
    window = ChessboardCornerFinderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
