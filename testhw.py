import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox, QComboBox
from PyQt5.QtCore import QTimer

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

        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.find_corner_button = QPushButton("1.1 Find Corners")
        self.find_corner_button.clicked.connect(self.find_and_display_corners)
        layout.addWidget(self.find_corner_button)
        self.find_corner_button.setEnabled(False)  

        self.find_intrinsic_button = QPushButton("1.2 Find Intrinsic")
        self.find_intrinsic_button.clicked.connect(self.find_intrinsic_matrix)
        layout.addWidget(self.find_intrinsic_button)
        self.find_intrinsic_button.setEnabled(False)

        # self.image_selection_combo = QComboBox()
        # self.image_selection_combo.activated.connect(self.update_current_index)
        # layout.addWidget(self.image_selection_combo)
        # self.find_extrinsic_button = QPushButton("1.3 Find Extrinsic")
        # self.find_extrinsic_button.clicked.connect(self.find_extrinsic_matrix)
        # layout.addWidget(self.find_extrinsic_button)
        # self.find_extrinsic_button.setEnabled(False)

        

        self.width = 11
        self.height = 8
        self.image_paths = []
        self.image = []


    def load_image(self):
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
                # self.update_image_selection_combo()
                # self.find_extrinsic_button.setEnabled(True)
                self.current_image_index = 0
                self.image = []
                for img_path in self.image_paths:
                    img = cv2.imread(img_path)
                    self.image.append(img)


                QMessageBox.information(self, "Info", "Images loaded successfully.")
            else:
                QMessageBox.warning(self, "Warning", "Images loaded failed.")

    def update_image_selection_combo(self):
        self.image_selection_combo.clear()
        for i in range(len(self.image_paths)):
            self.image_selection_combo.addItem(str(i+1))
    
    def update_current_index(self, index):
        self.current_image_index = index
    
    def find_and_display_corners(self):##1.1
        for img in self.image:
            grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(grayImg, (self.width, self.height), None)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                cv2.cornerSubPix(grayImg, corners, (5, 5), (-1, -1), criteria)
                cornerImage = cv2.drawChessboardCorners(img, (self.width, self.height), corners, ret)
                cornerImage = cv2.resize(cornerImage, (800, 600))
                if cornerImage is not None:
                    cv2.imshow("Chessboard Corners", cornerImage)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                QMessageBox.warning(self, "Warning", "No corners found in the image.")

    
    def generate_object_points(self):#can merge to func
        object_point = np.zeros((self.width * self.height, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        return object_point
    
    def find_intrinsic_matrix(self):##1.2
        if self.image_paths:
            if len(self.image_paths) > 0:
                object_points = []
                image_points = []
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

                for img_path in self.image_paths:
                    image = cv2.imread(img_path)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray_image, (self.width, self.height), None)
                    if ret:
                        cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), criteria)
                        object_points.append(self.generate_object_points())
                        image_points.append(corners)
                    else:
                        QMessageBox.warning(self, "Warning", "No corners found in the image.")
                
                if len(object_points) > 0 and len(image_points) > 0:
                    _, intrinsic_matrix, _, _, _ = cv2.calibrateCamera(object_points, image_points, gray_image.shape[::-1], None, None)
                    print("Intrinsic Matrix:")
                    print(intrinsic_matrix)
                else:
                    QMessageBox.warning(self, "Warning", "No calibration data available. Please load and find corners in images first.")
            else:
                QMessageBox.information(self, "Info", "No images loaded.")
        else:
            QMessageBox.warning(self, "Warning", "Please load images first.")

    # def find_extrinsic_matrix(self):
    #     if self.image_paths:
    #         if len(self.image_paths) > 0:
    #             if self.current_image_index >= 0 and self.current_image_index < len(self.image_paths):
    #                 image_path = self.image_paths[self.current_image_index]
    #                 image = cv2.imread(image_path)
    #                 gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #                 ret, corners = cv2.findChessboardCorners(gray_image, (self.width, self.height), None)
                    
    #                 if ret:
    #                     object_points = self.generate_object_points()
    #                     cv2.cornerSubPix(gray_image, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                        
    #                     _, intrinsic_matrix, _, _, _ = cv2.calibrateCamera(object_points, [corners], gray_image.shape[::-1], None, None)
    #                     print("Intrinsic Matrix:")
    #                     print(intrinsic_matrix)
    #                 else:
    #                     QMessageBox.warning(self, "Warning", "No corners found in the selected image.")
    #             else:
    #                 QMessageBox.warning(self, "Warning", "Please select an image first.")
    #         else:
    #             QMessageBox.information(self, "Info", "No images loaded.")
    #     else:
    #         QMessageBox.warning(self, "Warning", "Please load images first.")



        

def main():
    app = QApplication(sys.argv)
    window = ChessboardCornerFinderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
