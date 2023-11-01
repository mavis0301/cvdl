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

        self.find_intrinsic_button = QPushButton("1.2 Find Intrinsic")
        self.find_intrinsic_button.clicked.connect(self.find_and_display_intrinsic_matrix)
        layout.addWidget(self.find_intrinsic_button)

        self.image_selection_combo = QComboBox()
        self.image_selection_combo.activated.connect(self.update_current_index)
        layout.addWidget(self.image_selection_combo)
        self.find_extrinsic_button = QPushButton("1.3 Find Extrinsic")
        self.find_extrinsic_button.clicked.connect(self.find_extrinsic_matrix)
        layout.addWidget(self.find_extrinsic_button)

        self.find_intrinsic_button = QPushButton("1.4 Find Distortion")
        self.find_intrinsic_button.clicked.connect(self.find_and_display_distortion)
        layout.addWidget(self.find_intrinsic_button)
        
        self.find_intrinsic_button = QPushButton("1.5 Show Result")
        self.find_intrinsic_button.clicked.connect(self.show_result1)
        layout.addWidget(self.find_intrinsic_button)
        

        self.width = 11
        self.height = 8
        self.image_paths = []
        self.current_image_index = 0
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.ret = None
        self.intrinsic_matrix = None
        self.dist = None
        # self.corners = None
        self.rvecs = None
        self.tvecs = None

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
                self.update_image_selection_combo()
                self.current_image_index = 0
                self.image = []
                for img_path in self.image_paths:
                    img = cv2.imread(img_path)
                QMessageBox.information(self, "Info", "Images loaded successfully.")
            else:
                QMessageBox.warning(self, "Warning", "Images loaded failed.")

    def update_image_selection_combo(self):
        self.image_selection_combo.clear()
        for i in range(len(self.image_paths)):
            self.image_selection_combo.addItem(str(i+1))
    
    def update_current_index(self, index):
        self.current_image_index = index

    def findCorner(self, img):
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(grayImg, (self.width, self.height), None)
        if ret:
            cv2.cornerSubPix(grayImg, corners, (5, 5), (-1, -1), self.criteria)
            return grayImg,corners
        else:
            QMessageBox.warning(self, "Warning", "No corners found in the image.")

    def generate_object_points(self):#can merge to func
        object_point = np.zeros((self.width * self.height, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:self.width, 0:self.height].T.reshape(-1, 2)
        return object_point
    
    def find_and_display_corners(self):##1.1
       for img_path in self.image_paths:
            img = cv2.imread(img_path)
            grayImg, corners = self.findCorner(img)
            cv2.cornerSubPix(grayImg, corners, (5, 5), (-1, -1), self.criteria)
            cornerImage = cv2.drawChessboardCorners(img, (self.width, self.height), corners, True)
            cornerImage = cv2.resize(cornerImage, (800, 600))
            if cornerImage is not None:
                cv2.imshow("Chessboard Corners", cornerImage)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                QMessageBox.warning(self, "Warning", "No corners image.")

    def find_instrinsic_matrix(self):
        object_points = []
        image_points = []
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            grayImg, corners = self.findCorner(img)
            object_points.append(self.generate_object_points())
            image_points.append(corners)            

        self.ret, self.intrinsic_matrix, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(object_points, image_points, grayImg.shape[::-1], None, None)


    def find_and_display_intrinsic_matrix(self):##1.2
        if len(self.image_paths) > 0:
            self.find_instrinsic_matrix()
            print("Intrinsic Matrix:")
            print(self.intrinsic_matrix)
        else:
            QMessageBox.information(self, "Info", "No images loaded.")

    def find_extrinsic_matrix(self):
        selected_index = self.current_image_index
        selected_image_path = self.image_paths[selected_index]
        self.find_instrinsic_matrix()
        rotation_matrix, _ = cv2.Rodrigues(self.rvecs[selected_index])
        extrinsic_matrix = np.hstack((rotation_matrix, self.tvecs[selected_index]))
        print("Extrinsic Matrix:")
        print(extrinsic_matrix)
    
    def find_and_display_distortion(self):
        self.find_instrinsic_matrix()
        print(self.dist)

    def show_result1(self):
        self.find_instrinsic_matrix()
        for i in self.image_paths:
            img = cv2.imread(i)
            undistorted_image = cv2.undistort(img, self.intrinsic_matrix, self.dist)
            img = cv2.resize(img, (800, 600))
            undistorted_image = cv2.resize(undistorted_image, (800, 600))
            merged_image = np.hstack((img, undistorted_image))
            cv2.imshow("Original and Undistorted", merged_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()







        

def main():
    app = QApplication(sys.argv)
    window = ChessboardCornerFinderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
