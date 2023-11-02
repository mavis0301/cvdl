import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QFileDialog, QMessageBox, QComboBox, QLineEdit
from PyQt5.QtCore import Qt
import os

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

        self.load_image_button = QPushButton("Load folder")
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)

        self.find_corner_button = QPushButton("1.1 Find Corners")
        self.find_corner_button.clicked.connect(self.find_and_display_corners)
        layout.addWidget(self.find_corner_button) 

        self.find_intrinsic_button = QPushButton("1.2 Find intrinsic")
        self.find_intrinsic_button.clicked.connect(self.find_and_display_intrinsic_matrix)
        layout.addWidget(self.find_intrinsic_button)

        self.image_selection_combo = QComboBox()
        self.image_selection_combo.activated.connect(self.update_current_index)
        layout.addWidget(self.image_selection_combo)
        self.find_extrinsic_button = QPushButton("1.3 Find extrinsic")
        self.find_extrinsic_button.clicked.connect(self.find_extrinsic_matrix)
        layout.addWidget(self.find_extrinsic_button)

        self.find_intrinsic_button = QPushButton("1.4 Find distortion")
        self.find_intrinsic_button.clicked.connect(self.find_and_display_distortion)
        layout.addWidget(self.find_intrinsic_button)
        
        self.find_intrinsic_button = QPushButton("1.5 Show result")
        self.find_intrinsic_button.clicked.connect(self.show_result1)
        layout.addWidget(self.find_intrinsic_button)

        self.inputWord = QLineEdit(self)
        self.inputWord.setMaxLength(6)
        layout.addWidget(self.inputWord)
        
        self.find_intrinsic_button = QPushButton("2.1 show words on board")
        self.find_intrinsic_button.clicked.connect(self.showWord)
        layout.addWidget(self.find_intrinsic_button)

        self.find_intrinsic_button = QPushButton("2.2 show words vertical")
        self.find_intrinsic_button.clicked.connect(self.showWord_Vertical)
        layout.addWidget(self.find_intrinsic_button)

        self.open_ImgL = QPushButton("Load Image_L", self)
        # self.open_file_button.setGeometry(150, 80, 200, 30)
        self.open_ImgL.clicked.connect(self.loadImgL)
        layout.addWidget(self.open_ImgL)

        self.open_ImgR = QPushButton("Load Image_R", self)
        # self.open_file_button.setGeometry(150, 80, 200, 30)
        self.open_ImgR.clicked.connect(self.loadImgR)
        layout.addWidget(self.open_ImgR)

        self.find_disparity = QPushButton("3.1 stereo disparity map", self)
        # self.open_file_button.setGeometry(150, 80, 200, 30)
        self.find_disparity.clicked.connect(self.findDisparityMap)
        layout.addWidget(self.find_disparity)


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
        self.wordLoc = [[7,5],[4,5],[1,5],[7,2],[4,2],[1,2]]

    def load_image(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a folder")
        if folder:
            file_list = os.listdir(folder)
            bmp_files = [file for file in file_list if file.lower().endswith(".bmp")]
            self.image_paths = [os.path.join(folder, bmp_file) for bmp_file in bmp_files]
            self.update_image_selection_combo()
            self.current_image_index = 0
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

    def generate_object_points(self):
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
        self.find_instrinsic_matrix()
        rotation_matrix, _ = cv2.Rodrigues(self.rvecs[selected_index])
        extrinsic_matrix = np.hstack((rotation_matrix, self.tvecs[selected_index]))
        print("Extrinsic Matrix",selected_index+1,":")
        print(extrinsic_matrix)
    
    def find_and_display_distortion(self):
        self.find_instrinsic_matrix()
        print('Distortion Matrix')
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


    def showWord(self):##2.1
        fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        input = self.inputWord.text()
        letters = list(input)
        chlist = np.empty((0, 3), dtype=np.int32)
        for c in range(len(letters)):
            ch = fs.getNode(letters[c]).mat()
            ch = np.float32(ch).reshape(-1,3)
            ch[:,0] = ch[:,0]+self.wordLoc[c][0]
            ch[:,1] = ch[:,1]+self.wordLoc[c][1]
            chlist = np.vstack((chlist, ch))
        for i in range(len(self.image_paths)):
            image = cv2.imread(self.image_paths[i])
            self.find_instrinsic_matrix()
            imgpts, jac = cv2.projectPoints(chlist, self.rvecs[i], self.tvecs[i], self.intrinsic_matrix, self.dist)
            imgpts = imgpts.astype(np.int32)
            for j in range(len(imgpts)//2):
                image = cv2.line(image, tuple(imgpts[j*2].ravel()), tuple(imgpts[j*2+1].ravel()), (0, 0, 255), 5)
            image = cv2.resize(image, (800, 600))
            cv2.imshow("word", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def showWord_Vertical(self):##2.2
        fs = cv2.FileStorage('Q2_Image/Q2_lib/alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        input = self.inputWord.text()
        letters = list(input)
        chlist = np.empty((0, 3), dtype=np.int32)
        for c in range(len(letters)):
            ch = fs.getNode(letters[c]).mat()
            ch = np.float32(ch).reshape(-1,3)
            ch[:,0] = ch[:,0]+self.wordLoc[c][0]
            ch[:,1] = ch[:,1]+self.wordLoc[c][1]
            chlist = np.vstack((chlist, ch))
        for i in range(len(self.image_paths)):
            image = cv2.imread(self.image_paths[i])
            self.find_instrinsic_matrix()
            imgpts, jac = cv2.projectPoints(chlist, self.rvecs[i], self.tvecs[i], self.intrinsic_matrix, self.dist)
            imgpts = imgpts.astype(np.int32)
            for j in range(len(imgpts)//2):
                image = cv2.line(image, tuple(imgpts[j*2].ravel()), tuple(imgpts[j*2+1].ravel()), (0, 0, 255), 5)
            image = cv2.resize(image, (800, 600))
            cv2.imshow("word", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def loadImgL(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgL_path, _ = QFileDialog.getOpenFileName(self, "Open PNG File", "", "PNG Files (*.png);;All Files (*)", options=options)
        self.imgL = cv2.imread(self.imgL_path, cv2.IMREAD_GRAYSCALE)


    def loadImgR(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.imgR_path, _ = QFileDialog.getOpenFileName(self, "Open PNG File", "", "PNG Files (*.png);;All Files (*)", options=options)
        self.imgR = cv2.imread(self.imgR_path, cv2.IMREAD_GRAYSCALE)

    def findDisparityMap(self):
        imgl = self.imgL
        imgr = self.imgR
        imgL = cv2.resize(imgl, None, fx=0.6, fy=0.6)
        imgR = cv2.resize(imgr, None, fx=0.6, fy=0.6)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)
        disparity = stereo.compute(imgL, imgR)
        disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disparity = disparity.astype(np.uint8)
        disparity= cv2.resize(disparity, (800, 600))
        imgL= cv2.resize(imgL, (800, 600))
        imgR= cv2.resize(imgR, (800, 600))
        cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
        cv2.imshow("R",imgR)
        cv2.imshow("L",imgL)
        cv2.imshow("disparity",disparity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    
    






        

def main():
    app = QApplication(sys.argv)
    window = ChessboardCornerFinderApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
