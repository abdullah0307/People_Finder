import csv
import os
import threading
import time
from datetime import datetime

import cv2
import dlib
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QThreadPool, pyqtSlot, QRunnable, QObject
from PyQt5.QtWidgets import QFileDialog

import My_model
from del_dialoge import Ui_Dialog as DD
from fill_dialoge import Ui_Dialog as FD
from finalUI import Ui_MainWindow
from ip_dialoge import Ui_Dialog as IP
from loader import Ui_loader
from select_dialoge import Ui_Dialog as SD
from update_window import Ui_update_window

counter = 0
thread_break = False


class Signals(QObject):
    return_signal = pyqtSignal(str)


class Thread(QRunnable):
    signal = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()
        self.signal = Signals()

    @pyqtSlot()
    def run(self):
        while True:
            if thread_break:
                break
            time.sleep(1)
            result = "output/training_plot.jpg"
            self.signal.return_signal.emit(result)


class Main:

    def __init__(self):
        # initialize the model
        self.trainer_model = My_model.AD_Model()

        self.threadpool = QThreadPool()
        self.ipaddress = ""
        self.name = ""
        self.Id = ""
        self.gender = ""
        self.age = ""
        self.phone = ""
        self.address = ""
        self.sample_taken = False
        self.show_graph = False

        # todo loading window
        self.loading_window = QtWidgets.QMainWindow()
        self.loading_obj = Ui_loader()
        self.loading_obj.setupUi(self.loading_window)
        self.loading_window.show()

        # todo setting timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.loader_progress_bar)
        self.timer.start(35)

        # todo main_window
        self.main_window = QtWidgets.QMainWindow()
        self.main_obj = Ui_MainWindow()
        self.main_obj.setupUi(self.main_window)

        # connect buttons to their functions
        self.main_obj.camera1.clicked.connect(self.sample_from_camera)
        self.main_obj.gallery1.clicked.connect(self.sample_from_gallery)
        self.main_obj.table.itemClicked.connect(self.existing_item_clicked)
        self.main_obj.train4_1.clicked.connect(self.Start_training_thread)
        self.main_obj.track_web.clicked.connect(self.Track_subject_on_WebCam)
        self.main_obj.track_video.clicked.connect(self.Track_subject_on_video)
        self.main_obj.track_ip.clicked.connect(self.show_ipdialogue)
        self.main_obj.track3_1.clicked.connect(self.main_obj.track3.click)
        self.main_obj.train3_1.clicked.connect(self.main_obj.train3.click)
        self.main_obj.report5_1.clicked.connect(self.main_obj.report5.click)
        self.main_obj.table.itemClicked.connect(self.show_report)
        self.main_obj.video6.clicked.connect(self.play_video)

        # todo loading data
        self.load_data()

        # todo ipaddress dialogue
        self.ip_input_dia = QtWidgets.QDialog()
        self.ip_input_obj = IP()
        self.ip_input_obj.setupUi(self.ip_input_dia)
        self.ip_input_obj.ip_address.setMaxLength(100)

        # todo update_window
        self.update_window = QtWidgets.QMainWindow()
        self.update_obj = Ui_update_window()
        self.update_obj.setupUi(self.update_window)
        self.ip_input_obj.ip_buttonBox.accepted.connect(self.Track_Subject_Over_Ip_Camera)
        self.ip_input_obj.ip_buttonBox.rejected.connect(self.ip_input_dia.close)

        # todo fill_dialogue
        self.fill_dia = QtWidgets.QDialog()
        self.fill_obj = FD()
        self.fill_obj.setupUi(self.fill_dia)
        self.fill_obj.ok.clicked.connect(lambda: self.fill_dia.destroy())

        # todo del_dialogue
        self.del_dia = QtWidgets.QDialog()
        self.del_obj = DD()
        self.del_obj.setupUi(self.del_dia)

        # todo select_dialoge
        self.select_dia = QtWidgets.QDialog()
        self.select_obj = SD()
        self.select_obj.setupUi(self.select_dia)

        # set maximum value of epochs to 1000
        self.main_obj.epochs.setRange(0, 1000)

        # Start threading of training
        self.graph_thread = Thread()
        self.training_thread = threading.Thread(target=self.StartTraining)

        # Start the screen from main screen
        self.main_obj.stackedWidget.setCurrentWidget(self.main_obj.page1)

        self.update_obj.cancel.clicked.connect(self.update_window.close)

        self.select_obj.ok.clicked.connect(self.select_dia.close)
        self.del_obj.del_buttonBox.rejected.connect(self.main_obj.table.clearSelection)
        self.update_obj.cancel.clicked.connect(self.main_obj.table.clearSelection)
        # self.update_obj.update.clicked.connect(self)

    def play_video(self):
        # Select that activated row
        row = self.main_obj.table6.selectedItems()

        # Update the data
        Id = row[0].text()
        name = row[1].text()
        video = row[4].text()

        video_path = "Report/" + str(name) + "." + str(Id) + "." + str(video) + ".mp4"
        cap = cv2.VideoCapture(video_path)
        print(video_path)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Report/" + name + "." + Id + "." + video + ".mp4", frame)
                if cv2.waitKey(10) & 0xff == ord('q'):
                    break
            else:
                break
        cv2.destroyAllWindows()
        cap.release()

    def show_report(self):
        self.main_obj.table6.clear()
        with open("Report/report.csv") as f:

            file_data = []
            row = csv.reader(f)
            for x in row:
                file_data.append(x)
            self.main_obj.table6.setRowCount(0)
            file_data = iter(file_data)
            next(file_data)
            for row, rd in enumerate(file_data):
                self.main_obj.table6.insertRow(row)
                for col, data in enumerate(rd):
                    self.main_obj.table6.setItem(row, col, QtWidgets.QTableWidgetItem(str(data)))

    # Start the Thread of Training
    def Start_training_thread(self):

        if self.training_thread.is_alive():
            return

        # Update graph thread
        self.graph_thread.signal.return_signal.connect(self.update_Graph)
        self.threadpool.start(self.graph_thread)
        self.training_thread.start()

        # After that reset the graph picture
        reset_image = cv2.imread("output/new.jpg")
        cv2.imwrite("output/training_plot.jpg", reset_image)

    # Update the recent saved graph picture
    def update_Graph(self, path):
        self.main_obj.label_graph.setPixmap(QtGui.QPixmap(path))
        self.main_obj.progressBar4.setValue(int((self.trainer_model.get_epoch() / self.main_obj.epochs.value()) * 100))

    # Check weather the ip address is valid or not
    def validate_ip(self, s):
        a = s.split('.')
        if len(a) != 4:
            return False
        return True

    # Here we show the dialogue box to input the ipaddress
    def show_ipdialogue(self):

        # Check that values are entered or not
        if self.Id != "" and self.name != "":
            self.ip_input_dia.show()
            print(self.ipaddress)
        else:
            self.fill_obj.label.setText("Select a Subject to track")
            self.fill_dia.show()

    # The subject will be track over the ipaddress
    def Track_Subject_Over_Ip_Camera(self):

        # Now get the ipaddress of the video from the webcam
        self.ipaddress = self.ip_input_obj.ip_address.text()

        if not self.validate_ip(self.ipaddress):
            self.fill_obj.label.setText("IP not Valid")
            self.fill_dia.show()

        try:
            # load the model if exists
            recognizer = My_model.AD_Model()

            if not os.path.exists("model/model.h5"):
                self.fill_obj.label.setText("No Model trained")
                self.fill_dia.show()

                return

            # read it and load the files
            recognizer.load_model("model/model.h5")

            # load the essential file for classification of faces
            detector = dlib.get_frontal_face_detector()

            font = cv2.FONT_HERSHEY_SIMPLEX
            df = pd.read_csv("Subject Details/subjects_data.csv")

            # To calculate the frame rate
            fps_start_time = datetime.now()
            total_frames = 0

            # start the ipaddress
            cam = cv2.VideoCapture(self.ipaddress)

            # Load Report
            report_df = pd.read_csv("Report/report.csv")

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
            n = len(os.listdir("Report"))
            vid_out = cv2.VideoWriter('Report/' + self.name + "." + self.Id + "." + str(n) + ".mp4", fourcc, 20.0,
                                      (int(w), int(h)))

            # Take sample flag
            record = False

            # message to display
            msg = " Searching for face"
            msg1 = self.Id + " " + self.name
            msg2 = "press r to record"

            # Now start the loop for the video capturing
            while True:
                ret, im = cam.read()
                if not ret:
                    continue
                im = cv2.flip(im, 1)

                # coordinates of the screen
                y, x, _ = im.shape

                # To calculate the fps rate in the video from camera
                total_frames = total_frames + 1
                fps_end_time = datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)
                fps_text = "FPS: {:.2f}".format(fps)

                # Current time and date
                now = datetime.now()
                timestamp = datetime.timestamp(now)
                current_time = now.strftime("%H:%M:%S")
                current_date = datetime.fromtimestamp(timestamp)

                # convert the frame into grayscale and get the face Coordinates
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                # Check face or not found
                if len(faces) is not 0:

                    # Now get all the face in the frame
                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()

                        # Now get the reign of interest of the face and get the prediction over that face
                        roi = gray[y1:y2, x1:x2]

                        resized_roi = None
                        Id = None
                        conf = 0
                        try:
                            resized_roi = cv2.resize(roi, (100, 100))
                        except:
                            continue
                        # predict the subject
                        Id, conf = recognizer.predict(resized_roi)
                        print(Id)
                        if Id == 0:
                            # Draw a red box over the face
                            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            # Put the text of unknown over the face
                            cv2.putText(im, "unknown", (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)

                            continue

                        if int(self.Id) == df['ID'][Id - 1]:
                            name = df['NAME'][Id - 1]
                            Id_and_name = str(Id - 1) + " " + name

                            # draw a green rectangle over the face
                            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # show the result of the faces name and id over that face and their confidence
                            cv2.putText(im, str(Id_and_name), (x1, y2 + 50), font, 1, (0, 255, 0), 1)
                            cv2.putText(im, "confidence:" + str(int(conf)) + "%", (x1, y2 + 30), font, 1,
                                        (0, 255, 0), 1)
                            ts = time.time()
                            date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                            timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                            report_df.loc[len(report_df)] = [self.Id, self.name, date, timeStamp, int(n)]

                        else:
                            # draw a read rectangle over the face
                            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            # display the message of the face is unknown
                            cv2.putText(im, "UNKNOWN", (x1, y2 + 50), font, 1, (0, 255, 0), 1)

                        report_df = report_df.drop_duplicates(subset=['ID'], keep='first')

                else:
                    msg = " Searching for face"

                font = cv2.FONT_HERSHEY_TRIPLEX
                size = 0.7
                thickness = 1
                x_offset = 200

                # Bottom left corner message area
                cv2.putText(im, "Status:" + msg, (10, y - 20), font, size, (0, 255, 0), thickness)
                cv2.putText(im, "People Finder", (10, y - 40), font, size, (255, 255, 255), thickness)

                # Write the count of the images
                cv2.putText(im, fps_text, (10, 40), font, size, (0, 255, 0), thickness)

                # Write the count of the images
                cv2.putText(im, "Searching for:" + msg1, (10, 60), font, size, (0, 255, 0), thickness)

                # Write the current date of the images
                cv2.putText(im, str(current_date.date()), (x - x_offset, 40), font, size, (0, 255, 0), thickness)

                # Write the current date of the images
                cv2.putText(im, str(current_time), (x - x_offset, 60), font, size, (0, 255, 0), thickness)

                # Write the messages at the bottom left corner of the screen
                cv2.putText(im, "q to quit", (x - x_offset, y - 20), font, size, (0, 255, 0), thickness)
                cv2.putText(im, msg2, (x - x_offset, y - 40), font, size, (0, 255, 0), thickness)

                cv2.imshow('LIVE CAMERA STREAM FROM ' + self.ipaddress, im)

                # record if wanted
                if record:
                    msg2 = "Recording..."
                    # write the flipped frame
                    vid_out.write(im)
                else:
                    msg2 = "r to record"

                # wait for 100 milliseconds
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if record:
                        record = False
                    else:
                        record = True

            # Saving the report
            report_df.to_csv("Report/report.csv", index=False)

            # release the camera and the window will be destroyed
            cam.release()
            vid_out.release()
            cv2.destroyAllWindows()

        except Exception as e:
            self.fill_obj.label.setText(str(e))
            self.fill_dia.show()

    def Track_subject_on_WebCam(self):

        # return if the subject is not selected
        if self.Id == "" and self.name == "":
            self.fill_obj.label.setText("Please select a subject to track")
            self.fill_dia.show()

            return
        # return of the model is not their
        if not os.path.exists("model/model.h5"):
            self.fill_obj.label.setText("No Model trained")
            self.fill_dia.show()

            return

        # load the model if exists
        recognizer = My_model.AD_Model()

        # read it and load the files
        recognizer.load_model("model/model.h5")

        # face detector
        detector = dlib.get_frontal_face_detector()

        # set the font of the text
        font = cv2.FONT_HERSHEY_SIMPLEX

        # load the csv file
        df = pd.read_csv("Subject Details/subjects_data.csv")

        # To calculate the frame rate
        fps_start_time = datetime.now()
        total_frames = 0

        cam = None

        # Check for the camera is found or no found
        try:
            # start the default webcam
            cam = cv2.VideoCapture(0)

        except Exception as e:
            self.fill_obj.label.setText("webcam not found")
            self.fill_dia.show()
            return

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n = len(os.listdir("Report"))
        vid_out = cv2.VideoWriter('Report/' + self.name + "." + self.Id + "." + str(n) + ".mp4", fourcc, 20.0,
                                  (int(w), int(h)))

        # Load Report
        report_df = pd.read_csv("Report/report.csv")

        # record flag
        record = False

        # message to display
        msg = " Searching for face"
        msg1 = self.Id + " " + self.name
        msg2 = "press r to record"

        # Now start the loop for the video capturing
        while True:
            ret, im = cam.read()
            im = cv2.flip(im, 1)

            # coordinates of the screen
            y, x, _ = im.shape

            # To calculate the fps rate in the video from camera
            total_frames = total_frames + 1
            fps_end_time = datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)
            fps_text = "FPS: {:.2f}".format(fps)

            # Current time and date
            now = datetime.now()
            timestamp = datetime.timestamp(now)
            current_time = now.strftime("%H:%M:%S")
            current_date = datetime.fromtimestamp(timestamp)

            # convert the frame into grayscale and get the face Coordinates
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            # Check face or not found
            if len(faces) != 0:

                # Now get all the face in the frame
                for face in faces:

                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()

                    # Now get the reign of interest of the face and get the prediction over that face
                    roi = gray[y1:y2, x1:x2]

                    msg = "Tracking face"

                    try:
                        resized_roi = cv2.resize(roi, (100, 100))
                    except:
                        continue
                    Id, conf = recognizer.predict(resized_roi)

                    # Skip the unknown prediction values
                    if Id == 0:
                        # Draw a red box over the face
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Put the text of unknown over the face
                        cv2.putText(im, "unknown", (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)

                        continue

                    if Id in df[df['ID'] == int(self.Id)].values:

                        name = df[df['ID'] == int(self.Id)]['NAME'].values
                        Id_and_name = str(self.Id) + " " + name

                        # draw a green rectangle over the face
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # show the result of the faces name and id over that face and their confidence
                        cv2.putText(im, str(Id_and_name), (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)
                        cv2.putText(im, "conf: " + str(int(conf)) + "%", (x1, y2 + 30), font, 0.9,
                                    (255, 255, 255), 2)
                        ts = time.time()
                        date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                        timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        report_df.loc[len(report_df)] = [self.Id, self.name, date, timeStamp, int(n)]

                    else:

                        # draw a read rectangle over the face
                        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # display the message of the face is unknown
                        cv2.putText(im, "unknown", (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)

                    report_df = report_df.drop_duplicates(subset=['ID'], keep='first')
            else:
                msg = " Searching for face"

            font = cv2.FONT_HERSHEY_TRIPLEX
            size = 0.7
            thickness = 1
            x_offset = 200

            # Bottom left corner message area
            cv2.putText(im, "Status:" + msg, (10, y - 20), font, size, (0, 255, 0), thickness)
            cv2.putText(im, "People Finder", (10, y - 40), font, size, (255, 255, 255), thickness)

            # Write the count of the images
            cv2.putText(im, fps_text, (10, 40), font, size, (0, 255, 0), thickness)

            # Write the count of the images
            cv2.putText(im, "Searching for:" + msg1, (10, 60), font, size, (0, 255, 0), thickness)

            # Write the current date of the images
            cv2.putText(im, str(current_date.date()), (x - x_offset, 40), font, size, (0, 255, 0), thickness)

            # Write the current date of the images
            cv2.putText(im, str(current_time), (x - x_offset, 60), font, size, (0, 255, 0), thickness)

            # Write the messages at the bottom left corner of the screen
            cv2.putText(im, "q to quit", (x - x_offset, y - 20), font, size, (0, 255, 0), thickness)
            cv2.putText(im, msg2, (x - x_offset, y - 40), font, size, (0, 255, 0), thickness)

            # display that frame
            cv2.imshow('webcam live video', im)

            # record if wanted
            if record:
                msg2 = "Recording..."
                # write the flipped frame
                vid_out.write(im)
            else:
                msg2 = "r to record"

            # wait for 100 milliseconds
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                if record:
                    record = False
                else:
                    record = True

        # Saving the report
        report_df.to_csv("Report/report.csv", index=False)

        # release the camera and the window will be destroyed
        cam.release()
        vid_out.release()
        cv2.destroyAllWindows()

    def Track_subject_on_video(self):

        if self.Id == "" and self.name == "":
            self.fill_obj.label.setText("Select a subject to track")
            self.fill_dia.show()
            return

        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # record flag
        record = False

        # message to display
        msg = " Searching for face"
        msg1 = self.Id + " " + self.name
        msg2 = "press r to record"

        # now load the files
        df = pd.read_csv("Subject Details/subjects_data.csv")
        path = QFileDialog.getOpenFileName(caption="Select the video to track the subject", directory="",
                                           filter="Video Files (*.mp4);;Video Files (*.webm);;Video Files (*.avi)",
                                           options=options)
        recognizer = My_model.AD_Model()

        # check for model exists or not
        if not os.path.exists("model/model.h5"):
            self.fill_obj.label.setText("No model trained")
            self.fill_dia.show()

            return

        # Load Report
        report_df = pd.read_csv("Report/report.csv")

        recognizer.load_model("model/model.h5")
        detector = dlib.get_frontal_face_detector()

        # check for the path is not empty
        if path[0] is "":
            self.fill_obj.label.setText("No video file selected")
            self.fill_dia.show()

            return

        video = cv2.VideoCapture(path[0])

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        n = len(os.listdir("Report"))
        vid_out = cv2.VideoWriter('Report/' + self.name + "." + self.Id + "." + str(n) + ".mp4", fourcc, 20.0,
                                  (int(w), int(h)))

        font = cv2.FONT_HERSHEY_SIMPLEX

        # To calculate the frame rate
        fps_start_time = datetime.now()
        total_frames = 0

        # Take sample flag
        record = False

        # message to display
        msg = "Message"
        msg1 = self.Id + " " + self.name
        msg2 = "r to record"

        try:
            # check while the video is paying window is open
            while video.isOpened():

                # read the data frame by frame
                ret, im = video.read()

                # coordinates of the screen
                y, x, _ = im.shape

                # To calculate the fps rate in the video from camera
                total_frames = total_frames + 1
                fps_end_time = datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)
                fps_text = "FPS: {:.2f}".format(fps)

                # Current time and date
                now = datetime.now()
                timestamp = datetime.timestamp(now)
                current_time = now.strftime("%H:%M:%S")
                current_date = datetime.fromtimestamp(timestamp)

                # while the frame is reading do all this stuff
                if ret:
                    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    # check for face exist in it or not
                    if len(faces) is not 0:

                        # get all the face Coordinates and draw a rectangle with message
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # not get the region of interest from the grayscale image
                            roi = gray[y1:y2, x1:x2]
                            resized_roi = None
                            Id = None
                            conf = 0

                            msg = "Tracking face"

                            try:
                                resized_roi = cv2.resize(roi, (100, 100))
                                Id, conf = recognizer.predict(resized_roi)
                            except:
                                continue

                            # get the confidence from the model
                            Id, conf = recognizer.predict(resized_roi)

                            if Id == 0:
                                # Draw a red box over the face
                                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                # Put the text of unknown over the face
                                cv2.putText(im, "unknown", (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)

                                continue

                            # check if the recognizer recognize any human already known
                            if conf > 90 and int(self.Id) == df["ID"][Id - 1]:

                                # get the nam and the id of the face from the csv file
                                name = df['NAME'][Id - 1]
                                Id_and_name = str(Id) + " " + name
                                print(Id_and_name)

                                # draw a green box over the face
                                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # show the result of the faces name and id over that face and their
                                # confidence
                                cv2.putText(im, str(Id_and_name), (x1, y2 + 50), font, 0.9, (255, 255, 255),
                                            2)
                                cv2.putText(im, "conf: " + str(int(conf)) + "%", (x1, y2 + 30), font, 0.9,
                                            (255, 255, 255), 2)
                                ts = time.time()
                                date = datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                                timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                                report_df.loc[len(report_df)] = [self.Id, self.name, date, timeStamp, int(n)]
                            else:

                                # Draw a red box over the face
                                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                # Put the text of unknown over the face
                                cv2.putText(im, "unknown", (x1, y2 + 50), font, 0.9, (255, 255, 255), 2)

                            report_df = report_df.drop_duplicates(subset=['ID'], keep='first')
                    else:
                        msg = " Searching for face"

                    font = cv2.FONT_HERSHEY_TRIPLEX
                    size = 0.7
                    thickness = 1
                    x_offset = 200

                    # Bottom left corner message area
                    cv2.putText(im, "Status:" + msg, (10, y - 20), font, size, (0, 255, 0), thickness)
                    cv2.putText(im, "People Finder", (10, y - 40), font, size, (255, 255, 255), thickness)

                    # Write the count of the images
                    cv2.putText(im, fps_text, (10, 40), font, size, (0, 255, 0), thickness)

                    # Write the count of the images
                    cv2.putText(im, "Searching for:" + msg1, (10, 60), font, size, (0, 255, 0), thickness)

                    # Write the current date of the images
                    cv2.putText(im, str(current_date.date()), (x - x_offset, 40), font, size, (0, 255, 0), thickness)

                    # Write the current date of the images
                    cv2.putText(im, str(current_time), (x - x_offset, 60), font, size, (0, 255, 0), thickness)

                    # Write the messages at the bottom left corner of the screen
                    cv2.putText(im, "q to quit", (x - x_offset, y - 20), font, size, (0, 255, 0), thickness)
                    cv2.putText(im, msg2, (x - x_offset, y - 40), font, size, (0, 255, 0), thickness)

                    # Display the video frame
                    cv2.imshow('Video', im)

                    # record if wanted
                    if record:
                        msg2 = "Recording..."
                        # write the flipped frame
                        vid_out.write(im)
                    else:
                        msg2 = "r to record"

                    # wait for 100 milliseconds
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        if record:
                            record = False
                        else:
                            record = True
                else:
                    break

        except:
            pass

        # Saving the report
        report_df.to_csv("Report/report.csv", index=False)

        video.release()
        cv2.destroyAllWindows()

    def getImagesAndLabels(self, path):

        # List all the folders in the Subject Images folder
        categories = sorted(os.listdir(path))

        # Initialize the sets and image size
        img_size = 100
        data = []
        target = []

        for category in categories:

            folder_path = os.path.join(path, category)
            img_names = os.listdir(folder_path)

            for img_name in img_names:
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)

                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Converting the image into gray scale
                    resized = cv2.resize(gray, (img_size, img_size))

                    data.append(resized)
                    target.append(category)
                    # appending the image and the label(categorized) into the list (dataset)

                except Exception as e:
                    print('Exception:', e)
        return data, target

    def StartTraining(self):

        # Here we can extract all the images and their ids
        faces, Ids = self.getImagesAndLabels("Subject Images/")

        # Here we start the training of the model
        self.trainer_model.train(faces, Ids, int(self.main_obj.epochs.text()))

        # After training save that model
        self.trainer_model.save_model("model/model.h5")

        # Break the thread
        global thread_break
        thread_break = True

        # Reset the image and progress bar
        self.main_obj.label_graph.setPixmap(QtGui.QPixmap('output/new.jpg'))
        self.main_obj.progressBar4.setValue(0)

        # reset the flag of thread
        thread_break = False

        # Start threading of training
        self.graph_thread = Thread()
        self.training_thread = threading.Thread(target=self.StartTraining)

    def existing_item_clicked(self):

        # Select that activated row
        self.row = self.main_obj.table.selectedItems()

        # Update the data
        self.Id = self.row[0].text()
        self.name = self.row[1].text()
        self.age = self.row[2].text()
        self.gender = self.row[3].text()
        self.phone = self.row[4].text()
        self.address = self.row[5].text()

        # update the data over the Train window
        self.main_obj.id_label4.setText(self.Id)
        self.main_obj.name_label4.setText(self.name)
        self.main_obj.age_label4.setText(self.age)
        self.main_obj.gender_label4.setText(self.gender)
        self.main_obj.ph_label4.setText(self.phone)
        self.main_obj.address_label4.setText(self.address)
        self.main_obj.pic4.setPixmap(QtGui.QPixmap("known people/" + self.name + "." + self.Id + ".jpg"))

        # update the data over the Track window
        self.main_obj.id_label5.setText(self.Id)
        self.main_obj.name_label5.setText(self.name)
        self.main_obj.age_label5.setText(self.age)
        self.main_obj.gender_label5.setText(self.gender)
        self.main_obj.ph_label5.setText(self.phone)
        self.main_obj.address_label5.setText(self.address)
        self.main_obj.pic5.setPixmap(QtGui.QPixmap("known people/" + self.name + "." + self.Id + ".jpg"))

        # Update the image over the Report Window
        self.main_obj.pic6.setPixmap(QtGui.QPixmap("known people/" + self.name + "." + self.Id + ".jpg"))

    def check_Id_exists(self, check_id):
        if os.path.exists("Subject Details/subjects_data.csv"):
            with open('Subject Details/subjects_data.csv', 'r') as f:
                d_reader = csv.DictReader(f)
                for line in d_reader:
                    if str(line['ID']) == str(check_id):
                        return True
                    else:
                        continue
                return False

    def check_name_exists(self, check_name):
        if os.path.exists("Subject Details/subjects_data.csv"):
            with open('Subject Details/subjects_data.csv', 'r') as f:
                d_reader = csv.DictReader(f)
                for line in d_reader:
                    if str(line['NAME']) == str(check_name):
                        return True
                    else:
                        continue
                return False

    def sample_from_camera(self):

        # Store the name and Id for the face Sample
        self.name = self.main_obj.name.text().upper()
        self.age = self.main_obj.age.text().upper()
        self.phone = self.main_obj.ph_number.text().upper()
        self.address = self.main_obj.address.text().upper()

        file = open("Subject Details/subjects_data.csv")
        reader = csv.reader(file)
        self.Id = str(len(list(reader)))

        if self.main_obj.female.isChecked():
            self.gender = "FEMALE"
        elif self.main_obj.male.isChecked():
            self.gender = "MALE"

        # Check for the inputted name is exist or not
        if not self.check_name_exists(self.name):

            # Check for the validation of the inputted value are correct or not
            if self.age.isdigit() and self.phone.isdigit() and self.address.isalpha():

                # initialize the Variables
                sample_image_Captured = False
                cam = cv2.VideoCapture(0)
                fps_start_time = datetime.now()
                total_frames = 0
                detector = dlib.get_frontal_face_detector()
                sampleNum = 0

                # Take sample flag
                take_sample = False

                try:
                    # start the loop for the image capturing and all the rest of stuff
                    while True:
                        ret, img = cam.read()

                        # corrdinates of the screen
                        x, y, _ = img.shape

                        # Current time and date
                        now = datetime.now()
                        timestamp = datetime.timestamp(now)
                        current_time = now.strftime("%H:%M:%S")
                        current_date = datetime.fromtimestamp(timestamp)

                        # To calculate the fps rate in the video from camera
                        total_frames = total_frames + 1
                        fps_end_time = datetime.now()
                        time_diff = fps_end_time - fps_start_time
                        if time_diff.seconds == 0:
                            fps = 0.0
                        else:
                            fps = (total_frames / time_diff.seconds)
                        fps_text = "FPS: {:.2f}".format(fps)

                        # message to display
                        msg = "Message"

                        # Create the grayscale image for the face classification
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = detector(gray)

                        if len(faces) is not 0:
                            # Draw a box around the face in each frame
                            for face in faces:
                                x1 = face.left()
                                y1 = face.top()
                                x2 = face.right()
                                y2 = face.bottom()

                                # Check for Sample image is Captured or not
                                if not sample_image_Captured:
                                    new = cv2.resize(img[y1:y2, x1:x2], (219, 230))
                                    cv2.imwrite("known people/" + self.name + "." + self.Id + ".jpg", new)
                                    sample_image_Captured = True

                                # draw a rectangle around the face that is detected
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                                # resize sample dimension
                                roi = img[y1: y2, x1: x2]
                                roi = cv2.resize(roi, (100, 100))

                                # update the image features
                                from keras.preprocessing.image import ImageDataGenerator
                                image_gen = ImageDataGenerator(rotation_range=30,  # rotate the image 30 degrees
                                                               width_shift_range=0.1,
                                                               # Shift the pic width by a max of 10%
                                                               height_shift_range=0.1,
                                                               # Shift the pic height by a max of 10%
                                                               rescale=1 / 255,  # Rescale the image by normalzing it.
                                                               shear_range=0.2,
                                                               # Shear means cutting away part of the image (max 20%)
                                                               zoom_range=0.2,  # Zoom in by 20% max
                                                               horizontal_flip=True,  # Allo horizontal flipping
                                                               fill_mode='nearest'
                                                               # Fill in missing pixels with the nearest filled value
                                                               )
                                roi = image_gen.random_transform(roi)

                                if take_sample:
                                    # saving the captured face in the dataset folder TrainingImage
                                    if os.path.exists("Subject Images/" + self.Id):
                                        cv2.imwrite(
                                            "Subject Images/" + self.Id + "/" + self.name + "." + self.Id + "." + str(
                                                sampleNum) + ".jpg", roi)
                                    else:
                                        os.mkdir("Subject Images/" + self.Id)
                                        cv2.imwrite(
                                            "Subject Images/" + self.Id + "/" + self.name + "." + self.Id + "." + str(
                                                sampleNum) + ".jpg", roi)
                                    # m essage to display
                                    msg = " Taking samples of face"

                                    # incrementing sample number
                                    sampleNum = sampleNum + 1
                                else:
                                    msg = " Tracking face"
                                    take_sample = False
                        else:
                            msg = " Searching for face"

                        font = cv2.FONT_HERSHEY_TRIPLEX
                        size = 0.7
                        thickness = 1

                        # Bottom left corner message area
                        cv2.putText(img, "Status:" + msg, (10, y - 200), font, size, (0, 255, 0), thickness)
                        cv2.putText(img, "People Finder", (10, y - 230), font, size, (255, 255, 255), thickness)

                        # Write the count of the images
                        cv2.putText(img, fps_text, (10, 40), font, size, (0, 255, 0), thickness)

                        # Write the count of the images
                        cv2.putText(img, "Samples:" + str(sampleNum), (10, 60), font, size, (0, 255, 0), thickness)

                        # Write the current date of the images
                        cv2.putText(img, str(current_date.date()), (x - 50, 40), font, size, (0, 255, 0), thickness)

                        # Write the current date of the images
                        cv2.putText(img, str(current_time), (x - 50, 60), font, size, (0, 255, 0), thickness)

                        # Write the messages at the bottom left corner of the screen
                        cv2.putText(img, "q to quit", (x - 50, y - 200), font, size, (0, 255, 0), thickness)
                        cv2.putText(img, "t for samples", (x - 50, y - 230), font, size, (0, 255, 0), thickness)

                        # display the frame
                        cv2.imshow("Sample Taking Window", img)

                        # wait for 100 miliseconds
                        key = cv2.waitKey(1)
                        if key == ord('q'):
                            break
                        elif key == ord('t'):
                            take_sample = True
                        else:
                            take_sample = False

                        # break if the sample number is more than 200
                        if sampleNum == 200:
                            break
                except:
                    pass
                # Set the flag that the samples are taken or not
                self.sample_taken = True

                # Release the camera and close the window
                cam.release()
                cv2.destroyAllWindows()

            else:
                self.fill_obj.label.setText("Invalid input values")
                self.fill_dia.show()
        else:
            self.fill_obj.label.setText("Name already exists")
            self.fill_dia.show()

    def sample_from_gallery(self):

        # open the dialogue box to select the file
        options = QtWidgets.QFileDialog.Options()

        # Check that the sample image is captured or not
        sample_image_captured = False

        # Store the name and Id for the face Sample
        self.name = self.main_obj.name.text().upper()
        self.age = self.main_obj.age.text().upper()
        self.phone = self.main_obj.ph_number.text().upper()
        self.address = self.main_obj.address.text().upper()

        if self.main_obj.female.isChecked():
            self.gender = "FEMALE"
        elif self.main_obj.male.isChecked():
            self.gender = "MALE"

        file = open("Subject Details/subjects_data.csv")
        reader = csv.reader(file)
        self.Id = str(len(list(reader)))

        # Check for name already exists or not
        if self.check_name_exists(self.name):
            self.fill_obj.label.setText("Name already exists")
            self.fill_dia.show()
            return

            # Now Check the inputted values are empty or not
        if not self.age.isdigit() and not self.name.isalpha() and not self.phone.isdecimal():
            self.fill_obj.label.setText("Invalid input values")
            self.fill_dia.show()
            return

        # open the Dialogue box to get the images paths
        images = QtWidgets.QFileDialog.getOpenFileNames(caption="Select the images of the subject", directory="",
                                                        filter="Image Files (*.jpg);;Image Files (*.png)",
                                                        options=options)

        # Check if the images are taken or not
        if len(images[0]) == 0:
            self.fill_obj.label.setText("No images provided")
            self.fill_dia.show()
            return

        detector = dlib.get_frontal_face_detector()
        count = 0

        try:
            for image in images[0]:
                image = cv2.imread(image)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                if count == 200:
                    break
                if len(faces) != 0:
                    for face in faces:
                        x1 = face.left()
                        y1 = face.top()
                        x2 = face.right()
                        y2 = face.bottom()

                        if not sample_image_captured:
                            new = cv2.resize(image[y1: y2, x1: x2], (219, 230))
                            cv2.imwrite("known people/" + self.name + "." + self.Id + ".jpg", new)
                            sample_image_captured = True
                        roi = image[y1: y2, x1: x2]
                        roi = cv2.resize(roi, (100, 100))

                        # Save the subject image in a new folder
                        if os.path.exists("Subject Images/" + self.Id):
                            cv2.imwrite(
                                "Subject Images/" + self.Id + "/" + self.name + "." + self.Id + "." + str(
                                    count) + ".jpg", roi)

                        else:
                            os.mkdir("Subject Images/" + self.Id)
                            cv2.imwrite(
                                "Subject Images/" + self.Id + "/" + self.name + "." + self.Id + "." + str(
                                    count) + ".jpg", roi)

                        count += 1
        except:
            pass

        # Set the flag that the samples are taken or not
        self.sample_taken = True

    def train_progress_bar(self):

        global counter
        # todo setting progress bar value
        self.main_obj.progressBar4.setValue(counter)
        self.main_obj.progressBar4.setTextVisible(True)
        if counter > 100:
            self.timer2.stop()
            counter = 0
            self.main_obj.progressBar4.setValue(counter)
            self.main_obj.progressBar4.setTextVisible(False)
        counter += 1

    def start(self):
        self.timer2 = QtCore.QTimer()
        self.timer2.start(100)
        self.timer2.timeout.connect(self.train_progress_bar)

    def loader_progress_bar(self):
        # Counter is initialized here
        global counter

        # todo setting progress bar value
        self.loading_obj.progressBar.setValue(counter)
        if counter > 100:
            self.timer.stop()
            counter = 0

            # todo open app and close loader
            self.loading_window.close()
            self.main_window.show()
        counter += 1

    def load_data(self):
        if self.check():  # calls the check function

            # todo reading file
            self.file_data = self.read_file()

            # todo writing in table
            self.main_obj.table.setRowCount(0)
            self.file_data = iter(self.file_data)
            next(self.file_data)
            for row, rd in enumerate(self.file_data):
                self.main_obj.table.insertRow(row)
                for col, data in enumerate(rd):
                    self.main_obj.table.setItem(row, col, QtWidgets.QTableWidgetItem(str(data)))
            del self.file_data
        else:
            return

    def write_file(self, file_data):
        with open("Subject Details/subjects_data.csv", "w") as f:
            wo = csv.writer(f, lineterminator='\n')
            wo.writerows(self.file_data)
            del wo
        # todo load the file into table
        self.load_data()

    def read_file(self):

        self.file_data = list()

        with open("Subject Details/subjects_data.csv") as f:
            self.ro = csv.reader(f)
            for x in self.ro:
                self.file_data.append(x)
            del self.ro
        return self.file_data

    def insert(self):
        self.check()
        self.row = list()

        # todo taking values and inserting new person
        self.row.append(self.main_obj.name.text().upper())
        self.row.append(self.main_obj.age.text().upper())
        if self.main_obj.male.isChecked():
            self.row.append("MALE")
        elif self.main_obj.female.isChecked():
            self.row.append("FEMALE")
        self.row.append(self.main_obj.ph_number.text().upper())
        self.row.append(self.main_obj.address.text().upper())

        # todo checking validation
        if "" in self.row:
            self.fill_obj.label.setText("Please fill all the field(s)")
            self.fill_dia.show()

            return

        elif not self.sample_taken:

            self.fill_obj.label.setText("Samples are not taken")
            self.fill_dia.show()
            return
        else:
            # todo read file data
            self.file_data = self.read_file()

            # todo auto increment in ID
            if len(self.file_data) == 1:
                self.row.insert(0, 1)
            else:
                self.row.insert(0, int(self.file_data[len(self.file_data) - 1][0]) + 1)

            # todo appending row in file
            with open("Subject Details/subjects_data.csv", "a") as f:
                self.wo = csv.writer(f, lineterminator='\n')
                self.wo.writerow(self.row)
                del self.wo

            # todo releasing memory
            del self.row
            del self.file_data
            self.main_obj.name.clear()
            self.main_obj.age.clear()
            self.main_obj.ph_number.clear()
            self.main_obj.address.clear()

            # todo calls the load function for updating
            self.load_data()

            self.sample_taken = False

    def search(self):
        self.txt = self.main_obj.search_bar.text()

        # todo if nothing is searched then load the whole file
        if self.txt == "":
            self.load_data()
            return

        else:
            # todo reading file
            self.file_data = self.read_file()

            self.searched_data = list()

            # todo matching values and saving the matched ones (searching)
            for x in self.file_data:
                if self.txt.casefold() in x[1].casefold() and x[1] != "NAME":
                    # if x[1].casefold().startswith(self.txt.casefold()):
                    self.searched_data.append(x)

            # todo writing in table
            self.main_obj.table.setRowCount(0)
            for row, rd in enumerate(self.searched_data):
                self.main_obj.table.insertRow(row)
                for col, data in enumerate(rd):
                    self.main_obj.table.setItem(row, col, QtWidgets.QTableWidgetItem(data))

            # todo releasing memory
            del self.file_data
            del self.searched_data
            del self.txt

    def make_sure(self):
        if self.main_obj.table.selectedItems() != []:
            self.del_dia.show()
            self.del_obj.del_buttonBox.accepted.connect(self.delete_row)
        else:
            self.select_dia.show()

    def delete_row(self):
        if self.main_obj.table.selectedItems() != []:

            # todo selected row
            self.row = self.main_obj.table.selectedItems()

            # todo ID number
            self.index = self.row[0].text()
            self.name = self.row[1].text()

            # remove the image from known folder
            if os.path.exists("known people/" + self.name + "." + self.index + ".jpg"):
                os.remove("known people/" + self.name + "." + self.index + ".jpg")

                # Show the deleted message
                print("known people/" + self.name + "." + self.index + ".jpg Deleted")

            # remove the subject images folder
            if os.path.exists("Subject Images/" + self.Id):
                # first we need to empty the folder
                for image_path in os.listdir("Subject Images/" + self.Id):
                    os.remove("Subject Images/" + self.Id + "/" + image_path)
                # After that we remove the folder
                os.rmdir("Subject Images/" + self.Id)

            # todo reading data from file.
            self.file_data = self.read_file()

            # todo find the selected one and delete it.
            for i, x in enumerate(self.file_data):

                # todo both are strings.
                if self.index == x[0]:
                    self.deleted_row = self.file_data.pop(i)
                    break
                else:
                    pass

            # todo overwrite the file
            self.write_file(self.file_data)

            # reset the values
            self.Id = None
            self.name = None
            self.age = None
            self.gender = None
            self.phone = None
            self.address = None

            # reset the data over the Train window
            self.main_obj.id_label4.setText(self.Id)
            self.main_obj.name_label4.setText(self.name)
            self.main_obj.age_label4.setText(self.age)
            self.main_obj.gender_label4.setText(self.gender)
            self.main_obj.ph_label4.setText(self.phone)
            self.main_obj.address_label4.setText(self.address)
            self.main_obj.pic4.setPixmap(QtGui.QPixmap("images/avatar.jpeg"))

            # reset the data over the Track window
            self.main_obj.id_label5.setText(self.Id)
            self.main_obj.name_label5.setText(self.name)
            self.main_obj.age_label5.setText(self.age)
            self.main_obj.gender_label5.setText(self.gender)
            self.main_obj.ph_label5.setText(self.phone)
            self.main_obj.address_label5.setText(self.address)
            self.main_obj.pic5.setPixmap(QtGui.QPixmap("images/avatar.jpeg"))

            # Update the image over the Report Window
            self.main_obj.pic6.setPixmap(QtGui.QPixmap("images/avatar.jpeg"))

            return self.deleted_row

    def open_update_win(self):
        if self.main_obj.table.selectedItems() != []:
            self.row = self.main_obj.table.selectedItems()
            # todo fill exixting values
            self.update_obj.updated_name.setText(self.row[1].text())
            self.update_obj.updated_age.setText(self.row[2].text())
            if self.row[3].text()=="MALE":
               self.update_obj.updated_gender.setCurrentIndex(0)
            else:
                self.update_obj.updated_gender.setCurrentIndex(1)
            #self.update_obj.updated_gender.setText(self.row[3].text())
            self.update_obj.updated_number.setText(self.row[4].text())
            self.update_obj.updated_address.setText(self.row[5].text())

            self.update_window.show()
            self.update_obj.update.clicked.connect(self.update_values)
        else:
            self.select_dia.show()
            return

    def update_values(self):
        if self.main_obj.table.selectedItems() != []:
            self.row = list()
            # todo making list of updated values
            self.row.append(self.main_obj.table.selectedItems()[0].text())
            self.row.append(self.update_obj.updated_name.text().upper())
            self.row.append(self.update_obj.updated_age.text().upper())
            self.row.append(self.update_obj.updated_gender.currentText().upper())
            #self.row.append(self.update_obj.updated_gender.text().upper())
            self.row.append(self.update_obj.updated_number.text().upper())
            self.row.append(self.update_obj.updated_address.text().upper())

            # todo checking validation
            if "" in self.row:
                self.fill_dia.show()
                return
            else:
                self.file_data = self.read_file()

                for i, x in enumerate(self.file_data):
                    if x[0] == self.row[0]:
                        self.file_data[i] = self.row
                        break
                    else:
                        pass
                # todo overwrite the file
                self.write_file(self.file_data)

                self.update_window.close()

    def check(self):
        if not os.path.exists("Subject Details/subjects_data.csv"):
            with open("Subject Details/subjects_data.csv", "w") as f:
                self.wo = csv.writer(f, lineterminator='\n')
                self.wo.writerow(["ID", "NAME", "AGE", "GENDER", "PHONE NO#", "ADDRESS"])
                del self.wo
            return False
        else:
            return True

    def main_function(self):
        # todo        ____________  buttons connections  ____________
        """indexes
        new=1
        select=2
        train=3
        track=4
        report=5
        """
        # todo page1
        # todo with current widget
        self.main_obj.select1.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentWidget(self.main_obj.page3))
        # todo with current index
        self.main_obj.new1.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))

        # todo page2
        self.main_obj.new2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))
        self.main_obj.select2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))
        self.main_obj.train2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))
        self.main_obj.track2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))
        self.main_obj.report2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(5))
        self.main_obj.back2.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))  # back to select

        # todo page3
        self.main_obj.new3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))
        self.main_obj.select3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))
        self.main_obj.train3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))
        self.main_obj.track3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))
        self.main_obj.report3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(5))
        self.main_obj.back3.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(0))  # back to start

        # todo page4
        self.main_obj.new4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))
        self.main_obj.select4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))
        self.main_obj.train4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))
        self.main_obj.track4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))
        self.main_obj.report4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(5))
        self.main_obj.back4.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))  # back to new

        # todo page5
        self.main_obj.new5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))
        self.main_obj.select5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))
        self.main_obj.train5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))
        self.main_obj.track5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))
        self.main_obj.report5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(5))
        self.main_obj.back5.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))  # back to train

        # todo page6
        self.main_obj.new6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(1))
        self.main_obj.select6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(2))
        self.main_obj.train6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(3))
        self.main_obj.track6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))
        self.main_obj.report6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(5))
        self.main_obj.back6.clicked.connect(lambda: self.main_obj.stackedWidget.setCurrentIndex(4))  # back to track

        # todo        ____________  submitting new person  ____________
        self.main_obj.submit.clicked.connect(self.insert)

        # todo        ____________  searching  ____________
        self.main_obj.search_bar.textChanged.connect(self.search)

        # todo        ____________  deleting  ____________
        self.main_obj.delete3.clicked.connect(self.make_sure)

        # todo        ____________  updating  ____________
        self.main_obj.update3.clicked.connect(self.open_update_win)




if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    obj = Main()
    obj.main_function()
    sys.exit((app.exec_()))
