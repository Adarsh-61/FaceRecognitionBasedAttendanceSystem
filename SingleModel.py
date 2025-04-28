"""
Face Recognition Based Attendance System - Enhanced Version
This module provides an optimized implementation of a face recognition-based attendance system
with improved performance, code structure, and UI responsiveness.
"""

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2
import os
import csv
import subprocess
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import threading
import logging
import queue
from pathlib import Path
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("attendance_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger("FaceAttendance")

HAAR_CASCADE_FILE = "haarcascade_frontalface_default.xml"
TRAINING_IMAGE_DIR = "TrainingImage"
TRAINING_LABEL_DIR = "TrainingImageLabel"
STUDENT_DETAILS_DIR = "StudentDetails"
ATTENDANCE_DIR = "Attendance"
STUDENT_DETAILS_FILE = os.path.join(STUDENT_DETAILS_DIR, "StudentDetails.csv")
PASSWORD_FILE = os.path.join(TRAINING_LABEL_DIR, "psd.txt")
TRAINER_FILE = os.path.join(TRAINING_LABEL_DIR, "Trainner.yml")

for directory in [
    TRAINING_IMAGE_DIR,
    TRAINING_LABEL_DIR,
    STUDENT_DETAILS_DIR,
    ATTENDANCE_DIR,
]:
    Path(directory).mkdir(exist_ok=True, parents=True)


class FaceRecognitionSystem:
    """Main class for the Face Recognition Attendance System"""

    def __init__(self, window):
        """Initialize the Face Recognition System with the main window"""
        self.window = window

        self.queue = queue.Queue()

        self.setup_window()
        self.setup_ui()
        self.load_registration_count()

        self.face_cascade = None
        self.recognizer = None
        self.initialize_face_recognition()

        self.master = None
        self.old = None
        self.new = None
        self.nnew = None

        self.process_queue()

    def process_queue(self):
        """Process the queue for UI updates from background threads"""
        try:
            while True:
                task, *args = self.queue.get_nowait()

                if task == "update_message":
                    self.message1.configure(text=args[0])
                elif task == "update_registration_count":
                    self.load_registration_count()
                elif task == "add_to_treeview":
                    self.tv.insert("", 0, text=args[0], values=args[1:])

                self.queue.task_done()
        except queue.Empty:
            pass
        finally:
            self.window.after(100, self.process_queue)

    def setup_window(self):
        """Configure the main window properties"""
        self.window.geometry("1920x1080")
        self.window.resizable(True, False)
        self.window.title("Face Recognition Based Attendance System Developed By A&J")
        self.window.configure(background="#ffff00")

        self.setup_menu()

    def setup_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(
            self.window,
            relief="flat",
            bg="#333",
            fg="#fff",
            activebackground="#555",
            activeforeground="#fff",
        )

        filemenu = tk.Menu(
            menubar,
            tearoff=0,
            bg="#444",
            fg="#fff",
            activebackground="#666",
            activeforeground="#fff",
        )

        filemenu.add_command(label="Other Models", command=self.run_multimodel)
        filemenu.add_command(label="Change Password", command=self.change_pass)
        filemenu.add_command(label="Contact Us", command=self.contact)
        filemenu.add_command(label="Exit", command=self.window.destroy)

        menubar.add_cascade(label="Help", font=("comic", 12, "bold"), menu=filemenu)
        self.window.configure(menu=menubar)

    def setup_ui(self):
        """Set up the user interface components"""
        message3 = tk.Label(
            self.window,
            text="Face Recognition Based Attendance System",
            fg="white",
            bg="#ff0000",
            width=60,
            height=1,
            font=("comic", 29, "bold"),
        )
        message3.place(x=40, y=10)

        self.setup_date_time()

        self.setup_frames()

        self.setup_treeview()

        self.setup_controls()

        copyWrite = tk.Text(
            self.window,
            background=self.window.cget("background"),
            borderwidth=0,
            font=("comic", 20, "bold"),
        )
        copyWrite.tag_configure("superscript", offset=5)
        copyWrite.insert("insert", "Developed By A&J", "", "", "superscript")
        copyWrite.configure(state="disabled", fg="blue")
        copyWrite.pack(side="left")
        copyWrite.place(x=630, y=730)

    def setup_date_time(self):
        """Set up date and time display"""
        ts = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        day, month, year = date_str.split("-")

        mont = {
            "01": "Jan",
            "02": "Feb",
            "03": "Mar",
            "04": "Apr",
            "05": "May",
            "06": "Jun",
            "07": "Jul",
            "08": "Aug",
            "09": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }

        frame4 = tk.Frame(self.window, bg="#c4c6ce")
        frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

        datef = tk.Label(
            frame4,
            text=day + "-" + mont[month] + "-" + year + "   &",
            fg="#0000ff",
            bg="#ffff00",
            width=55,
            height=1,
            font=("comic", 22, "bold"),
        )
        datef.pack(fill="both", expand=1)

        frame3 = tk.Frame(self.window, bg="#c4c6ce")
        frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

        self.clock = tk.Label(
            frame3,
            fg="#0000ff",
            bg="#ffff00",
            width=55,
            height=1,
            font=("comic", 22, "bold"),
        )
        self.clock.pack(fill="both", expand=1)
        self.tick()

    def setup_frames(self):
        """Set up the main content frames"""
        self.frame1 = tk.Frame(self.window, bg="#a05fbf")
        self.frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.70)

        head1 = tk.Label(
            self.frame1,
            text="                       For Already Registered                       ",
            fg="#ffffff",
            bg="#ff1493",
            font=("comic", 17, "bold"),
        )
        head1.place(x=0, y=0)

        self.frame2 = tk.Frame(self.window, bg="#a05fbf")
        self.frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.70)

        head2 = tk.Label(
            self.frame2,
            text="                       For New Registrations                       ",
            fg="#ffffff",
            bg="#ff1493",
            font=("comic", 17, "bold"),
        )
        head2.grid(row=0, column=0)

        lbl = tk.Label(
            self.frame2,
            text="Enter ID",
            width=20,
            height=1,
            fg="black",
            bg="#a05fbf",
            font=("comic", 17, "bold"),
        )
        lbl.place(x=80, y=55)

        self.txt = tk.Entry(
            self.frame2, width=32, fg="black", font=("comic", 15, "bold")
        )
        self.txt.place(x=30, y=88)

        lbl2 = tk.Label(
            self.frame2,
            text="Enter Name",
            width=20,
            fg="black",
            bg="#a05fbf",
            font=("comic", 17, "bold"),
        )
        lbl2.place(x=80, y=140)

        self.txt2 = tk.Entry(
            self.frame2, width=32, fg="black", font=("comic", 15, "bold")
        )
        self.txt2.place(x=30, y=173)

        self.message1 = tk.Label(
            self.frame2,
            text="1)Take Images  >>>  2)Save Profile",
            bg="#a05fbf",
            fg="black",
            width=39,
            height=1,
            activebackground="#3ffc00",
            font=("comic", 15, "bold"),
        )
        self.message1.place(x=7, y=230)

        self.message = tk.Label(
            self.frame2,
            text="",
            bg="#a05fbf",
            fg="black",
            width=39,
            height=1,
            activebackground="#3ffc00",
            font=("comic", 16, "bold"),
        )
        self.message.place(x=7, y=450)

        lbl3 = tk.Label(
            self.frame1,
            text="Attendance",
            width=20,
            fg="black",
            bg="#a05fbf",
            height=1,
            font=("comic", 17, "bold"),
        )
        lbl3.place(x=100, y=115)

    def setup_treeview(self):
        """Set up the attendance treeview"""
        style = ttk.Style()
        style.configure("Treeview", font=("comic", 11), rowheight=25)
        style.configure(
            "Treeview.Heading", font=("comic", 12, "bold"), background="#a05fbf"
        )

        self.tv = ttk.Treeview(self.frame1, height=13, columns=("name", "date", "time"))
        self.tv.column("#0", width=82)
        self.tv.column("name", width=130)
        self.tv.column("date", width=133)
        self.tv.column("time", width=133)
        self.tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
        self.tv.heading("#0", text="ID")
        self.tv.heading("name", text="NAME")
        self.tv.heading("date", text="DATE")
        self.tv.heading("time", text="TIME")

        scroll = ttk.Scrollbar(self.frame1, orient="vertical", command=self.tv.yview)
        scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky="ns")
        self.tv.configure(yscrollcommand=scroll.set)

    def setup_controls(self):
        """Set up control buttons with original styling"""
        clearButton = tk.Button(
            self.frame2,
            text="Clear",
            command=self.clear,
            fg="black",
            bg="#ff4500",
            width=11,
            activebackground="skyblue",
            font=("comic", 11, "bold"),
        )
        clearButton.place(x=335, y=86)

        clearButton2 = tk.Button(
            self.frame2,
            text="Clear",
            command=self.clear2,
            fg="black",
            bg="#ff4500",
            width=11,
            activebackground="skyblue",
            font=("comic", 11, "bold"),
        )
        clearButton2.place(x=335, y=172)

        takeImg = tk.Button(
            self.frame2,
            text="Take Images",
            command=self.take_images,
            fg="white",
            bg="#6d00fc",
            width=34,
            height=1,
            activebackground="white",
            font=("comic", 15, "bold"),
        )
        takeImg.place(x=30, y=300)

        trainImg = tk.Button(
            self.frame2,
            text="Save Profile",
            command=self.psw,
            fg="white",
            bg="#339933",
            width=34,
            height=1,
            activebackground="white",
            font=("comic", 15, "bold"),
        )
        trainImg.place(x=30, y=380)

        trackImg = tk.Button(
            self.frame1,
            text="Take Attendance",
            command=self.track_images,
            fg="black",
            bg="#3ffc00",
            width=35,
            height=1,
            activebackground="white",
            font=("comic", 15, "bold"),
        )
        trackImg.place(x=30, y=50)

        quitWindow = tk.Button(
            self.frame1,
            text="Quit",
            command=self.window.destroy,
            fg="black",
            bg="#ff0000",
            width=35,
            height=1,
            activebackground="white",
            font=("comic", 15, "bold"),
        )
        quitWindow.place(x=30, y=450)

    def tick(self):
        """Update the clock display"""
        time_string = time.strftime("%H:%M:%S")
        self.clock.config(text=time_string)
        self.clock.after(200, self.tick)

    def contact(self):
        """Display contact information"""
        mess._show(
            title="Contact Us",
            message="Please Contact Us At :- 'pandeyadarsh3115@gmail.com'",
        )

    def initialize_face_recognition(self):
        """Initialize face recognition components"""
        if not os.path.isfile(HAAR_CASCADE_FILE):
            mess._show(
                title="Some Files Are Missing",
                message="Please download haarcascade_frontalface_default.xml from OpenCV GitHub repository!",
            )
            logger.error(f"Haar cascade file not found: {HAAR_CASCADE_FILE}")
            return False

        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_FILE)

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        logger.info("Face recognition components initialized successfully")
        return True

    def load_registration_count(self):
        """Load and display the current registration count"""
        res = 0
        if os.path.isfile(STUDENT_DETAILS_FILE):
            try:
                with open(STUDENT_DETAILS_FILE, "r") as csvFile1:
                    reader1 = csv.reader(csvFile1)
                    for l in reader1:
                        res = res + 1
                res = (res // 2) - 1
                logger.info(f"Loaded {res} registrations from {STUDENT_DETAILS_FILE}")
            except Exception as e:
                logger.error(f"Error loading registration count: {e}")
                res = 0
        else:
            logger.info(f"Student details file not found: {STUDENT_DETAILS_FILE}")
            res = 0

        self.message.configure(text=f"Total Registrations Till Now  : {res}")

    def update_registration_count(self):
        """Update the registration count display"""
        self.queue.put(("update_registration_count",))

    def clear(self):
        """Clear ID input field"""
        self.txt.delete(0, "end")
        self.message1.configure(text="1)Take Images  >>>  2)Save Profile")

    def clear2(self):
        """Clear Name input field"""
        self.txt2.delete(0, "end")
        self.message1.configure(text="1)Take Images  >>>  2)Save Profile")

    def check_password(self):
        """Check if password exists and verify it"""
        if os.path.isfile(PASSWORD_FILE):
            with open(PASSWORD_FILE, "r") as tf:
                key = tf.read()
            return key
        else:
            return self.create_new_password()

    def create_new_password(self):
        """Create a new password if none exists"""
        new_pas = tsd.askstring(
            "Old Password Not Found!", "Please Enter New Password Below!", show="*"
        )
        if new_pas is None:
            mess._show(
                title="No Password Was Entered",
                message="Password Not Set! Please Try Again!",
            )
            return None
        else:
            with open(PASSWORD_FILE, "w") as tf:
                tf.write(new_pas)
            mess._show(
                title="Password Registered",
                message="New Password Registered Successfully!!",
            )
            return new_pas

    def psw(self):
        """Password verification before saving profile"""
        key = self.check_password()
        if key is None:
            return

        password = tsd.askstring("Password", "Enter Password:", show="*")
        if password == key:
            self.train_images()
        elif password is None:
            pass
        else:
            mess._show(title="Wrong Password", message="You Entered Wrong Password!")

    def change_pass(self):
        """Change the password"""
        self.master = tk.Toplevel(self.window)
        self.master.geometry("450x165")
        self.master.resizable(False, False)
        self.master.title("Change Password")
        self.master.configure(background="white")

        lbl4 = tk.Label(
            self.master,
            text="Enter Old Password:",
            bg="white",
            font=("comic", 12, "bold"),
        )
        lbl4.place(x=10, y=10)

        self.old = tk.Entry(
            self.master,
            width=25,
            fg="black",
            relief="solid",
            font=("comic", 12, "bold"),
            show="*",
        )
        self.old.place(x=205, y=10)

        lbl5 = tk.Label(
            self.master,
            text="Enter New Password:",
            bg="white",
            font=("comic", 12, "bold"),
        )
        lbl5.place(x=10, y=45)

        self.new = tk.Entry(
            self.master,
            width=25,
            fg="black",
            relief="solid",
            font=("comic", 12, "bold"),
            show="*",
        )
        self.new.place(x=205, y=45)

        lbl6 = tk.Label(
            self.master,
            text="Confirm New Password:",
            bg="white",
            font=("comic", 12, "bold"),
        )
        lbl6.place(x=10, y=80)

        self.nnew = tk.Entry(
            self.master,
            width=25,
            fg="black",
            relief="solid",
            font=("comic", 12, "bold"),
            show="*",
        )
        self.nnew.place(x=205, y=80)

        cancel = tk.Button(
            self.master,
            text="Cancel",
            command=self.master.destroy,
            fg="white",
            bg="#ff0000",
            height=1,
            width=25,
            activebackground="white",
            font=("comic", 10, "bold"),
        )
        cancel.place(x=210, y=120)

        save1 = tk.Button(
            self.master,
            text="Save",
            command=self.save_pass,
            fg="white",
            bg="#339933",
            height=1,
            width=25,
            activebackground="white",
            font=("comic", 10, "bold"),
        )
        save1.place(x=10, y=120)
        self.master.mainloop()

    def save_pass(self):
        """Save the new password"""
        key = self.check_password()
        if key is None:
            self.master.destroy()
            return

        op = self.old.get()
        newp = self.new.get()
        nnewp = self.nnew.get()

        if op == key:
            if newp == nnewp:
                with open(PASSWORD_FILE, "w") as txf:
                    txf.write(newp)
                mess._show(
                    title="Password Changed", message="Password Changed Successfully!!"
                )
                self.master.destroy()
            else:
                mess._show(title="Error", message="Confirm New Password!")
        else:
            mess._show(
                title="Wrong Password", message="Please Enter Correct Old Password!"
            )

    def take_images(self):
        """Capture images for training"""
        Id = self.txt.get()
        name = self.txt2.get()

        if not Id or not name:
            mess._show(title="Error", message="Please enter both ID and Name!")
            return

        if not (name.isalpha() or " " in name):
            self.message1.configure(text="Enter Alphabetical Name")
            return

        serial = self.get_next_serial()

        self.message1.configure(text="Initializing Camera...")

        threading.Thread(
            target=self._capture_images, args=(Id, name, serial), daemon=True
        ).start()

    def get_next_serial(self):
        """Get the next serial number for registration"""
        serial = 0
        if os.path.isfile(STUDENT_DETAILS_FILE):
            with open(STUDENT_DETAILS_FILE, "r") as csvFile1:
                reader1 = csv.reader(csvFile1)
                for l in reader1:
                    serial = serial + 1
            serial = serial // 2
        else:
            os.makedirs(os.path.dirname(STUDENT_DETAILS_FILE), exist_ok=True)
            with open(STUDENT_DETAILS_FILE, "a+") as csvFile1:
                writer = csv.writer(csvFile1)
                writer.writerow(["SERIAL NO.", "", "ID", "", "NAME"])
                serial = 1

        return serial

    def _capture_images(self, Id, name, serial):
        """Capture and save face images for training"""
        try:
            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)

            detector = self.face_cascade

            sampleNum = 0

            self.queue.put(("update_message", "Capturing Images..."))

            buffer_size = (480, 640, 3)
            img_buffer = np.zeros(buffer_size, dtype=np.uint8)

            start_time = time.time()
            frame_count = 0
            last_fps_update = start_time

            while True:
                ret, img = cam.read()
                if not ret:
                    logger.error("Failed to capture image from camera")
                    break

                img_buffer[: img.shape[0], : img.shape[1], :] = img

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(100, 100),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )

                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_update

                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_update = current_time
                    cv2.putText(
                        img,
                        f"FPS: {fps:.1f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    sampleNum += 1

                    filename = os.path.join(
                        TRAINING_IMAGE_DIR, f"{name}.{serial}.{Id}.{sampleNum}.jpg"
                    )

                    y_margin = int(h * 0.1)
                    x_margin = int(w * 0.1)
                    y1 = max(0, y - y_margin)
                    y2 = min(gray.shape[0], y + h + y_margin)
                    x1 = max(0, x - x_margin)
                    x2 = min(gray.shape[1], x + w + x_margin)

                    face_img = gray[y1:y2, x1:x2]
                    cv2.imwrite(filename, face_img)

                    cv2.putText(
                        img,
                        f"Images Captured: {sampleNum}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Taking Images", img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif sampleNum >= 100:
                    break

            cam.release()
            cv2.destroyAllWindows()

            row = [serial, "", Id, "", name]
            with open(STUDENT_DETAILS_FILE, "a+") as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)

            self.update_registration_count()

            self.queue.put(
                ("update_message", f"Images Captured Successfully for ID: {Id}")
            )
            logger.info(f"Captured {sampleNum} images for ID: {Id}, Name: {name}")

        except Exception as e:
            logger.error(f"Error capturing images: {e}")
            self.queue.put(("update_message", "Error capturing images!"))

    @lru_cache(maxsize=128)
    def _preprocess_image(self, image_path):
        """Preprocess image for training with caching for better performance"""
        pil_img = Image.open(image_path).convert("L")
        return np.array(pil_img, "uint8")

    def train_images(self):
        """Train the face recognition model with captured images"""
        try:
            self.message1.configure(text="Training Model...")

            threading.Thread(target=self._train_model, daemon=True).start()

        except Exception as e:
            logger.error(f"Error starting training: {e}")
            self.message1.configure(text="Error Training Model!")

    def _train_model(self):
        """Train the model in a background thread"""
        try:
            faces, IDs = self.get_images_and_labels(TRAINING_IMAGE_DIR)

            if len(faces) == 0:
                self.queue.put(("update_message", "No training data found!"))
                return

            self.recognizer.train(faces, np.array(IDs))

            self.recognizer.save(TRAINER_FILE)

            self.queue.put(("update_message", "Profile Saved Successfully!"))
            logger.info(f"Model trained successfully with {len(faces)} images")

        except Exception as e:
            logger.error(f"Error training model: {e}")
            self.queue.put(("update_message", "Error Training Model!"))

    def get_images_and_labels(self, path):
        """Get images and corresponding labels for training"""
        image_paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
        faces = []
        IDs = []

        total_images = len(image_paths)
        processed = 0

        batch_size = 10
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]

            for image_path in batch:
                try:
                    img_numpy = self._preprocess_image(image_path)

                    ID = int(os.path.split(image_path)[-1].split(".")[1])

                    faces.append(img_numpy)
                    IDs.append(ID)

                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    continue

                processed += 1

            progress = int((processed / total_images) * 100)
            self.queue.put(("update_message", f"Processing Images: {progress}%"))

        return faces, IDs

    def track_images(self):
        """Track and record attendance using face recognition"""
        for k in self.tv.get_children():
            self.tv.delete(k)

        if not os.path.isfile(TRAINER_FILE):
            mess._show(title="Data Missing", message="Please train the model first!")
            return

        if not os.path.isfile(STUDENT_DETAILS_FILE):
            mess._show(
                title="Details Missing",
                message="Students Details Are Missing, Please Check!",
            )
            return

        self.message1.configure(text="Initializing Camera...")

        threading.Thread(target=self._track_attendance, daemon=True).start()

    def _track_attendance(self):
        """Track attendance using face recognition (runs in separate thread)"""
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(TRAINER_FILE)

            df = pd.read_csv(STUDENT_DETAILS_FILE)

            cam = cv2.VideoCapture(0)
            cam.set(3, 640)
            cam.set(4, 480)

            font = cv2.FONT_HERSHEY_SIMPLEX

            ts = time.time()
            date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            attendance_file = os.path.join(ATTENDANCE_DIR, f"Attendance_{date}.csv")

            if not os.path.isfile(attendance_file):
                with open(attendance_file, "w") as f:
                    f.write("Id,Name,Date,Time\n")

            recorded_ids = set()

            if os.path.isfile(attendance_file):
                with open(attendance_file, "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if len(row) > 0:
                            recorded_ids.add(row[0])

            process_this_frame = True

            start_time = time.time()
            frame_count = 0
            last_fps_update = start_time

            buffer_size = (480, 640, 3)
            img_buffer = np.zeros(buffer_size, dtype=np.uint8)
            small_buffer_size = (240, 320, 3)
            small_buffer = np.zeros(small_buffer_size, dtype=np.uint8)

            self.queue.put(("update_message", "Taking Attendance..."))

            while True:
                ret, img = cam.read()
                if not ret:
                    logger.error("Failed to capture image from camera")
                    break

                img_buffer[: img.shape[0], : img.shape[1], :] = img

                frame_count += 1
                current_time = time.time()
                elapsed = current_time - last_fps_update

                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    frame_count = 0
                    last_fps_update = current_time
                    cv2.putText(
                        img,
                        f"FPS: {fps:.1f}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                if process_this_frame:
                    small_frame = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                    small_buffer[
                        : small_frame.shape[0], : small_frame.shape[1], :
                    ] = small_frame

                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

                    faces = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(50, 50),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )

                    faces = [
                        (int(x * 2), int(y * 2), int(w * 2), int(h * 2))
                        for (x, y, w, h) in faces
                    ]

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)

                        face_img = cv2.cvtColor(
                            img[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY
                        )

                        try:
                            serial, conf = recognizer.predict(face_img)

                            if conf < 70:
                                ID = str(
                                    df.loc[df["SERIAL NO."] == serial]["ID"].values[0]
                                )
                                name = str(
                                    df.loc[df["SERIAL NO."] == serial]["NAME"].values[0]
                                )

                                cv2.putText(
                                    img,
                                    name,
                                    (x, y + h + 30),
                                    font,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )

                                if ID not in recorded_ids:
                                    timeStamp = datetime.datetime.fromtimestamp(
                                        ts
                                    ).strftime("%H:%M:%S")

                                    with open(attendance_file, "a") as csvFile:
                                        writer = csv.writer(csvFile)
                                        writer.writerow([ID, name, date, timeStamp])

                                    recorded_ids.add(ID)

                                    self.queue.put(
                                        ("add_to_treeview", ID, name, date, timeStamp)
                                    )
                            else:
                                cv2.putText(
                                    img,
                                    "Unknown",
                                    (x, y + h + 30),
                                    font,
                                    1,
                                    (255, 255, 255),
                                    2,
                                )
                        except Exception as e:
                            logger.error(f"Error during face recognition: {e}")
                            continue

                process_this_frame = not process_this_frame

                cv2.imshow("Taking Attendance", img)

                if cv2.waitKey(1) == ord("q"):
                    break

            cam.release()
            cv2.destroyAllWindows()

            self.queue.put(("update_message", "Attendance Completed"))

        except Exception as e:
            logger.error(f"Error tracking attendance: {e}")
            self.queue.put(("update_message", f"Error: {str(e)}"))

    def run_multimodel(self):
        """Run the multi-model script"""
        try:
            subprocess.run(["python", "Multimodel1.py"], check=True)
        except subprocess.CalledProcessError as e:
            mess.showerror("Error", f"Failed To Run Multimodel1.py: {e}")
            logger.error(f"Failed to run Multimodel1.py: {e}")


def main():
    """Main function to start the application"""
    try:
        window = tk.Tk()

        face_system = FaceRecognitionSystem(window)

        window.mainloop()

    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        mess._show(title="Error", message=f"Application crashed: {str(e)}")


if __name__ == "__main__":
    main()
