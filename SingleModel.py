import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2, os
import csv
import subprocess
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def tick():
    time_string = time.strftime("%H:%M:%S")
    clock.config(text=time_string)
    clock.after(200, tick)


def contact():
    mess._show(
        title="Contact Us",
        message="Please Contact Us At :- 'pandeyadarsh3115@gmail.com'",
    )


def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(
            title="Some Files Are Missing", message="Please Contact Us For Help!!!"
        )
        window.destroy()


def save_pass():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring(
            "Old Password Not Found!", "Please Enter New Password Below!", show="*"
        )
        if new_pas is None:
            mess._show(
                title="No Password Was Entered",
                message="Password Not Set! Please Try Again!",
            )
        else:
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(
                title="Password Registered",
                message="New Password Registered Successfully!!",
            )
            return
    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    if op == key:
        if newp == nnewp:
            txf = open("TrainingImageLabel/psd.txt", "w")
            txf.write(newp)
        else:
            mess._show(title="Error", message="Confirm New Password!")
            return
    else:
        mess._show(title="Wrong Password", message="Please Enter Correct Old Password!")
        return
    mess._show(title="Password Changed", message="Password Changed Successfully!!")
    master.destroy()


def change_pass():
    global master
    master = tk.Tk()
    master.geometry("450x165")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="white")
    lbl4 = tk.Label(
        master, text="Enter Old Password:", bg="white", font=("comic", 12, " bold ")
    )
    lbl4.place(x=10, y=10)
    global old
    old = tk.Entry(
        master,
        width=25,
        fg="black",
        relief="solid",
        font=("comic", 12, " bold "),
        show="*",
    )
    old.place(x=205, y=10)
    lbl5 = tk.Label(
        master, text="Enter New Password:", bg="white", font=("comic", 12, " bold ")
    )
    lbl5.place(x=10, y=45)
    global new
    new = tk.Entry(
        master,
        width=25,
        fg="black",
        relief="solid",
        font=("comic", 12, " bold "),
        show="*",
    )
    new.place(x=205, y=45)
    lbl6 = tk.Label(
        master, text="Confirm New Password:", bg="white", font=("comic", 12, " bold ")
    )
    lbl6.place(x=10, y=80)
    global nnew
    nnew = tk.Entry(
        master,
        width=25,
        fg="black",
        relief="solid",
        font=("comic", 12, " bold "),
        show="*",
    )
    nnew.place(x=205, y=80)
    cancel = tk.Button(
        master,
        text="Cancel",
        command=master.destroy,
        fg="white",
        bg="#ff0000",
        height=1,
        width=25,
        activebackground="white",
        font=("comic", 10, " bold "),
    )
    cancel.place(x=210, y=120)
    save1 = tk.Button(
        master,
        text="Save",
        command=save_pass,
        fg="white",
        bg="#339933",
        height=1,
        width=25,
        activebackground="white",
        font=("comic", 10, " bold "),
    )
    save1.place(x=10, y=120)
    master.mainloop()


def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel/psd.txt")
    if exists1:
        tf = open("TrainingImageLabel/psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring(
            "Old Password Not Found!", "Please Enter New Password Below!", show="*"
        )
        if new_pas is None:
            mess._show(
                title="No Password Was Entered",
                message="Password Not Set! Please Try Again!",
            )
        else:
            tf = open("TrainingImageLabel/psd.txt", "w")
            tf.write(new_pas)
            mess._show(
                title="Password Registered",
                message="New Password Registered Successfully!!",
            )
            return
    password = tsd.askstring("Password", "Enter Password:", show="*")
    if password == key:
        TrainImages()
    elif password is None:
        pass
    else:
        mess._show(title="Wrong Password", message="You Entered Wrong Password!")


def clear():
    txt.delete(0, "end")
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


def clear2():
    txt2.delete(0, "end")
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)


def TakeImages():
    check_haarcascadefile()
    columns = ["SERIAL NO.", "", "ID", "", "NAME"]
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")

    if exists:
        with open("StudentDetails/StudentDetails.csv", "r") as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = serial // 2
        csvFile1.close()
    else:

        with open("StudentDetails/StudentDetails.csv", "a+") as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()

    Id = txt.get()
    name = txt2.get()

    if (name.isalpha()) or (" " in name):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite(
                    "TrainingImage/ "
                    + name
                    + "."
                    + str(serial)
                    + "."
                    + Id
                    + "."
                    + str(sampleNum)
                    + ".jpg",
                    gray[y : y + h, x : x + w],
                )
                cv2.imshow("Taking Images", img)
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()

        row = [serial, "", Id, "", name]
        with open("StudentDetails/StudentDetails.csv", "a+") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()

        update_registration_count()

        res = "Images Taken For ID : " + Id
        message1.configure(text=res)
    else:
        message.configure(text="Enter Correct Name!")


def update_registration_count():

    res = 0
    exists = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists:
        with open("StudentDetails/StudentDetails.csv", "r") as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                res = res + 1
        res = (res // 2) - 1
        csvFile1.close()
    else:
        res = 0
    message.configure(text="Total Registrations Till Now  : " + str(res))


def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title="No Registrations", message="Please Register Someone First!")
        return
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Profile Saved Successfully!!"
    message1.configure(text=res)
    message.configure(text="Total Registrations Till Now  : " + str(ID[0]))


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert("L")
        imageNp = np.array(pilImage, "uint8")
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids


def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")

    for k in tv.get_children():
        tv.delete(k)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("TrainingImageLabel/Trainner.yml")

    if exists3:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    else:
        mess._show(
            title="Data Missing", message="Please Click On Save Profile To Reset Data!"
        )
        return

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ["Id", "", "Name", "", "Date", "", "Time"]

    exists1 = os.path.isfile("StudentDetails/StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
    else:
        mess._show(
            title="Details Missing",
            message="Students Details Are Missing, Please Check!",
        )
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()

    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    attendance_file = f"Attendance/Attendance_{date}.csv"
    recorded_ids = set()

    if os.path.isfile(attendance_file):
        with open(attendance_file, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) > 0:
                    recorded_ids.add(row[0])

    while True:
        ret, im = cam.read()
        if not ret:
            print("Failed To Capture Image!")
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y : y + h, x : x + w])

            if conf < 50:
                ID = str(df.loc[df["SERIAL NO."] == serial]["ID"].values[0])
                name = str(df.loc[df["SERIAL NO."] == serial]["NAME"].values[0])

                if ID not in recorded_ids:
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    attendance = [ID, "", name, "", date, "", timeStamp]

                    with open(attendance_file, "a+") as csvFile:
                        writer = csv.writer(csvFile)
                        writer.writerow(attendance)
                    recorded_ids.add(ID)

                    tv.insert(
                        "",
                        0,
                        text=ID,
                        values=(name, date, timeStamp),
                    )

                cv2.putText(im, name, (x, y + h + 30), font, 1, (255, 255, 255), 2)
            else:
                cv2.putText(im, "Unknown", (x, y + h + 30), font, 1, (255, 255, 255), 2)

        cv2.imshow("Taking Attendance", im)
        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()


global key
key = ""

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
day, month, year = date.split("-")

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

window = tk.Tk()
window.geometry("1920x1080")
window.resizable(True, False)
window.title("Face Recognition Based Attendance System Developed By A&J")
window.configure(background="#ffff00")

frame1 = tk.Frame(window, bg="#a05fbf")
frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.70)

frame2 = tk.Frame(window, bg="#a05fbf")
frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.70)

message3 = tk.Label(
    window,
    text="Face Recognition Based Attendance System",
    fg="white",
    bg="#ff0000",
    width=60,
    height=1,
    font=("comic", 29, " bold "),
)
message3.place(x=40, y=10)

frame3 = tk.Frame(window, bg="#c4c6ce")
frame3.place(relx=0.52, rely=0.09, relwidth=0.09, relheight=0.07)

frame4 = tk.Frame(window, bg="#c4c6ce")
frame4.place(relx=0.36, rely=0.09, relwidth=0.16, relheight=0.07)

datef = tk.Label(
    frame4,
    text=day + "-" + mont[month] + "-" + year + "   &",
    fg="#0000ff",
    bg="#ffff00",
    width=55,
    height=1,
    font=("comic", 22, " bold "),
)
datef.pack(fill="both", expand=1)

clock = tk.Label(
    frame3, fg="#0000ff", bg="#ffff00", width=55, height=1, font=("comic", 22, " bold ")
)
clock.pack(fill="both", expand=1)
tick()

head2 = tk.Label(
    frame2,
    text="                       For New Registrations                       ",
    fg="#ffffff",
    bg="#ff1493",
    font=("comic", 17, " bold "),
)
head2.grid(row=0, column=0)

head1 = tk.Label(
    frame1,
    text="                       For Already Registered                       ",
    fg="#ffffff",
    bg="#ff1493",
    font=("comic", 17, " bold "),
)
head1.place(x=0, y=0)

lbl = tk.Label(
    frame2,
    text="Enter ID",
    width=20,
    height=1,
    fg="black",
    bg="#a05fbf",
    font=("comic", 17, " bold "),
)
lbl.place(x=80, y=55)

txt = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, " bold "))
txt.place(x=30, y=88)

lbl2 = tk.Label(
    frame2,
    text="Enter Name",
    width=20,
    fg="black",
    bg="#a05fbf",
    font=("comic", 17, " bold "),
)
lbl2.place(x=80, y=140)

txt2 = tk.Entry(frame2, width=32, fg="black", font=("comic", 15, " bold "))
txt2.place(x=30, y=173)

message1 = tk.Label(
    frame2,
    text="1)Take Images  >>>  2)Save Profile",
    bg="#a05fbf",
    fg="black",
    width=39,
    height=1,
    activebackground="#3ffc00",
    font=("comic", 15, " bold "),
)
message1.place(x=7, y=230)

message = tk.Label(
    frame2,
    text="",
    bg="#a05fbf",
    fg="black",
    width=39,
    height=1,
    activebackground="#3ffc00",
    font=("comic", 16, " bold "),
)
message.place(x=7, y=450)

lbl3 = tk.Label(
    frame1,
    text="Attendance",
    width=20,
    fg="black",
    bg="#a05fbf",
    height=1,
    font=("comic", 17, " bold "),
)
lbl3.place(x=100, y=115)

res = 0
exists = os.path.isfile("StudentDetails/StudentDetails.csv")
if exists:
    with open("StudentDetails/StudentDetails.csv", "r") as csvFile1:
        reader1 = csv.reader(csvFile1)
        for l in reader1:
            res = res + 1
    res = (res // 2) - 1
    csvFile1.close()
else:
    res = 0
message.configure(text="Total Registrations Till Now  : " + str(res))

menubar = tk.Menu(
    window,
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


def run_Multimodel1():
    try:
        subprocess.run(["python", "Multimodel1.py"], check=True)
    except subprocess.CalledProcessError as e:
        mess.showerror("Error", f"Failed To Run Multimodel1.py: {e}")


filemenu.add_command(label="Other Models", command=run_Multimodel1)
filemenu.add_command(label="Change Password", command=change_pass)
filemenu.add_command(label="Contact Us", command=contact)
filemenu.add_command(label="Exit", command=window.destroy)

menubar.add_cascade(label="Help", font=("Helvetica", 12, "bold"), menu=filemenu)

tv = ttk.Treeview(frame1, height=13, columns=("name", "date", "time"))
tv.column("#0", width=82)
tv.column("name", width=130)
tv.column("date", width=133)
tv.column("time", width=133)
tv.grid(row=2, column=0, padx=(0, 0), pady=(150, 0), columnspan=4)
tv.heading("#0", text="ID")
tv.heading("name", text="NAME")
tv.heading("date", text="DATE")
tv.heading("time", text="TIME")

scroll = ttk.Scrollbar(frame1, orient="vertical", command=tv.yview)
scroll.grid(row=2, column=4, padx=(0, 100), pady=(150, 0), sticky="ns")
tv.configure(yscrollcommand=scroll.set)

clearButton = tk.Button(
    frame2,
    text="Clear",
    command=clear,
    fg="black",
    bg="#ff4500",
    width=11,
    activebackground="skyblue",
    font=("comic", 11, " bold "),
)
clearButton.place(x=335, y=86)
clearButton2 = tk.Button(
    frame2,
    text="Clear",
    command=clear2,
    fg="black",
    bg="#ff4500",
    width=11,
    activebackground="skyblue",
    font=("comic", 11, " bold "),
)
clearButton2.place(x=335, y=172)
takeImg = tk.Button(
    frame2,
    text="Take Images",
    command=TakeImages,
    fg="white",
    bg="#6d00fc",
    width=34,
    height=1,
    activebackground="white",
    font=("comic", 15, " bold "),
)
takeImg.place(x=30, y=300)
trainImg = tk.Button(
    frame2,
    text="Save Profile",
    command=psw,
    fg="white",
    bg="#339933",
    width=34,
    height=1,
    activebackground="white",
    font=("comic", 15, " bold "),
)
trainImg.place(x=30, y=380)
trackImg = tk.Button(
    frame1,
    text="Take Attendance",
    command=TrackImages,
    fg="black",
    bg="#3ffc00",
    width=35,
    height=1,
    activebackground="white",
    font=("comic", 15, " bold "),
)
trackImg.place(x=30, y=50)
quitWindow = tk.Button(
    frame1,
    text="Quit",
    command=window.destroy,
    fg="black",
    bg="#ff0000",
    width=35,
    height=1,
    activebackground="white",
    font=("comic", 15, " bold "),
)
quitWindow.place(x=30, y=450)

copyWrite = tk.Text(
    window,
    background=window.cget("background"),
    borderwidth=0,
    font=("comic", 20, "bold"),
)
copyWrite.tag_configure("superscript", offset=5)
copyWrite.insert("insert", "Developed By A&J", "", "", "superscript")
copyWrite.configure(state="disabled", fg="blue")
copyWrite.pack(side="left")
copyWrite.place(x=630, y=730)

window.configure(menu=menubar)
window.mainloop()
