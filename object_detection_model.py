import tkinter as Tk
from tkinter import filedialog
from tkinter import *
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt

CATEGORIES = ["folder1"]

global arr
arr = []


def directory():
    global img_dir
    global arr
    img_dir = filedialog.askdirectory()
    print(img_dir)
    for category in CATEGORIES:
        path = os.path.join(img_dir, category)
        for img in os.listdir(path):
            arr.append(os.path.join(path, img))
    print(arr[0])


# Declaration of the window with size and title
root = Tk()
root.geometry("1200x600")
root.title("Intern Work - Ankit Saini")
root.configure(background='white')
# Opening folder path


i = 0
flag = 1
path = ''

def image_next():
    while 1:
        global flag
        if flag == 1:
            global i
            i = i+1
            load = Image.open(arr[i])
            global path
            path = arr[i]
            load = load.resize((600, 600), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = Label(root, image=render)
            img.image = render
            img.place(x=300, y=0)
            flag = 0
        else:
            flag = 1
            break


def image_prev():
    while 1:
        global flag
        if flag == 1:
            global i
            i = i-1
            load = Image.open(arr[i])
            global path
            path = arr[i]
            load = load.resize((600, 600), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = Label(root, image=render)
            img.image = render
            img.place(x=300, y=0)
            flag = 0
        else:
            flag = 1
            break


P = IntVar(); C = IntVar(); D = IntVar(); V = IntVar(); Ch = IntVar(); E_G = IntVar(); var=IntVar(); Anno = IntVar()
def ML():
    threshold = float(entry1.get())
    arr1_label = []
    if(P.get()):
        arr1_label.append(int(1))
    if(C.get()):
        arr1_label.append((int(17)))
    if (D.get()):
        arr1_label.append(int(18))
    if (V.get()):
        for i in range(3, 7):
            arr1_label.append(int(i))
    if (C.get()):
        arr1_label.append(int(62))
    if (E_G.get()):
        for i in range(72, 83):
            arr1_label.append((int(i)))


    if var.get() == 1:
        graph_path = '/root/PycharmProjects/Intern Project/faster_rcnn.pb'
    if var.get() == 2:
        graph_path = '/root/PycharmProjects/Intern Project/ssd_mobilenet.pb'
    if var.get() == 3:
        graph_path = '/root/PycharmProjects/Intern Project/ssd.pb'
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        global path
        img = plt.imread(path)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            if classId in arr1_label:
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > threshold:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    x1 = []; y1 = []; x2 = []; y2 = []
                    x1.append(x)
                    y1.append(y)
                    x2.append(right)
                    y2.append(bottom)
                    if Anno.get():
                        file1 = open(path + ".txt", "a")
                        file1.write(repr(x1) + repr(y1) + repr(x2) + repr(y2))
                        file1.writelines("\n")
                        file1.close()
                        del x1
                        del y1
                        del x2
                        del y2
    C_det = Canvas(root, width=600, height=600, bg='white')
    C_det.place(x=300, y=0)
    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    img = Label(C_det, image=photo)
    img.image = photo
    img.place(x=0, y=0)


# Making layout of the window


# Left Side of the window
frame_left = Frame(root, height=600, width=300, background='white')
frame_left.place(x=0, y=0)

button1 = Button(frame_left, text='Open folder', bd='1', height='2', width='14', command=directory)
button1.place(x=100, y=100)
button2 = Button(frame_left, text='Next Image', bd='1', height='2', width='14', command=image_next)
button2.place(x=100, y=220)
button3 = Button(frame_left, text='Previous Image', bd='1', height='2', width='14', command=image_prev)
button3.place(x=100, y=340)
che = Checkbutton(frame_left, text='Save Annotations\n(.txt) file', bd='1', variable=Anno)
che.place(x=100, y=460)


# Right Side of the window
frame_right = Frame(root, height=300, width=300, background='white')
frame_right.place(x=900, y=0)
frame_right_down = Frame(root, height=300, width=300, background='white')
frame_right_down.place(x=900, y=300)
label1 = Label(frame_right, text='Select Model', background='white')
label1.place(x=100, y=100)
radio1 = Radiobutton(frame_right, text='Faster R-CNN', variable=var, value=1, background='white')
radio1.place(x=100, y=130)
radio2 = Radiobutton(frame_right, text='Mobilenet', variable=var, value=2, background='white')
radio2.place(x=100, y=160)
radio3 = Radiobutton(frame_right, text='SSD', variable=var, value=3, background='white')
radio3.place(x=100, y=190)

label2 = Label(frame_right, text='Detection \nThreshold', background='white')
label2.place(x=100, y=250)
entry1 = Entry(frame_right, bd=1, highlightcolor='blue', width=7, background='white')
entry1.place(x=180, y=255)

label3 = Label(frame_right_down, text='Label Filters', background='white')
label3.place(x=100, y=30)
check = Checkbutton(frame_right_down, text='Person', variable=P, background='white')
check.place(x=100, y=60)
check = Checkbutton(frame_right_down, text='Cat', variable=C, background='white')
check.place(x=100, y=90)
check = Checkbutton(frame_right_down, text='Dog', variable=D, background='white')
check.place(x=100, y=120)
check = Checkbutton(frame_right_down, text='Vehicles', variable=V, background='white')
check.place(x=100, y=150)
check = Checkbutton(frame_right_down, text='Chair', variable=Ch, background='white')
check.place(x=100, y=180)
check = Checkbutton(frame_right_down, text='Electronic Goods', variable=E_G, background='white')
check.place(x=100, y=215)

detect_b = Button(frame_right_down, text='Detect', width='20', command=ML)
detect_b.place(x=50, y=250)

# Center Image

C1 = Canvas(root, height=600, width=600, bg='white')
C1.place(x=300, y=0)

root.mainloop()
