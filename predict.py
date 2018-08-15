import tensorflow as tf
import numpy as np
import json
from tkinter import *
import tkinter.messagebox as messagebox
from subprocess import run
from sys import exit

labelsValues = [
    "red-ish",
    "green-ish",
    "blue-ish",
    "orange-ish",
    "yellow-ish",
    "pink-ish",
    "purple-ish",
    "brown-ish",
    "grey-ish"
]

def updateModel():
    ans = messagebox.askquestion("Update model","The window will be closed and restarted when the process completes.\
    \nThe process will take a while...\nDo you wish to continue?")
    if ans == "yes":
        window.destroy()
        run("python getData.py")
        run("python processData.py")
        run("python train.py")
        run("python predict.py")
        exit(0)
    else:
        if(noModel):
            exit(0)

def predict():
    r = R.get()/255
    g = G.get()/255
    b = B.get()/255
    prediction_lbl.configure(text=labelsValues[np.argmax(model.predict(
        tf.constant([[r,g,b]], dtype=tf.float32)))])

def update(event):
    preview.configure(bg='#{:2x}{:2x}{:2x}'.format(R.get(),G.get(),B.get()).replace(" ", "0"))

tf.enable_eager_execution()
noModel = False

#LAYOUT
window = Tk()
windowWidth = window.winfo_reqwidth()
windowHeight = window.winfo_reqheight()
positionRight = int(window.winfo_screenwidth()/2 - windowWidth/2)
positionDown = int(window.winfo_screenheight()/2 - windowHeight/2)
window.geometry("+{}+{}".format(positionRight, positionDown))
window.geometry('200x170')
window.title("Color Classifier")

menu = Menu(window)
window.config(menu=menu)
fileMenu = Menu(menu)
menu.add_cascade(label = "File", menu=fileMenu)
fileMenu.add_command(label = "Update model", command=updateModel)
preview = Label(window, width=10, height=5)
R_lbl = Label(window, text="R", fg='red')
G_lbl = Label(window, text="G", fg='green')
B_lbl = Label(window, text="B", fg='blue')
R = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='red', command=update)
G = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='green', command=update)
B = Scale(window, from_=0, to=255, orient=HORIZONTAL, fg='blue', command=update)
predictButton = Button(window,text="Predict", command = predict)
prediction_lbl = Label(window, text = "")
R_lbl.grid(row= 0, sticky=S)
G_lbl.grid(row= 1, sticky=S)
B_lbl.grid(row= 2, sticky=S)
R.grid(row= 0,column =1)
G.grid(row= 1,column =1)
B.grid(row= 2,column =1)
preview.configure(bg='#{:2x}{:2x}{:2x}'.format(R.get(),G.get(),B.get()).replace(" ", "0"))
predictButton.grid(columnspan=3)
prediction_lbl.grid(row = 0, column = 2, columnspan = 3)
preview.grid(row = 1, column = 2, sticky = N, rowspan = 3)
window.columnconfigure(0, weight=1)
window.columnconfigure(1, weight=1)
window.columnconfigure(2, weight=1)

try:
    with open("model.json"):
        pass
except FileNotFoundError:
    noModel = True
    ans = messagebox.askquestion("No model found","Do you wish to start the creation of a new model?")
    if ans == "yes":
        updateModel()
    else:
        exit(0)

#LOADING MODEL
model = tf.keras.models.model_from_json(json.load(open("model.json"))["model"], custom_objects={})
model.load_weights("model_weights.h5")


window.mainloop()
