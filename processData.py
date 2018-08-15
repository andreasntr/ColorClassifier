from firebase import firebase
import json
import numpy as np

data = json.load(open("data.json"))['data']
cols = []
lbls = []
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
for submission in data:
    color = []
    color.append(submission["r"] / 255)
    color.append(submission["g"] / 255)
    color.append(submission["b"] / 255)
    cols.append(color)
    lbls.append(labelsValues.index(submission["label"]))

colors = np.array(cols, dtype=np.float32)
labels = np.array(lbls, dtype=np.int8)
np.savez_compressed("processedData", colors = colors, labels = labels)
