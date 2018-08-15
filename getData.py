from firebase import firebase
import json
db = firebase.FirebaseApplication("https://color-classification.firebaseio.com", None)
jsonData = db.get("/colors", None)
data = []

for user in jsonData.keys():
    entry = {}
    entry['r'] = jsonData[user]['r']
    entry['g'] = jsonData[user]['g']
    entry['b'] = jsonData[user]['b']
    entry['label'] = jsonData[user]['label']
    data.append(entry)
json.dump({"data":data}, open("data.json", "w"))
