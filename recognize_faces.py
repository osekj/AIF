import face_recognition
import pickle
import argparse
import cv2

argsParser = argparse.ArgumentParser()
argsParser.add_argument("-e", "--encodings", required=True,
                        help="path to input directory of faces + images")
argsParser.add_argument("-i", "--image", required=True,
                        help="path to serialized db of facial encodings")
args = vars(argsParser.parse_args())

detectionMethod = "cnn"

print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgbImage, model=detectionMethod)
encodings = face_recognition.face_encodings(rgbImage, boxes)

names = []

for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unkown"

    if True in matches:
        matchedIndexes = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIndexes:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

            name = max(counts, key=counts.get)

        names.append(name)


# Showing a rectangle with a given name around the detected face
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
