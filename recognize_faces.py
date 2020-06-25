import face_recognition
import pickle
import argparse
import cv2

# region Initializing command argument parser
argsParser = argparse.ArgumentParser()
argsParser.add_argument("-e", "--encodings", required=True,
                        help="path to input directory of faces + images")
argsParser.add_argument("-i", "--image", required=True,
                        help="path to serialized db of facial encodings")
args = vars(argsParser.parse_args())
#endregion

# region Prepare image and data
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

image = cv2.imread(args["image"])
rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgbImage)
encodings = face_recognition.face_encodings(rgbImage, boxes)

names = []
# endregion

# region Finding best fit
# Loop over the encodings to find the best fit to a given face
for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)

    # As at the beginning we don't know whose face it is we set it to "Unknown"
    name = "Unknown"

    if True in matches:
        matchedIndexes = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matchedIndexes:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

            # Getting the name with the highest number of "votes"
            name = max(counts, key=counts.get)

        names.append(name)
# endregion

# region Displaying an image
# Showing an image with a rectangle around the face with a given written above
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    name = name.replace('_', ' ')
    name = name.upper()
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
# endregion
