import pickle
import os
import argparse
import paths
import cv2
import face_recognition


argsParser = argparse.ArgumentParser()
argsParser.add_argument("-i", "--dataset", required=True,
                        help="path to input directory of faces + images")
argsParser.add_argument("-e", "--encodings", required=True,
                        help="path to serialized db of facial encodings")
args = vars(argsParser.parse_args())

detectionMethod = "cnn"

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

# loop over the images
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgbImage, model = detectionMethod)
    encodings = face_recognition.face_encodings(rgbImage, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
