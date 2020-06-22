import pickle
import os
import argparse
import paths
import cv2
import face_recognition

# region Initializing command argument parser
argsParser = argparse.ArgumentParser()
argsParser.add_argument("-i", "--dataset", required=True,
                        help="path to input directory of faces + images")
argsParser.add_argument("-e", "--encodings", required=True,
                        help="path to serialized db of facial encodings")
args = vars(argsParser.parse_args())
# endregion

# region Prepare images and data
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []
# endregion

# region Encoding
for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgbImage)
    encodings = face_recognition.face_encodings(rgbImage, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
# endregion

# region Saving encodings to disk

# It's better to save encodings as it takes some time to
# obtain them and we don't want to wait every time when
# there is a need to recognize a face

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
# endregion
