import dlib
import numpy as np
import face_recognition_models

# region Variables

face_detector = dlib.get_frontal_face_detector()

predictor_model = face_recognition_models.pose_predictor_model_location()
pose_predictor = dlib.shape_predictor(predictor_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)

# endregion

dlib.DLIB_USE_CUDA = True


# region Functions used to make sure that face rectangle is within the bound of the image and convert things

# Converts a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
def _rect_to_css(rect):
    return rect.top(), rect.right(), rect.bottom(), rect.left()


# Converts a tuple in (top, right, bottom, left) order to a dlib `rect` object
def _css_to_rect(css):
    return dlib.rectangle(css[3], css[0], css[1], css[2])


# Makes sure a tuple in (top, right, bottom, left) order is within the bounds of the image
# and returns a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
def _trim_css_to_bounds(css, image_shape):
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


# endregion

# region Functions used to encode faces

# Returns a list of dlib 'rect' objects of found face locations
def raw_face_locations(img, number_of_times_to_up_sample=1):
    return cnn_face_detector(img, number_of_times_to_up_sample)


# Returns
def raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


# Returns list of tuples of found face locations in css (top, right, bottom, left) order
def face_locations(img, number_of_times_to_up_sample=1):
    return [_trim_css_to_bounds(_rect_to_css(face.rect), img.shape) for face in
            raw_face_locations(img, number_of_times_to_up_sample)]


# Returns a list of 128-dimensional face encodings (one for each face in the image)
def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    raw_landmarks = raw_face_landmarks(face_image, known_face_locations)
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# endregion

# region Functions used to recognize a given face

# Returns a numpy ndarray with the distance for each face in the same order as the 'faces' array
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


# Returns a list of True/False values indicating which known_face_encodings match the face encoding to check
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

# endregion
