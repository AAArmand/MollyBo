import dlib


class LandmarkDetector:
    _PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    _PREDICTOR = dlib.shape_predictor(_PREDICTOR_PATH)

    faces = []

    def __init__(self, img_face):
        self.img_face = img_face
        self.find_faces()

    #  Get the landmarks/parts for the face in box d.
    def _get_face_points_list(self, face_box):
        return list(map(lambda p: (p.x, p.y), self._PREDICTOR(self.img_face, face_box).parts()))

    def find_faces(self):
        detector = dlib.get_frontal_face_detector()
        dets = detector(self.img_face, 1)

        for k, d in enumerate(dets):
            points = self._get_face_points_list(d)[:27]

            lane1 = points[17:20]
            lane1.reverse()
            lane2 = points[24:27]
            lane2.reverse()
            face = points[:17]
            face.extend(lane2)
            face.extend(lane1)

            x = [point[0] for point in points]
            y = [point[1] for point in points]

            self.faces.append({
                'points_list': face,
                'left':  min(x),
                'right': max(x),
                'top':  min(y),
                'bottom': max(y),

            })
