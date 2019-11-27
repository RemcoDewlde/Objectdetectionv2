import cv2
from threading import Thread


class Transform:
    def __init__(self, frame):
        self.frame = frame
        self.thread = Thread(target=self.to_blob(), args=(), daemon=True)

    def to_blob(self):
        # convert the frame / image to a blob
        x = cv2.dnn.blobFromImage(self.frame, 0.00033, (416, 416), (0, 0, 0), True, crop=False)
        return x

    def stop(self):
        self.thread.join()


