import cv2
from threading import Thread
from queue import Queue


class Transform:
    def __init__(self, frame):
        self.frame = frame
        self.thread = Thread(target=self.to_blob(), args=(), daemon=True)

    def to_blob(self):
        # convert the frame / image to a blob
        bframe = cv2.dnn.blobFromImage(self.frame, 0.00055, (416, 416), (0, 0, 0), True, crop=False)
        return bframe

    def stop(self):
        self.thread.join()


