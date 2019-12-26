from threading import Thread

import cv2


class Transform:

    def __init__(self, frame):
        self.frame = frame
        self.thread = Thread(target=self.to_blob(), args=(), daemon=True)

    def to_blob(self):
        """
        convert the frame / image to a blob
        """
        x = cv2.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        return x

    def stop(self):
        """
        Join the thread to the main thread
        """
        self.thread.join()
