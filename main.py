from collections import Counter

import cv2
import imutils
import numpy as np
import pafy as pafy

from classes.color import Color
from classes.fps import FPS
from classes.transform import Transform
from classes.videoStream import VideoStream
from classes.yolov import Net

url = 'https://youtu.be/25EgbhdVESE'
vlink = pafy.new(url)
play = vlink.getbest()

net = Net('weights/test.weights', 'cfg/test.cfg', 'cfg/coco.names')
vs = VideoStream(play.url).start()
fps = FPS().start()
colors = Color(net.classes, True)

while vs.more():
    frame = vs.read()
    frame = imutils.resize(frame, width=500)
    height, width, channels = frame.shape

    # Detecting objects
    blob = Transform(frame).to_blob()
    outs = net.get_outs(blob)

    class_ids = []
    confidences = []
    boxes = []
    labels = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.02:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w // 2)
                y = int(center_y - h // 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    labels = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(net.classes[class_ids[i]])
            labels.append(label)
            confidence = confidences[i]
            color = colors.color_array[class_ids[i]]

            # draw circle to check where the box starts
            # cv2.circle(frame, (x, y), 12,(255, 0, 0), 2 )

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, color,
                        2)

    # show queue size
    counter = dict(Counter(labels))
    cv2.putText(frame, str(counter), (0, 0 + 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
    # text = Text().draw(counter, frame, (0, 145, 255))

    # Show frame in window named Frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
