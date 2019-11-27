import pafy as pafy
import cv2
import imutils
from classes.videoStream import VideoStream
from classes.fps import FPS
from classes.transform import Transform
from classes.yolov import Net

url = 'https://youtu.be/9b2dxc1QalM'
vlink = pafy.new(url)
play = vlink.getbest()

net = Net('weights/yolov3.weights', 'cfg/yolov3.cfg', 'cfg/coco.names')
vs = VideoStream('vid.mp4').start()
fps = FPS().start()

while vs.more():
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # Detecting objects
    blob = Transform(frame).to_blob()
    outs = net.get_outs(blob)
    print(outs)

    cv2.putText(frame, str(vs.Q.qsize()), (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_4)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    fps.update()

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
