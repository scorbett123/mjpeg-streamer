import os
import sys
import time
import threading
import numpy as np
import cv2 as cv
import mjpeg_streamer

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()

        self.cond = threading.Condition()

        self.running = False

        self.frame = None

        self.latestnum = 0

        self.callback = None
        
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            # block for fresh frame
            (rv, img) = self.capture.read()
            assert rv
            counter += 1

            # publish the frame
            with self.cond: # lock the condition for this operation
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()

            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)

            return (self.latestnum, self.frame)

def main():
    server = mjpeg_streamer.MjpegServer()
    stream = mjpeg_streamer.Stream("demo", (640, 480))
    server.add_stream(stream)

    server.start()

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    # wrap it
    fresh = FreshestFrame(cap)

    while True:
        _, frame = fresh.read()
        cv.imshow("frame", frame)
        stream.set_frame(frame)
        cv.waitKey(0)

    print("done!")

    fresh.release()

    cv.destroyWindow("frame")


if __name__ == '__main__':
    main()