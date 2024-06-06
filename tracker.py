import cv2
import numpy as np
from ultralytics import YOLO

import argparse


# State transition vector
#
# x = [x, y, a, h, v_x, v_y, v_a, v_h]
# x = x_ + v_x * dt
# y = y_ + v_y * dt
# a = a_ + v_a * dt
# h = h_ + v_h * dt
#
# x = F * x_ + B * u                        (1)
#   B: control matrix (there is no control in this case)
#   u: control (there is no control in this case)
#
# Then
# x = F * x_                                (2)
#
# Replace v_x * dt with d_x
# x = [x, y, a, h, d_x, d_y, d_a, d_h]
# x = x_ + d_x
# y = y_ + d_y
# a = a_ + d_a
# h = h_ + d_h
#
# State transition matrix (F):
# [1, 0, 0, 0, 1, 0, 0, 0]
# [0, 1, 0, 0, 0, 1, 0, 0]
# [0, 0, 1, 0, 0, 0, 1, 0]
# [0, 0, 0, 1, 0, 0, 0, 1]
# [0, 0, 0, 0, 1, 0, 0, 0]
# [0, 0, 0, 0, 0, 1, 0, 0]
# [0, 0, 0, 0, 0, 0, 1, 0]
# [0, 0, 0, 0, 0, 0, 0, 1]
#
# Process noise covariance matrix (Q):
#   np.eye(8) * 0.1
#
# Prediction covariance matrix
# P = F * P_ * F^T + Q                      (3)


# Measurement transition
#
# sensor reading (x)
# x = [x, y, a, h, d_x, d_y, d_a, d_h]
#
# From estimation (x) to measurement (Z)
# Z = H * x
#
# H: measurement matrix:
# [1, 0, 0, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 0]
# [0, 0, 1, 0, 0, 0, 0, 0]
# [0, 0, 0, 1, 0, 0, 0, 0]
# [0, 0, 0, 0, 1, 0, 0, 0]
# [0, 0, 0, 0, 0, 1, 0, 0]
# [0, 0, 0, 0, 0, 0, 1, 0]
# [0, 0, 0, 0, 0, 0, 0, 1]
#
# Sensor noise covariance matrix (R)
#   np.eye(8) * 0.1
#
# Measurement covariance matrix (S)
# S = H * P * H^T + R                       (4)


# Kalman process:
#
# (𝜇0,Σ0) = (H * x, H * P * H^T)            (5)
# (𝜇1,Σ1) = (Z, S)                          (6)
#
# Gaussian distribution combine (𝜇0,Σ0) and (𝜇1,Σ1):
#
# K = Σ0 * (Σ0 + Σ1)^-1                       (7)
#   K = (H * P * H^T) * (H * P * H^T + S)^-1
#
# 𝜇 = 𝜇0 + K * (𝜇1 - 𝜇0)                    (8)
#   𝜇 = H * x' = H * x + K * (Z - H * x)
#
# Σ = Σ0 - K * Σ0                           (9)
#   Σ = H * P * H^T - K * (H * P * H^T)
#
#
# For simplicity
#   Suppose: 𝜇 = H * x'
#   x' = H^-1 * 𝜇 = x + K' * (Z - H * x)    (10)
#
#   K' = H^-1 * K
#      = H^-1 * (H * P * H^T) * (H * P * H^T + S)^-1
#      = P * H^T * (H * P * H^T + S)^-1
#   K = H * K'

#   Suppose: Σ = H * P' * H^T
#   H * P' * H^T = Σ
#                = Σ0 - K * Σ0
#                = H * P * H^T - K * (H * P * H^T)
#                = H * P * H^T - H * K' * (H * P * H^T)
#                = H * (P - K' * H * P) * H^T
#   P' = P - K' * H * P                     (11)
#
#   (10) can be used in (1) for next iteration
#   (11) can be used in (3) for next iteration


class KalmanFilter(object):

    def __init__(self, x_, P_, F, Q, H, R):
        # previous state
        self.x_ = x_

        # previous covariance matrix
        self.P_ = P_

        # state transition matrix
        self.F = F

        # process covariance matrix
        self.Q = Q

        # measurement matrix
        self.H = H

        # measurement covariance matrix
        self.R = R
        self.S = None

        # kalman gain
        self.K = None

        # predict internal status
        self.x_p = None
        self.P_p = None

    def predict(self):
        """
        x = F * x_                                (2)
        P = F * P_ * F^T + Q                      (3)
        """
        # predict state
        self.x_p = np.dot(self.F, self.x_)
        self.P_p = np.dot(np.dot(self.F, self.P_), self.F.T) + self.Q

    def update(self, xywh):
        """
        xywh: new detection info

        x' = H^-1 * 𝜇 = x + K' * (Z - H * x)    (10)
        P' = P - K' * H * P                     (11)
        S = H * P * H^T + R                       (4)
        K' = P * H^T * (H * P * H^T + S)^-1
        """
        # kalman gain
        self.S = np.dot(np.dot(self.H, self.P_p), self.H.T) + self.R
        self.K = np.dot(np.dot(self.P_p, self.H.T),
                        np.linalg.inv(np.dot(np.dot(self.H,
                                                    self.P_p),
                                             self.H.T)
                                      + self.S))

        # transition x to sensor reading
        # x = [x, y, a, h, d_x, d_y, d_a, d_h]
        # Z = H * x
        deriv = np.array(xywh) - self.x_[:4]
        Z = np.concatenate((xywh, deriv))

        # optimization
        self.x_ = self.x_p + np.dot(self.K, (Z - np.dot(self.H, self.x_p)))
        self.P_ = self.P_p - np.dot(self.K, np.dot(self.H, self.P_p))

        return self.x_


def parse_args():
    parser = argparse.ArgumentParser(description='Track with KF')
    parser.add_argument('--cfg', type=str, default='yolov8n.yaml', help='config path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='model path')
    parser.add_argument('video', type=str, help='video path')
    args = parser.parse_args()
    return args


def xywh_to_ltrb(xywh):
    """
    xywh to tlbr
    """
    cx, cy, w, h = xywh
    x0 = cx - w/2
    y0 = cy - h/2
    x1 = cx + w/2
    y1 = cy + h/2
    return [x0, y0, x1, y1]


def render_box(frame, box, cls, name, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)], box_id=None):
    x0, y0, x1, y1 = box
    cv2.rectangle(frame,
                  (round(x0), round(y0)),
                  (round(x1), round(y1)),
                  colors[cls%len(colors)], 2)
    if box_id:
        cv2.putText(frame, f'{box_id}: {name}',
                    (round(x0), round(y0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, colors[cls%len(colors)], 2)

def render_detections(frame, detections, colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
    for det in detections:
        x0, y0, x1, y1 = xywh_to_ltrb(det[:4])
        cls = det[4]
        name = det[5]
        print(name, x0, y0, x1, y1)
        render_box(frame, (x0, y0, x1, y1), cls, name, colors)
    return frame


class TrackingObject(object):
    tid = 0

    TRACKING = 'TRACKING'
    LOST = 'LOST'

    def __init__(self, xywh, cls, name=None):
        self.oid = TrackingObject.next_tid()
        self.cls = cls
        self.name = name

        # previous state, init with first detection xywh, and all derivatives are 0
        self.x_ = np.concatenate((np.array(xywh), [0, 0, 0, 0]))

        # previous covariance matrix
        self.P_ = np.eye(8)

        # state transition matrix
        self.F = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                           [0, 1, 0, 0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0],
                           [0, 0, 0, 1, 0, 0, 0, 1],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # process covariance matrix
        self.Q = np.eye(8) * 0.1

        # measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1]])

        # measurement covariance matrix
        self.R = np.eye(8) * 0.1

        # init with tracking state
        self.tracking_state = self.TRACKING

        # store objects' detection track
        self.tracks = []

        self.kf = KalmanFilter(self.x_, self.P_, self.F, self.Q, self.H, self.R)

    @classmethod
    def next_tid(cls):
        cls.tid += 1
        return cls.tid

    def set_tracking_state(self):
        self.tracking_state = self.TRACKING

    def set_lost_state(self):
        self.tracking_state = self.LOST

    def is_lost(self):
        return self.tracking_state == self.LOST

    def xywh(self):
        return self.x_.tolist()[:4]

    def match_cls(self, cls):
        return self.cls == cls

    def update(self, detection):
        """
        update tracking object with new detection
        detection: xywh
        """
        self.tracks.append(detection)
        self.kf.predict()
        # update with detection
        self.x_ = self.kf.update(detection)


class Tracker(object):

    def __init__(self):
        self.objects = []

    def track(self, detections, iou_match_threshold=0.4, iou_new_threshold=0.01):
        """
        update tracking objects with new detections
        """
        if self.objects == []:
            self.objects = [TrackingObject(self.detection_xywh(det),
                                           self.detection_cls(det),
                                           self.detection_name(det))
                            for det in detections]

        # set all tracking objects to lost state before matching
        for obj in self.objects:
            obj.set_lost_state()

        # find best matches between detections and tracking objects
        detections_match_flag = [False] * len(detections)
        detections_best_iou = [0.0] * len(detections)
        for obj in self.objects:
            best_detection = None
            best_iou = 0
            for i, det in enumerate(detections):
                # skip if not same class
                if not obj.match_cls(self.detection_cls(det)):
                    continue

                # skip the matched detection
                if detections_match_flag[i]:
                    continue

                iou = self.iou_xywh(self.detection_xywh(det), obj.xywh())
                print('IOU', iou)
                if iou > best_iou and iou > iou_match_threshold:
                    best_iou = iou
                    best_detection = i

                print(detections_best_iou[i])
                if iou > detections_best_iou[i]:
                    detections_best_iou[i] = iou

            if best_detection is not None:
                detections_match_flag[best_detection] = True
                obj.set_tracking_state()
                # trigger kalman filter update
                obj.update(self.detection_xywh(detections[best_detection]))

        # add new tracking objects
        for i, det in enumerate(detections):
            if not detections_match_flag[i] and detections_best_iou[i] < iou_new_threshold:
                self.objects.append(TrackingObject(self.detection_xywh(det),
                                                   self.detection_cls(det),
                                                   self.detection_name(det)))

    @staticmethod
    def parse_results(results):
        """
        results to detections
            [cx, cy, w, h, cls, cls_name]
        """
        detections = []
        for result in results:
            for i in range(result.boxes.xywh.shape[0]):
                cx, cy, w, h = result.boxes.xywh[i].tolist()
                cls = result.boxes.cls.int().tolist()[i]
                detections.append([cx, cy, w, h, cls, result.names[cls]])
        return detections

    def detection_xywh(self, detection):
        """
        xywh in detections
        """
        return detection[:4]

    def detection_cls(self, detection):
        """
        class in detections
        """
        return detection[4]

    def detection_name(self, detection):
        """
        class name in detections
        """
        return detection[5]

    @staticmethod
    def iou_xywh(xywh1, xywh2, eps=1e-10):
        x0, y0, x1, y1 = xywh_to_ltrb(xywh1)
        x2, y2, x3, y3 = xywh_to_ltrb(xywh2)

        xmin = max(x0, x2)
        ymin = max(y0, y2)
        xmax = min(x1, x3)
        ymax = min(y1, y3)
        intersect_w = max((xmax - xmin), 0)
        intersect_h = max((ymax - ymin), 0)

        area1 = (x1 - x0) * (y1 - y0)
        area2 = (x3 - x2) * (y3 - y2)
        intersect_area = intersect_w * intersect_h

        # add eps to avoid divide by 0 issue
        return intersect_area / (area1 + area2 - intersect_area + eps)


def process(model, video):
    tracker = Tracker()

    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)

        detections = tracker.parse_results(results)
        print(detections)
        tracker.track(detections)

        for obj in tracker.objects:
            if obj.is_lost():
                continue
            render_box(frame, xywh_to_ltrb(obj.xywh()), obj.cls, obj.name, box_id=obj.oid)
        #render_detections(frame, detections)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 27:
            # ESC
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    print(f'Loading model: {args.model}')
    #model = YOLO(args.cfg).load(args.model)
    model = YOLO(args.model)

    process(model, args.video)


if __name__ == '__main__':
    main()
