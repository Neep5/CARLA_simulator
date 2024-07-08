import rclpy  # Импорт основной библиотеки для работы с ROS2 на Python
from rclpy.node import Node  # Импорт базового класса для создания нод
from sensor_msgs.msg import Image  # Импорт сообщения ROS2 для работы с изображениями
from cv_bridge import CvBridge  # Импорт библиотеки для преобразования между ROS и OpenCV изображениями
from ultralytics import YOLO  # Импорт модели YOLO для обнаружения и отслеживания объектов
import cv2  # Импорт библиотеки OpenCV для обработки изображений
import numpy as np  # Импорт библиотеки NumPy для работы с массивами
from filterpy.kalman import KalmanFilter  # Импорт фильтра Калмана из библиотеки filterpy
from scipy.optimize import linear_sum_assignment  # Импорт функции для решения задачи линейного назначения
from numba import jit  # Импорт библиотеки Numba для ускорения вычислений

try:
    import lap
    lap_available = True
except ImportError:
    lap_available = False

@jit
def linear_assignment(cost_matrix):
    if lap_available:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
    else:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
              (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0]])
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                trk.update(dets[d, :][0])
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = iou(det, trk)
    matched_indices = linear_assignment(-iou_matrix)
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'processed_video_stream', 10)
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.tracker = Sort()
        self.previous_positions = {}
        self.frame_count = 0

    def listener_callback(self, msg):
        current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(current_frame)
        if isinstance(results, list):
            results = results[0]
        bboxes = results.boxes.xyxy.cpu().numpy()

        if bboxes.size == 0:
            return

        dets = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            conf = bbox[4] if len(bbox) > 4 else 1.0
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)
        tracks = self.tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if track_id in self.previous_positions:
                prev_center_x, prev_center_y, prev_frame = self.previous_positions[track_id]
                distance = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                speed_m_s = distance / (self.frame_count - prev_frame)
                speed_km_h = speed_m_s * 3.6
            else:
                speed_km_h = 0.0

            self.previous_positions[track_id] = (center_x, center_y, self.frame_count)

            cv2.rectangle(current_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(current_frame, f'ID: {int(track_id)}', (int(center_x), int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(current_frame, f'Speed: {speed_km_h:.2f} km/h', (int(center_x), int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print(f'ID: {int(track_id)}, Speed: {speed_km_h:.2f} km/h, Coordinates: ({x1}, {y1}, {x2}, {y2})')

        cv2.imshow('Video', current_frame)
        cv2.waitKey(1)

        processed_frame = self.bridge.cv2_to_imgmsg(current_frame, encoding='bgr8')
        self.publisher_.publish(processed_frame)
        self.frame_count += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

