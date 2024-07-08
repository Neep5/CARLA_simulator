# треккер sort.py
import numpy as np
from numba import jit
from filterpy.kalman import KalmanFilter

@jit
def iou(bb_test, bb_gt):
    """
    Вычисляет Intersection over Union (IoU) между двумя ограничивающими прямоугольниками.
    Параметры:
    bb_test (numpy array): Ограничивающий прямоугольник в формате [x1, y1, x2, y2]
    bb_gt (numpy array): Ограничивающий прямоугольник в формате [x1, y1, x2, y2]
    Возвращает:
    float: Значение IoU
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
            + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return o

@jit
def convert_bbox_to_z(bbox):
    """
    Преобразует ограничивающий прямоугольник в формат, необходимый для фильтра Калмана.
    Параметры:
    bbox (numpy array): Ограничивающий прямоугольник в формате [x1, y1, x2, y2]
    Возвращает:
    numpy array: Ограничивающий прямоугольник в формате [x, y, s, r], где x, y - центр, s - площадь, r - соотношение сторон
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

@jit
def convert_x_to_bbox(x, score=None):
    """
    Преобразует вектор состояния x в формат ограничивающего прямоугольника.
    Параметры:
    x (numpy array): Вектор состояния в формате [x, y, s, r]
    score (float, optional): Оценка достоверности ограничивающего прямоугольника
    Возвращает:
    numpy array: Ограничивающий прямоугольник в формате [x1, y1, x2, y2] или [x1, y1, x2, y2, score] если score предоставлен
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker:
    """
    Этот класс представляет внутреннее состояние отдельного отслеживаемого объекта, наблюдаемого как ограничивающий прямоугольник.
    """
    count = 0

    def __init__(self, bbox):
        """
        Инициализирует трекер с использованием начального ограничивающего прямоугольника.
        """
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
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
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
        """
        Обновляет состояние вектора с наблюдаемым ограничивающим прямоугольником.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Продвигает состояние вектора и возвращает предсказанный ограничивающий прямоугольник.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Возвращает текущее состояние ограничивающего прямоугольника.
        """
        return convert_x_to_bbox(self.kf.x)

class Sort:
    """
    Sort: Многоцелевой трекер, использующий фильтры Калмана для отслеживания объектов.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Устанавливает ключевые параметры для трекера.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Обновляет трекер с новыми детекциями.
        Параметры:
        dets (numpy array): Список детекций в формате [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ...]
        Возвращает:
        numpy array: Список отслеживаемых объектов в формате [[x1, y1, x2, y2, id], [x1, y1, x2, y2, id], ...]
        """
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
    """
    Сопоставляет детекции с отслеживаемыми объектами (оба представлены ограничивающими прямоугольниками)
    Возвращает 3 списка: сопоставления, несопоставленные детекции и несопоставленные трекеры
    """
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

@jit
def linear_assignment(cost_matrix):
    """
    Решает задачу линейного назначения для матрицы затрат.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in range(len(y)) if y[i] >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

# Основная часть
import cv2
from ultralytics import YOLO
from sort import Sort

def process_video(video_path):
    model = YOLO("yolov8n.pt")
    tracker = Sort()        # Инициализация трекера SORT
    previous_positions = {} # Словарь для хранения предыдущих позиций объектов
    frame_count = 0         # Счетчик кадров

    cap = cv2.VideoCapture(video_path) # Открытие видеофайла

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Детекция объектов на кадре
        results = model(frame)
        if isinstance(results, list):
            results = results[0]
        bboxes = results.boxes.xyxy.cpu().numpy()

        if bboxes.size == 0:
            continue

        dets = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            conf = bbox[4] if len(bbox) > 4 else 1.0
            dets.append([x1, y1, x2, y2, conf])
        dets = np.array(dets)

        # Обновление трекера с новыми детекциями
        tracks = tracker.update(dets)

        for track in tracks:
            x1, y1, x2, y2, track_id = track
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Расчет скорости объекта
            if track_id in previous_positions:
                prev_center_x, prev_center_y, prev_frame = previous_positions[track_id]
                distance = np.sqrt((center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2)
                speed_m_s = distance / (frame_count - prev_frame)
                speed_km_h = speed_m_s * 3.6
            else:
                speed_km_h = 0.0

            # Сохранение текущей позиции объекта
            previous_positions[track_id] = (center_x, center_y, frame_count)

            # Рисование ограничивающего прямоугольника и текста на кадре
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f'ID: {int(track_id)}', (int(center_x), int(center_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'Speed: {speed_km_h:.2f} km/h', (int(center_x), int(center_y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Вывод информации об объекте в командную строку
            print(f'ID: {int(track_id)}, Speed: {speed_km_h:.2f} km/h, Coordinates: ({x1}, {y1}, {x2}, {y2})')

        # Отображение кадра
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = '/home/u/Desktop/Carla_ROS2/road.mp4'
    process_video(video_path)
