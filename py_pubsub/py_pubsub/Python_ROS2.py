import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2
import numpy as np

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
        
        self.speed_obj = speed_estimation.SpeedEstimator()
        self.speed_obj.set_args(reg_pts=[(0, 360), (1280, 360)], names=self.model.model.names)
        
        self.prev_tracks = []

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.track(cv_image, persist=True)

        tracks = self.process_results(results)

        if not tracks:
            self.get_logger().info('No detections in current frame')
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher_.publish(ros_image)
            cv2.imshow('Carla Camera', cv_image)
            cv2.waitKey(1)
            return

        filtered_tracks = self.filter_overtaking_vehicles(tracks)

        if not filtered_tracks:
            self.get_logger().info('No overtaking vehicles detected in current frame')
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.publisher_.publish(ros_image)
            cv2.imshow('Carla Camera', cv_image)
            cv2.waitKey(1)
            return

        cv_image = self.speed_obj.estimate_speed(cv_image, filtered_tracks)

        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        self.publisher_.publish(ros_image)
        
        self.get_logger().info('Publishing processed video frame')
        
        cv2.imshow('Carla Camera', cv_image)
        cv2.waitKey(1)

        self.prev_tracks = tracks

    def process_results(self, results):
        tracks = []
        for result in results:
            if hasattr(result, 'boxes'):
                for box in result.boxes:
                    tracks.append({
                        'class': self.model.model.names[int(box.cls[0])],
                        'bbox': box.xyxy[0].tolist(),
                        'id': int(box.id[0]) if box.id is not None else None
                    })
        return tracks

    def filter_overtaking_vehicles(self, tracks):
        filtered_tracks = []
        for track in tracks:
            if track['class'] in ['car', 'truck', 'bus']:
                if self.is_overtaking(track):
                    filtered_tracks.append(track)
        return filtered_tracks

    def is_overtaking(self, track):
        if not self.prev_tracks:
            return False
        
        for prev_track in self.prev_tracks:
            if prev_track['id'] == track['id']:
                prev_center = np.array([prev_track['bbox'][0] + prev_track['bbox'][2] / 2,
                                        prev_track['bbox'][1] + prev_track['bbox'][3] / 2])
                curr_center = np.array([track['bbox'][0] + track['bbox'][2] / 2,
                                        track['bbox'][1] + track['bbox'][3] / 2])
                direction = curr_center - prev_center
                if direction[0] > 0:  # Объект движется вправо
                    return True
        return False

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
