import rclpy                                        # Импорт основной библиотеки для работы с ROS2 на Python
from rclpy.node import Node                         # Импорт базового класса для создания нод
from sensor_msgs.msg import Image                   # Импорт сообщения ROS2 для работы с изображениями
from cv_bridge import CvBridge                      # Импорт библиотеки для преобразования между ROS и OpenCV изображениями
from ultralytics import YOLO                        # Импорт модели YOLO для обнаружения и отслеживания объектов
from ultralytics.solutions import speed_estimation  # Импорт модели для оценки скорости объектов
import cv2  

class MinimalSubscriber(Node):  # Создание класса MinimalSubscriber, который наследуется от Node

    def __init__(self):
        super().__init__('minimal_subscriber')  # Конструктор класса MinimalSubscriber
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.listener_callback,
            10)  # Создание подписчика на топик carla_video_stream
        self.publisher_ = self.create_publisher(Image, 'processed_video_stream', 10)            # Создание публикатора для сообщений Image на топик processed_video_stream
        self.bridge = CvBridge()                                                                # Создание объекта для преобразования между ROS и OpenCV изображениями
        self.model = YOLO("yolov8n.pt")  
        self.speed_obj = speed_estimation.SpeedEstimator()                                      # Инициализация модели для оценки скорости объектов
        self.speed_obj.set_args(reg_pts=[(0, 360), (1280, 360)], names=self.model.model.names)  # Установка аргументов для модели оценки скорости

    def listener_callback(self, msg):  # Функция, которая вызывается при получении сообщения
        # Конвертирование ROS2 Image сообщения в OpenCV изображение
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Отслеживание объектов
        tracks = self.model.track(cv_image, persist=True)

        # Оценка скорости объектов
        cv_image = self.speed_obj.estimate_speed(cv_image, tracks)

        # Конвертирование изображения в ROS2 Image сообщение
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Публикация сообщения на топик processed_video_stream
        self.publisher_.publish(ros_image)
        self.get_logger().info('Publishing processed video frame')
        
        # Отображение изображения с обнаруженными объектами
        cv2.imshow('Carla Camera', cv_image)
        cv2.waitKey(1)

def main(args=None):        # Функция main, которая инициализирует ROS2 и запускает цикл обработки сообщений
    rclpy.init(args=args)   # Инициализация ROS2

    minimal_subscriber = MinimalSubscriber()  # Создание объекта MinimalSubscriber

    rclpy.spin(minimal_subscriber)              # Запуск цикла обработки сообщений

    # Уничтожение ноды
    minimal_subscriber.destroy_node()
    rclpy.shutdown()        # Завершение работы ROS2

if __name__ == '__main__':  
    main()
