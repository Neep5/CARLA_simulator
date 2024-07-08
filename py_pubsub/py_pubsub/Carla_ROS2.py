import rclpy  						# Импорт основной библиотеки для работы с ROS2 на Python
from rclpy.node import Node  				# Импорт базового класса для создания нод
from sensor_msgs.msg import Image  			# Импорт сообщения ROS2 для работы с изображениями
from cv_bridge import CvBridge  			# Импорт библиотеки для преобразования между ROS и OpenCV изображениями
from ultralytics import YOLO  			# Импорт модели YOLO для обнаружения и отслеживания объектов
from ultralytics.solutions import speed_estimation  	# Импорт модели для оценки скорости объектов
import cv2  						# Импорт библиотеки OpenCV для обработки изображений

# Создание класса MinimalSubscriber, который наследуется от Node
class MinimalSubscriber(Node):

    def __init__(self):
        # Вызов конструктора базового класса Node с именем ноды 'minimal_subscriber'
        super().__init__('minimal_subscriber')
        
        # Создание подписчика на топик '/carla/ego_vehicle/rgb_front/image'
        # self.listener_callback будет вызвана при получении нового сообщения
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.listener_callback,
            10)  # Очередь размера 10
        
        # Создание публикатора для сообщений Image на топик 'processed_video_stream'
        self.publisher_ = self.create_publisher(Image, 'processed_video_stream', 10)
        
        # Создание объекта для преобразования между ROS и OpenCV изображениями
        self.bridge = CvBridge()
        
        # Инициализация модели YOLOv8 для обнаружения объектов
        self.model = YOLO("yolov8n.pt")
        
        # Инициализация объекта для оценки скорости обнаруженных объектов
        self.speed_obj = speed_estimation.SpeedEstimator()
        
        # Установка аргументов для модели оценки скорости
        self.speed_obj.set_args(reg_pts=[(0, 360), (1280, 360)], names=self.model.model.names)

    # Функция, которая вызывается при получении нового сообщения от подписчика
    def listener_callback(self, msg):
        # Конвертация ROS2 Image сообщения в изображение OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Обнаружение и отслеживание объектов с использованием модели YOLOv8
        tracks = self.model.track(cv_image, persist=True)

        # Оценка скорости обнаруженных объектов
        cv_image = self.speed_obj.estimate_speed(cv_image, tracks)

        # Конвертация изображения OpenCV обратно в ROS2 Image сообщение
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        # Публикация обработанного изображения на топик 'processed_video_stream'
        self.publisher_.publish(ros_image)
        
        # Логирование информации о публикации обработанного кадра
        self.get_logger().info('Publishing processed video frame')
        
        # Отображение изображения с обнаруженными объектами и оцененной скоростью
        cv2.imshow('Carla Camera', cv_image)
        cv2.waitKey(1)

# Основная функция, которая инициализирует ROS2 и запускает цикл обработки сообщений
def main(args=None):
    # Инициализация ROS2
    rclpy.init(args=args)

    # Создание объекта MinimalSubscriber
    minimal_subscriber = MinimalSubscriber()

    # Запуск цикла обработки сообщений
    rclpy.spin(minimal_subscriber)

    # Уничтожение ноды при завершении работы
    minimal_subscriber.destroy_node()
    rclpy.shutdown()  # Завершение работы ROS2

# Точка входа в программу
if __name__ == '__main__':
    main()


