import rclpy  # Импорт основной библиотеки для работы с ROS2 на Python
from rclpy.node import Node  # Импорт базового класса для создания нод

from std_msgs.msg import String  # Импорт сообщения ROS2, которое содержит строку


class MinimalPublisher(Node):  # Создание класса MinimalPublisher, который наследуется от Node

    def __init__(self):
        super().__init__('minimal_publisher')  # Конструктор класса MinimalPublisher
        self.publisher_ = self.create_publisher(String, 'topic', 10)  # Создание публикатора для сообщения String на топик topic
        timer_period = 0.5  # seconds  # Период таймера в секундах
        self.timer = self.create_timer(timer_period, self.timer_callback)  # Создание таймера, который вызывает функцию timer_callback каждые 0,5 секунды
        self.i = 0  # Счетчик для инкрементации номера сообщения

    def timer_callback(self):  # Функция, которая вызывается каждые 0,5 секунды
        msg = String()  # Создание сообщения String
        msg.data = 'Hello World: %d' % self.i  # Установка данных сообщения
        self.publisher_.publish(msg)  # Публикация сообщения на топик topic
        self.get_logger().info('Publishing: "%s"' % msg.data)  # Вывод сообщения в лог
        self.i += 1  # Инкремент счетчика


def main(args=None):  # Функция main, которая инициализирует ROS2 и запускает цикл обработки сообщений
    rclpy.init(args=args)  # Инициализация ROS2

    minimal_publisher = MinimalPublisher()  # Создание объекта MinimalPublisher

    rclpy.spin(minimal_publisher)  # Запуск цикла обработки сообщений

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()  # Уничтожение ноды
    rclpy.shutdown()  # Завершение работы ROS2


if __name__ == '__main__':  # Основная часть кода, которая вызывает функцию main
    main()
    
