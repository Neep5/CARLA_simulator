import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import onnxruntime as ort
import numpy as np
import math

# ... (Остальная часть кода остается без изменений)

def img_transform(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1600, int(320 / 0.6)))

    img = np.transpose(img, (2, 0, 1))/255

    img = img - np.reshape([0.485, 0.456, 0.406], (3, 1, 1))
    img = img / np.reshape([0.229, 0.224, 0.225], (3, 1, 1))
    img = img[np.newaxis, :, -320:, :]

    return np.float32(img)
    
def softmax(x):
    x = np.array(x)
    return (math.e ** x)/(np.sum(math.e ** x))


def pred2coords(outputs, original_image_width, original_image_height):
    local_width = 1
    row_anchor = np.linspace(0.42, 1, 72)
    col_anchor = np.linspace(0, 1, 81)
    row_lane_idx = [1, 2]
    col_lane_idx = [0, 3]
    coords = []

    pred = {'loc_row': outputs[0],
            'loc_col': outputs[1],
            'exist_row': outputs[2],
            'exist_col': outputs[3]}

    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = np.argmax(pred['loc_row'], 1)
    valid_row = np.argmax(pred['exist_row'], 1)

    max_indices_col = np.argmax(pred['loc_col'], 1)
    valid_col = np.argmax(pred['exist_col'], 1)

    for i in row_lane_idx:
        tmp = []
        if valid_row[0, :, i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0, k, i]:

                    all_ind = list(
                        range(
                            max(0, max_indices_row[0, k, i] - local_width),
                            min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1
                        )
                    )

                    out_tmp = np.sum(softmax(pred['loc_row'][0, all_ind, k, i]) * np.float32(all_ind)) + 0.5
                    out_tmp = out_tmp / (num_grid_row - 1) * original_image_width

                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))

            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0, :, i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0, k, i]:
                    all_ind = list(
                        range(
                            max(0, max_indices_col[0, k, i] - local_width),
                            min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1
                        )
                    )

                    out_tmp = np.sum(softmax(pred['loc_col'][0, all_ind, k, i]) * np.float32(all_ind)) + 0.5
                    out_tmp = out_tmp / (num_grid_col - 1) * original_image_height

                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))

            coords.append(tmp)

    return coords

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/carla/ego_vehicle/rgb_front/image',
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.session = ort.InferenceSession('/home/u/ROS_2/pred2coords.onnx')


    def image_callback(self, msg):
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        #cv2.imshow('cv_image',cv_image)
        #cv2.waitKey(1)
        transformed_image = img_transform(cv_image)
        #print(transformed_image.shape)
        outputs = self.session.run(None, {'input': transformed_image})
        #print(outputs)
        coords = pred2coords(outputs, original_image_width=cv_image.shape[1], original_image_height=cv_image.shape[0])

        if not len(coords):
            return
        #print(coords[0][0])
        #visualized_image = visualize_lanes(cv_image, coords)
        cv2.circle(cv_image,coords[0][0],1,(0,0,255),-1)
        # Отображение изображения на экране
        cv2.imshow('Detected Lanes', cv_image)#visualized_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)

    # Уничтожение всех окон OpenCV после остановки узла
    #cv2.destroyAllWindows()

    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
