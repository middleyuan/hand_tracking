import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking')
        
        # Initialize publishers
        self.left_hand_pose_pub = self.create_publisher(Pose, 'hand_tracking/left_hand_pose', 10)
        self.right_hand_pose_pub = self.create_publisher(Pose, 'hand_tracking/right_hand_pose', 10)
        self.left_gripper_pub = self.create_publisher(String, 'hand_tracking/left_gripper_command', 10)
        self.right_gripper_pub = self.create_publisher(String, 'hand_tracking/right_gripper_command', 10)
        self.image_pub = self.create_publisher(Image, 'hand_tracking/image_raw', 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            return
            
        # Create timer for processing frames (30 Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.process_frame)
        self.get_logger().info('Hand tracking node initialized')
    
    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            self.get_logger().warn("Failed to read frame from camera")
            return
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_label = results.multi_handedness[idx].classification[0].label  # "Left" or "Right"
                is_left = hand_label == "Left"
                
                hand_pose_pub = self.left_hand_pose_pub if is_left else self.right_hand_pose_pub
                gripper_pub = self.left_gripper_pub if is_left else self.right_gripper_pub
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    if id == 0:  # Wrist point
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0) if is_left else (0, 0, 255), cv2.FILLED)
                        
                        hand_pos = Pose()
                        hand_pos.position.x = 0.2
                        hand_pos.position.y = float(cx)
                        hand_pos.position.z = -float(cy)
                        hand_pos.orientation.x = 1.0
                        hand_pos.orientation.y = 0.0
                        hand_pos.orientation.z = 0.0
                        hand_pos.orientation.w = 0.0
                        hand_pose_pub.publish(hand_pos)
                
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                fingertip_ids = [4, 8, 12, 16, 20]
                fingertips = [hand_landmarks.landmark[i] for i in fingertip_ids]
                dists = [((f.x - fingertips[0].x) ** 2 + (f.y - fingertips[0].y) ** 2) ** 0.5 for f in fingertips[1:]]
                mean_dist = np.mean(dists)
                
                gripper_cmd = String()
                gripper_cmd.data = 'close' if mean_dist < 0.1 else 'open'
                gripper_pub.publish(gripper_cmd)
                
                cv2.putText(img, f'{hand_label} Hand: {gripper_cmd.data}', (10, 50 + 50 * idx), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0) if is_left else (0, 0, 255), 2)
                cv2.putText(img, f'({cx}, {-cy})', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0) if is_left else (0, 0, 255), 2)
        
        ros_image = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.image_pub.publish(ros_image)
        
    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
