import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import time

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking')
        
        # Initialize publishers
        self.hand_position_pub = self.create_publisher(Point, 'hand_tracking/hand_position', 10)
        self.image_pub = self.create_publisher(Image, 'hand_tracking/image_raw', 10)
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize camera (fixed to camera 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera")
            return
            
        # Timer variables
        self.p_time = 0
        self.c_time = 0
        
        # Create timer for processing frames (30 Hz)
        self.timer = self.create_timer(1.0/30.0, self.process_frame)
        self.get_logger().info('Hand tracking node initialized')
    
    def process_frame(self):
        success, img = self.cap.read()
        if not success:
            self.get_logger().warn("Failed to read frame from camera")
            return
            
        # Convert image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Highlight wrist point (id=0)
                    if id == 0:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                        
                        # Publish hand position
                        hand_pos = Point()
                        hand_pos.x = float(cx)
                        hand_pos.y = float(cy)
                        hand_pos.z = 0.0
                        self.hand_position_pub.publish(hand_pos)
                
                # Draw all landmarks
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Calculate and display FPS
        self.c_time = time.time()
        fps = 1.0 / (self.c_time - self.p_time) if (self.c_time - self.p_time) > 0 else 0
        self.p_time = self.c_time
        
        
        # Add FPS text to image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
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
        # Cleanup
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()