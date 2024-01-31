import cv2 
import mediapipe as mp
import time


class hand_detector():
    def __init__(self, mode=False, max_hands = 2):
        self.mode = mode
        self.max_hands = max_hands

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,min_tracking_confidence=.9)
        self.mp_draw = mp.solutions.drawing_utils 


    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:    
                if draw:
                    self.mp_draw.draw_landmarks(img, handLandmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0, draw=True):
        lm_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_number]
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)             
                lm_list.append([id,cx,cy])
        return lm_list


def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = hand_detector()
    while True: 
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)
        if len(lm_list)!=0:
            pass

        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time

        cv2.putText(img,str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255,0,255),3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()