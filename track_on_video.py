import cv2
import time
import torch
import torchvision
import threading
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from utils import nms


def MouseAction(event, x, y, file, param):
    global x1, y1, mode, ref, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame, (x, y), 20, (255, 0, 0), -1)
        if mode == 0:
            mode = 1
            x1, y1 = x, y
        else:
            mode = 0
            ref = []
            print("取消追蹤")


class MyThread(threading.Thread):
    def __init__(self, target=None, args=(), **kwargs):
        super(MyThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs
        self.__result__ = []

    def run(self):
        if self._target is None:
            return
        self.__result__ = self._target(*self._args, **self._kwargs)

    def get_result(self):
        return self.__result__


def RCNNTrack(model, image):
    img = (torch.as_tensor(image, dtype=torch.float32) / 255).to(device)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    output = model(img)
    boxes = output[0]['boxes'].data.cpu().numpy()
    scores = output[0]['scores'].data.cpu().numpy()
    boxes = nms(0, boxes, scores, iou_threshold=0.2, threshold=0.8)
    global mode
    if mode == 0:
        return boxes
    else:
        refbox = []
        global ref
        # 取得box內圖片
        if ref == []:
            for box in boxes:
                if box[0] < x1 < box[2] and box[1] < y1 < box[3]:
                    print("開始追蹤，座標(%d, %d)" % (x1, y1))
                    print("按任意位置取消追蹤")
                    refbox = [box]
                    break
        else:
            max_sim = 0.2
            for box in boxes:
                sim = ssim(cv2.resize(ref, (int(box[2]) - int(box[0]), int(box[3]) - int(box[1])), interpolation=cv2.INTER_AREA), image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :], multichannel=True)
                if sim > max_sim:
                    print(sim)
                    max_sim = sim
                    refbox = [box]

        if refbox == []:
            print("跟丟了QAQ")
            ref = []
            mode = 0
        else:
            print("跟到了!")
            ref = image[int(refbox[0][1]):int(refbox[0][3]), int(refbox[0][0]):int(refbox[0][2]), :]
        return refbox


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 讀取RCNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 2
model = model.to(device)
in_feature = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes).to(device)
model.load_state_dict(torch.load('./model.pth'))
model.to(device)
model.eval()
# 設定讀取影片
video_path = cv2.VideoCapture("example_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
ret, frame = video_path.read()
video_h, video_w = frame.shape[0], frame.shape[1]
out = cv2.VideoWriter("output_video.mp4", fourcc, 20, (video_w, video_h))
# 建立子執行緒(預測程序)
t = MyThread()
boxes = []
x1, y1 = 0, 0
mode = 0  # 0:顯示所有物體 1:顯示追蹤物體
ref = []

while True:
    # 影片擷取圖片
    with torch.no_grad():
        ret, frame = video_path.read()
        # 若子執行緒完成則取出結果並開始下一子執行緒
        if not t.is_alive():
            boxes = t.get_result()
            t = MyThread(target=RCNNTrack, args=(model, frame))
            t.start()

    if mode == 0:
        for box in boxes:
            cv2.rectangle(frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 220, 0), 2)
    else:
        for box in boxes:
            cv2.rectangle(frame,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 220), 2)
            cv2.putText(frame, "tracking", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 220), 1, cv2.LINE_AA)
    # 顯示處理後的圖片
    cv2.imshow('frame', frame)
    out.write(frame)
    cv2.setMouseCallback('frame', MouseAction)
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機
video_path.release()
out.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()

