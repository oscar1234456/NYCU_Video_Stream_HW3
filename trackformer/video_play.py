import cv2

videoCapture = cv2.VideoCapture('../test_video/test.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"fps:{fps}")
print(f"size:{size}")
print(f"fNUMS:{fNUMS}")

success, frame = videoCapture.read()
print(success)
while success:
    cv2.imshow('windows', frame)
    cv2.waitKey(1000 // int(fps))  # 每個frame須給它 1000毫秒=1秒 // 每秒幾個frames
    success, frame = videoCapture.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()
