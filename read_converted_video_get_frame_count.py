import cv2

video_paths = ["ffmpeg_output_test_hwaccel.mp4", # 1581.0
               "ffmpeg_output_test_opencv.mp4" # 1581.0
               ]

video_path = video_paths[0]

cap = cv2.VideoCapture(video_path)

# all the same. you can read them safely.

# do not read by time. read by frame.

frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("FRAME COUNT:", frame_count)

success = 1

while success:
    success, image = cap.read()
    cv2.imshow("Video", image)
    cv2.waitKey(0)
    # print(image.shape)
    print('CURSOR:', cap.get(cv2.CAP_PROP_POS_FRAMES)) # starting from 0, after read 1 frame now it is 1
    #
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    #
    # breakpoint()
    # final cursor: 1581 (double times!)
    # which you won't get new frames.
    #
    # to jump to nth frame, set cursor to n-1
    # this is RNN. make it iterative.