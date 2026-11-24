import cv2
import numpy as np
from pathlib import Path

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

board = np.full((600, 800), 255, dtype=np.uint8)

for marker_id, x, y in [(1, 80, 80), (2, 520, 80), (3, 300, 340)]:
    if hasattr(aruco, "generateImageMarker"):
        marker = aruco.generateImageMarker(dictionary, marker_id, 200)
    else:
        marker = aruco.drawMarker(dictionary, marker_id, 200)

    board[y:y+200, x:x+200] = marker

cv2.imwrite("aruco_board.png", board)
print("ArUco board created")
