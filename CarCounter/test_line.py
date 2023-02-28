import cv2
limits = [790, 600, 1240, 400]
mask = cv2.imread("../CarCounter/mask3.png")

cv2.line(mask, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

cv2.imshow("mask", mask)
cv2.waitKey(0)