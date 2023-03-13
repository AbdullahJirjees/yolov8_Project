import cv2
#limitsDown = [X1, Y1, X2, Y2]
limitsDown = [510, 150, 780, 150]
limitsUp = [10, 350, 250, 350]
mask = cv2.imread("../PeopleCounter/testMask.png")

cv2.line(mask, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)
cv2.line(mask, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
cv2.imshow("mask", mask)
cv2.waitKey(0)


