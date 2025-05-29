import cv2


a = cv2.imread('/nas/gregory/smile/data/detect/images/train/00018.png')
b = cv2.imread('/home/vsfh/Desktop/image.jpg')
c = cv2.imread('/nas/gregory/smile/data/mmask/train/00018.png')

a[c>0] = b[c>0]
cv2.imwrite('a.png', a)