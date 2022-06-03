import cv2
from defog_sr import defog, SuperResolution
test_img_path = 'Image/5.jpg'

prediction_rgb = cv2.imread(test_img_path)

prediction_rgb = defog(prediction_rgb, denoise=True)

# sr = SuperResolution()
# prediction_rgb = sr.upscale_image(prediction_rgb, denoise=True)

cv2.namedWindow('ORIGIN', cv2.WINDOW_NORMAL)
cv2.imshow('ORIGIN', prediction_rgb)
cv2.waitKey(0)
