import cv2
import mss
import numpy as np

ASSET = "assets/tab_shiurim.png"
CONFIDENCE = 0.6

with mss.mss() as sct:
    img = np.array(sct.grab(sct.monitors[1]))[:, :, :3]
    screen = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

template_bgr = cv2.imread(ASSET)
template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)

print("max match:", max_val)
print("best location:", max_loc)
