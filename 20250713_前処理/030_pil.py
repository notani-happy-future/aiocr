from PIL import Image
import numpy as np
import cv2

image_path = 'pencil.png'  # 日本語のままでOK

try:
    img_pil = Image.open(image_path).convert('RGB')
    img = np.array(img_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    print("✅ 読み込み成功")
except Exception as e:
    print("❌ 読み込み失敗:", e)
