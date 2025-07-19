from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ファイルパス
image_path = '請求書.png'

# 1. 読み込み（PIL → NumPy → BGR）
img_pil = Image.open(image_path).convert('RGB')
img = np.array(img_pil)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 表示用のRGB変換
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. ノイズ除去（やや弱めのガウシアンブラー）
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# 4. 二値化（適応的二値化：局所領域に強い）
adaptive = cv2.adaptiveThreshold(blurred, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 blockSize=11,  # 奇数
                                 C=2)

# 5. 輪郭検出（小文字や細かい領域も拾える設定）
contours, _ = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if 5 < w < 300 and 5 < h < 100:  # 小さすぎず大きすぎない矩形
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Contours画像もRGBに変換
contour_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

# 6. 結果表示（ステップごと）
titles = ['Original', 'Grayscale', 'Blurred', 'Binary', 'Contours']
images = [img_rgb, gray, blurred, adaptive, contour_rgb]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    cmap = 'gray' if i in [1, 2, 3] else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
#plt.show()

# 7. 切り出し保存フォルダを作成
output_dir = 'crops'
os.makedirs(output_dir, exist_ok=True)

# 8. 有効な輪郭を抽出し、切り出し保存・表示
crop_count = 0
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    
    # フィルタリング条件（前処理と同じ）
    if 5 < w < 300 and 5 < h < 100:
        roi = img[y:y+h, x:x+w]  # 元画像から切り出し（BGR）

        # ファイル保存（PNG）
        save_path = os.path.join(output_dir, f'crop_{crop_count:03}.png')
        cv2.imwrite(save_path, roi)
        crop_count += 1

        # 表示（RGBに変換して matplotlibで）
        #roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        #plt.imshow(roi_rgb)
        plt.title(f'Crop {crop_count}')
        plt.axis('off')
        plt.show()

print(f"{crop_count}個の領域を切り出して「{output_dir}/」に保存しました。")

