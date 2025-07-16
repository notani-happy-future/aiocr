import cv2
import numpy as np
import matplotlib.pyplot as plt

# ファイルパス（画像ファイル名に応じて変更）
image_path = 'C:\python_ohtani_works\\aiocr\\20250713_前処理\\pencil.png'
#image_path = '.\\鉛筆.png'
#image_path = '鉛筆.png'

# 1. 画像読み込み（カラー）
img = cv2.imread(image_path)
if img is None:
    print("画像の読み込みに失敗しました。パスやファイル名を確認してください。")
else:
    print("画像の読み込みに成功しました。")

# 2. グレースケール化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. ノイズ除去（ガウシアンブラー）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. 二値化（Otsuの自動しきい値法）
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. 輪郭検出（文字領域の候補）
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 10 and h > 10:  # 小さすぎるものは除外
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 6. 結果表示（ステップごと）
titles = ['Original', 'Grayscale', 'Blurred', 'Binary', 'Contours']
images = [img, gray, blurred, binary, contour_img]

plt.figure(figsize=(15, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    cmap = 'gray' if i != 0 and i != 4 else None
    plt.imshow(images[i], cmap=cmap)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
