import os

image_path = r'鉛筆.png'  # 必要に応じてフルパスに変更してもOK

print(f"パス: {image_path}")
print("ファイルが存在するか:", os.path.isfile(image_path))
