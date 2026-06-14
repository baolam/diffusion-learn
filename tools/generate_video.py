import os

import cv2
import os
import re

image_folder = 'samples'  # Thư mục chứa ảnh như trong hình của bạn
video_name = 'mnist_evolution.mp4'  # Tên video muốn xuất ra

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

def sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1]) if numbers else 0

images.sort(key=sort_key, reverse=True)

if not images:
    print("No images!")
    exit()

# 3. Đọc bức ảnh đầu tiên để lấy thông số kích thước (Width, Height)
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# 4. Thiết lập các thông số cho VideoWriter
fps = 10  # Bạn có thể tăng/giảm số này để chỉnh tốc độ video (khung hình/giây)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng mã hóa cho file .mp4
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

print(progress := f"Dạ, Forming {len(images)} images to video...")

# 5. Vòng lặp đọc từng ảnh và ghi vào video
for image in images:
    image_path = os.path.join(image_folder, image)
    frame = cv2.imread(image_path)
    
    # Ghi khung hình vào video
    video.write(frame)

# 6. Giải phóng tài nguyên sau khi hoàn thành
video.release()
cv2.destroyAllWindows()

print("Finish")