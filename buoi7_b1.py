import pygame
import random
import sys
from pygame.locals import *

# Khởi tạo Pygame
pygame.init()

# Cài đặt màn hình
RONG, CAO = 800, 600
man_hinh = pygame.display.set_mode((RONG, CAO))
pygame.display.set_caption("chem hoa qua")

# Màu sắc
TRANG = (255, 255, 255)

# Font chữ và đồng hồ
font = pygame.font.SysFont(None, 36)
dong_ho = pygame.time.Clock()

# Biến trò chơi
hoa_qua = []
diem_so = 0
thoi_gian_choi = 60  # Thời gian chơi là 60 giây
bat_dau_tinh_gio = pygame.time.get_ticks()

# Tải ảnh
anh_nen = pygame.image.load("background.jpg")  # Ảnh nền
anh_nen = pygame.transform.scale(anh_nen, (RONG, CAO))  # Điều chỉnh kích thước

# Tải ảnh hoa quả và ảnh hoa quả bị chém
anh_hoa_qua = [
    (pygame.image.load("qua1.png"), pygame.image.load("qua1 slided.png")),  # Cam
    (pygame.image.load("qua2.png"), pygame.image.load("qua2 slided.png")),  # Xoài
    (pygame.image.load("qua3.png"), pygame.image.load("qua3 slided.png")),  # Táo
]

# Lớp Hoa Quả
class HoaQua:
    def __init__(self, x, y, image, image_sliced):
        self.image = pygame.transform.scale(image, (80, 80))  # Kích thước ảnh hoa quả
        self.image_sliced = pygame.transform.scale(image_sliced, (80, 80))  # Ảnh quả bị chém
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = random.randint(2, 5)  # Điều chỉnh tốc độ chậm hơn (2 đến 5)
        self.sliced = False  # Trạng thái chém

    def roi(self):
        self.rect.y += self.speed  # Hoa quả rơi theo tốc độ

    def ve(self):
        if self.sliced:
            man_hinh.blit(self.image_sliced, self.rect.topleft)  # Vẽ quả đã chém
        else:
            man_hinh.blit(self.image, self.rect.topleft)  # Vẽ quả chưa chém

# Hàm tạo hoa quả mới
def tao_hoa_qua():
    x = random.randint(50, RONG - 50)
    y = random.randint(-100, -50)
    hinh_anh, hinh_anh_sliced = random.choice(anh_hoa_qua)
    hoa_qua_moi = HoaQua(x, y, hinh_anh, hinh_anh_sliced)
    hoa_qua.append(hoa_qua_moi)

# Hàm phát hiện chém quả
def phat_hien_chem(vi_tri_chuot):
    global diem_so
    for qua in hoa_qua:
        if qua.rect.collidepoint(vi_tri_chuot):
            if not qua.sliced:
                diem_so += 10  # Tăng điểm
                qua.sliced = True  # Đánh dấu quả đã bị chém

# Vòng lặp chính của trò chơi
while True:
    man_hinh.blit(anh_nen, (0, 0))  # Vẽ ảnh nền

    # Kiểm tra thời gian còn lại
    thoi_gian_da_troi = (pygame.time.get_ticks() - bat_dau_tinh_gio) / 1000  # Tính bằng giây
    thoi_gian_con_lai = max(0, thoi_gian_choi - thoi_gian_da_troi)

    # Hiển thị thời gian và điểm số
    chu_thoi_gian = font.render(f"Thoi gian con lai: {int(thoi_gian_con_lai)}", True, (0, 0, 0))
    chu_diem_so = font.render(f"Diem so: {diem_so}", True, (0, 0, 0))
    man_hinh.blit(chu_thoi_gian, (20, 20))
    man_hinh.blit(chu_diem_so, (20, 60))

    # Tạo hoa quả ngẫu nhiên
    if random.randint(0, 50) == 0:  # Xác suất ngẫu nhiên để tạo quả
        tao_hoa_qua()

    # Cập nhật và vẽ các hoa quả
    for qua in hoa_qua[:]:
        qua.roi()
        qua.ve()
        if qua.rect.y > CAO:
            hoa_qua.remove(qua)  # Xóa quả khi rơi khỏi màn hình

    # Phát hiện chém quả
    vi_tri_chuot = pygame.mouse.get_pos()
    if pygame.mouse.get_pressed()[0]:  # Kiểm tra khi nhấn chuột trái
        phat_hien_chem(vi_tri_chuot)

    # Kiểm tra hết giờ
    if thoi_gian_con_lai <= 0:
        # Màn hình kết thúc
        chu_game_over = font.render(f"Game Over! diem cua ban: {diem_so}", True, (255, 0, 0))
        man_hinh.blit(chu_game_over, (RONG // 4, CAO // 2))
        pygame.display.flip()
        pygame.time.wait(2000)  # Tạm dừng 2 giây
        pygame.quit()
        sys.exit()

    # Xử lý sự kiện
    for su_kien in pygame.event.get():
        if su_kien.type == QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
    dong_ho.tick(60)  # 60 FPS
