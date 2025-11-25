import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ================== GIAO DIỆN CHÍNH ==================
root = Tk()
root.title("Phân đoạn ảnh - K-means & Thresholding")
root.geometry("1200x700")
root.configure(bg="#F4F6F7")

# -------------- Biến toàn cục --------------
img_path = None
img_original = None
img_display = None
panel_original = None
panel_kmeans = None
panel_otsu = None
panel_manual = None
K = 3  # Số cụm K mặc định
DISPLAY_SIZE = (280, 280)
last_otsu_t = None


# ================== HÀM CHỌN ẢNH ==================
def choose_image():
    global img_path, img_original, img_display
    filetypes = [("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    img_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=filetypes)
    if not img_path:
        return

    img_original = cv2.imread(img_path)
    if img_original is None:
        messagebox.showerror("Lỗi", "Không đọc được ảnh! Hãy chọn lại.")
        return

    # Hiển thị ảnh gốc và tính trước các kết quả phân đoạn để so sánh
    show_all_segments()
    label_status.config(text=f"Đã chọn ảnh: {img_path.split('/')[-1]}")


# ================== HÀM HIỂN THỊ ẢNH ==================
def show_image(img, panel, size=DISPLAY_SIZE):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, size)
    im = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=im)
    panel.imgtk = imgtk
    panel.config(image=imgtk)


def compute_kmeans(img, k):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def compute_otsu(img, t_override=None):
    # If t_override is None, compute Otsu threshold automatically and return (img_bgr, t)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if t_override is None:
        t, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = float(t)
    else:
        t = float(t_override)
        _, thresh = cv2.threshold(blur, int(t), 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), t


def compute_manual(img, thresh_val=127):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def show_all_segments():
    global img_original, img_display, K
    if img_original is None:
        return
    img = cv2.resize(img_original, DISPLAY_SIZE)

    # gốc
    show_image(img, panel_original)

    # K-means
    img_k = compute_kmeans(img, K)
    show_image(img_k, panel_kmeans)

    # Otsu (compute and set slider if available)
    img_o, t = compute_otsu(img, None)
    show_image(img_o, panel_otsu)
    try:
        # store and update otsu slider/label only if present
        global last_otsu_t
        last_otsu_t = t
        if 'otsu_thresh_slider' in globals():
            otsu_thresh_slider.set(int(round(t)))
            otsu_label_value.config(text=f"T = {int(round(t))}")
    except Exception:
        pass

    # Manual
    manual_t = int(manual_thresh_slider.get()) if 'manual_thresh_slider' in globals() else 127
    img_m = compute_manual(img, manual_t)
    show_image(img_m, panel_manual)

    # mặc định lưu ảnh hiển thị là K-means
    img_display = img_k
    label_status.config(text=f"Hiển thị: Ảnh gốc, K-means (K={K}), Otsu, Manual")


# ================== PHÂN ĐOẠN ẢNH ==================
def segment_image(method):
    global img_original, img_display, K

    if img_original is None:
        messagebox.showwarning("Cảnh báo", "Hãy chọn ảnh trước!")
        return
    img = cv2.resize(img_original, DISPLAY_SIZE)

    if method == "kmeans":
        img_display = compute_kmeans(img, K)
        show_image(img_display, panel_kmeans)
        label_status.config(text=f"Phân đoạn K-means (K={K}) hoàn tất.")

    elif method == "otsu":
        img_display = compute_otsu(img)
        show_image(img_display, panel_otsu)
        label_status.config(text="Phân đoạn bằng Threshold Otsu hoàn tất.")

    elif method == "manual":
        img_display = compute_manual(img, 127)
        show_image(img_display, panel_manual)
        label_status.config(text="Phân đoạn bằng Threshold cố định (127) hoàn tất.")


# ================== LƯU KẾT QUẢ ==================
def save_result():
    global img_display
    if img_display is None:
        messagebox.showwarning("Cảnh báo", "Chưa có ảnh kết quả để lưu!")
        return

    save_path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
                                             title="Lưu kết quả phân đoạn")
    if save_path:
        cv2.imwrite(save_path, img_display)
        messagebox.showinfo("Thành công", f"Đã lưu ảnh tại:\n{save_path}")


# ================== TĂNG GIẢM GIÁ TRỊ K ==================
def update_k(val):
    global K
    K = int(val)
    label_k_value.config(text=f"K = {K}")
    # nếu đã có ảnh, cập nhật lại các phân đoạn để K-means thay đổi theo slider
    try:
        if img_original is not None:
            show_all_segments()
    except Exception:
        pass


# ================== GIAO DIỆN TKINTER ==================
title = Label(root, text="SO SÁNH PHÂN ĐOẠN ẢNH BẰNG K-MEANS & THRESHOLDING",
              bg="#F4F6F7", fg="#2E4053", font=("Arial", 18, "bold"))
title.pack(pady=10)

frame_images = Frame(root, bg="#F4F6F7")
frame_images.pack()

panel_original = Label(frame_images, bg="#D6DBDF")
panel_original.grid(row=0, column=0, padx=10, pady=10)
panel_kmeans = Label(frame_images, bg="#D6DBDF")
panel_kmeans.grid(row=0, column=1, padx=10, pady=10)
panel_otsu = Label(frame_images, bg="#D6DBDF")
panel_otsu.grid(row=0, column=2, padx=10, pady=10)
panel_manual = Label(frame_images, bg="#D6DBDF")
panel_manual.grid(row=0, column=3, padx=10, pady=10)

Label(frame_images, text="Ảnh gốc", font=("Arial", 13, "bold"), bg="#F4F6F7").grid(row=1, column=0)
Label(frame_images, text="K-means", font=("Arial", 13, "bold"), bg="#F4F6F7").grid(row=1, column=1)
Label(frame_images, text="Threshold Otsu", font=("Arial", 13, "bold"), bg="#F4F6F7").grid(row=1, column=2)
panel_manual_title = Label(frame_images, text="Threshold 127", font=("Arial", 13, "bold"), bg="#F4F6F7")
panel_manual_title.grid(row=1, column=3)

frame_controls = Frame(root, bg="#F4F6F7")
frame_controls.pack(pady=15)

btn_choose = Button(frame_controls, text="📂 Chọn ảnh", font=("Arial", 12, "bold"), bg="#5DADE2", fg="white",
                    command=choose_image, width=12)
btn_choose.grid(row=0, column=0, padx=10)

btn_manual = Button(frame_controls, text="Threshold 127", font=("Arial", 12, "bold"), bg="#F39C12", fg="white",
                    command=lambda: segment_image("manual"), width=14)
btn_manual.grid(row=0, column=1, padx=10)

btn_otsu = Button(frame_controls, text="Threshold Otsu", font=("Arial", 12, "bold"), bg="#E67E22", fg="white",
                  command=lambda: segment_image("otsu"), width=14)
btn_otsu.grid(row=0, column=2, padx=10)

btn_kmeans = Button(frame_controls, text="K-means", font=("Arial", 12, "bold"), bg="#28B463", fg="white",
                    command=lambda: segment_image("kmeans"), width=12)
btn_kmeans.grid(row=0, column=3, padx=10)

btn_save = Button(frame_controls, text="💾 Lưu kết quả", font=("Arial", 12, "bold"), bg="#45B39D", fg="white",
                  command=save_result, width=14)
btn_save.grid(row=0, column=4, padx=10)

frame_k = Frame(root, bg="#F4F6F7")
frame_k.pack(pady=10)
Label(frame_k, text="Chọn số cụm K:", font=("Arial", 12, "bold"), bg="#F4F6F7").pack(side=LEFT)
slider_k = Scale(frame_k, from_=2, to=8, orient=HORIZONTAL, length=200, command=update_k)
slider_k.set(3)
slider_k.pack(side=LEFT, padx=10)
label_k_value = Label(frame_k, text="K = 3", font=("Arial", 12), bg="#F4F6F7")
label_k_value.pack(side=LEFT)

# Manual threshold control
frame_manual_control = Frame(root, bg="#F4F6F7")
frame_manual_control.pack(pady=6)
Label(frame_manual_control, text="Manual Threshold:", font=("Arial", 12, "bold"), bg="#F4F6F7").pack(side=LEFT)
manual_thresh_slider = Scale(frame_manual_control, from_=0, to=255, orient=HORIZONTAL, length=300)
manual_thresh_slider.set(127)
manual_thresh_slider.pack(side=LEFT, padx=10)
label_manual_value = Label(frame_manual_control, text="T = 127", font=("Arial", 12), bg="#F4F6F7")
label_manual_value.pack(side=LEFT)

def update_manual_label(val):
    label_manual_value.config(text=f"T = {int(val)}")
    # nếu đã chọn ảnh, cập nhật vùng manual ngay
    try:
        if img_original is not None:
            show_all_segments()
    except Exception:
        pass

    # cập nhật tiêu đề panel và nút manual để hiển thị giá trị hiện tại
    try:
        panel_manual_title.config(text=f"Threshold {int(val)}")
    except Exception:
        pass
    try:
        btn_manual.config(text=f"Threshold {int(val)}")
    except Exception:
        pass

manual_thresh_slider.config(command=update_manual_label)

# Otsu threshold control (initialized when image selected)
frame_otsu_control = Frame(root, bg="#F4F6F7")
frame_otsu_control.pack(pady=6)
Label(frame_otsu_control, text="Otsu Threshold:", font=("Arial", 12, "bold"), bg="#F4F6F7").pack(side=LEFT)
otsu_thresh_slider = Scale(frame_otsu_control, from_=0, to=255, orient=HORIZONTAL, length=300)
otsu_thresh_slider.set(0)
otsu_thresh_slider.pack(side=LEFT, padx=10)
otsu_label_value = Label(frame_otsu_control, text="T = -", font=("Arial", 12), bg="#F4F6F7")
otsu_label_value.pack(side=LEFT)

def update_otsu_label(val):
    # Show the adjusted value and update the Otsu panel only
    try:
        otsu_label_value.config(text=f"T = {int(val)}")
        if img_original is not None:
            img = cv2.resize(img_original, DISPLAY_SIZE)
            img_o, _ = compute_otsu(img, int(val))
            show_image(img_o, panel_otsu)
    except Exception:
        pass

otsu_thresh_slider.config(command=update_otsu_label)

label_status = Label(root, text="Chưa chọn ảnh.", bg="#F4F6F7", fg="#2E4053", font=("Arial", 11, "italic"))
label_status.pack(pady=8)

root.mainloop()
