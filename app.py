# file: preprocessing_app.py

import os
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def list_file_count(folder):
    try:
        files = os.listdir(folder)
        return len(files)
    except Exception as e:
        st.error(f"Gagal membaca folder {folder}: {e}")
        return 0

def tampilkan_gambar(folder, judul_koleksi, maks_tampil=6):
    ekstensi_gambar = ['.jpg', '.jpeg', '.png']
    gambar_ditemukan = [f for f in os.listdir(folder) if f.lower().endswith(tuple(ekstensi_gambar))][:maks_tampil]

    st.subheader(f"Contoh gambar - {judul_koleksi}")
    cols = st.columns(len(gambar_ditemukan))
    for i, nama_file in enumerate(gambar_ditemukan):
        path = os.path.join(folder, nama_file)
        img = Image.open(path)
        cols[i].image(img, caption=nama_file, use_column_width=True)

def ubah_ke_grayscale(folder_input, folder_output):
    os.makedirs(folder_output, exist_ok=True)
    ekstensi = [".png", ".jpg", ".jpeg"]
    for file in os.listdir(folder_input):
        if file.lower().endswith(tuple(ekstensi)):
            img = Image.open(os.path.join(folder_input, file))
            img_gray = img.convert("L")
            img_gray.save(os.path.join(folder_output, file))

def ubah_ukuran(folder_input, folder_output, ukuran=(256, 256)):
    os.makedirs(folder_output, exist_ok=True)
    for file in os.listdir(folder_input):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder_input, file)
            img = Image.open(path).resize(ukuran, Image.Resampling.LANCZOS)
            img.save(os.path.join(folder_output, file))

def manual_otsu(image):
    pixel_number = image.size
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(image, bins=256, range=(0,256))
    final_thresh, final_var = -1, -1
    total_mean = np.sum(np.arange(256) * his) / pixel_number
    weight_b, mean_b = 0.0, 0.0
    for i in range(256):
        weight_b += his[i] * mean_weight
        weight_f = 1 - weight_b
        if weight_b == 0 or weight_f == 0:
            continue
        mean_b += i * his[i] * mean_weight
        mean_f = (total_mean - mean_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > final_var:
            final_var, final_thresh = var_between, i
    return final_thresh

def dilate(img_bin, kernel_size=3, iterations=1):
    pad = kernel_size // 2
    img_padded = np.pad(img_bin, pad, mode='constant')
    for _ in range(iterations):
        temp = np.zeros_like(img_bin)
        for y in range(img_bin.shape[0]):
            for x in range(img_bin.shape[1]):
                window = img_padded[y:y+kernel_size, x:x+kernel_size]
                temp[y,x] = 255 if np.any(window == 255) else 0
        img_padded = np.pad(temp, pad, mode='constant')
    return temp

def erode(img_bin, kernel_size=3, iterations=1):
    pad = kernel_size // 2
    img_padded = np.pad(img_bin, pad, mode='constant')
    for _ in range(iterations):
        temp = np.zeros_like(img_bin)
        for y in range(img_bin.shape[0]):
            for x in range(img_bin.shape[1]):
                window = img_padded[y:y+kernel_size, x:x+kernel_size]
                temp[y,x] = 255 if np.all(window == 255) else 0
        img_padded = np.pad(temp, pad, mode='constant')
    return temp

def lung_polygon_mask(image_gray):
    h, w = image_gray.shape
    mask_img = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    left_poly = [
        (w*0.25, h*0.10), (w*0.20, h*0.20), (w*0.18, h*0.30),
        (w*0.16, h*0.40), (w*0.16, h*0.50), (w*0.18, h*0.60),
        (w*0.20, h*0.70), (w*0.25, h*0.80), (w*0.35, h*0.85),
        (w*0.42, h*0.80), (w*0.45, h*0.70), (w*0.45, h*0.30),
        (w*0.42, h*0.20), (w*0.35, h*0.12)
    ]
    right_poly = [
        (w*0.55, h*0.30), (w*0.55, h*0.20), (w*0.58, h*0.12),
        (w*0.65, h*0.08), (w*0.72, h*0.12), (w*0.78, h*0.20),
        (w*0.80, h*0.30), (w*0.82, h*0.40), (w*0.82, h*0.50),
        (w*0.80, h*0.60), (w*0.78, h*0.70), (w*0.72, h*0.78),
        (w*0.65, h*0.82), (w*0.58, h*0.78), (w*0.55, h*0.70)
    ]
    draw.polygon(left_poly, fill=255)
    draw.polygon(right_poly, fill=255)
    return np.array(mask_img)

st.title("Preprocessing Gambar Citra X-ray")

folder_input = st.text_input("Masukkan path folder TB (misal: TB.533):", "TB.533")
folder_output = "hasil_preprocessing"

if st.button("Jalankan Preprocessing"):
    gray_folder = os.path.join(folder_output, "grayscale")
    resized_folder = os.path.join(folder_output, "resized")
    masked_folder = os.path.join(folder_output, "masked")

    ubah_ke_grayscale(folder_input, gray_folder)
    ubah_ukuran(gray_folder, resized_folder)

    sample_file = next((f for f in os.listdir(resized_folder) if f.endswith('.jpg')), None)
    if sample_file:
        img_path = os.path.join(resized_folder, sample_file)
        img = Image.open(img_path).convert("L")
        img_np = np.array(img)

        thresh = manual_otsu(img_np)
        otsu_mask = (img_np > thresh).astype(np.uint8) * 255
        mask_dilated = dilate(otsu_mask, 3, 2)
        mask_closed = erode(mask_dilated, 3, 2)
        polygon_mask = lung_polygon_mask(img_np)
        final_mask = np.where(polygon_mask > 0, img_np, 0)

        st.subheader("Contoh Proses pada 1 Gambar")
        fig, axs = plt.subplots(1, 5, figsize=(20,4))
        for ax, data, title in zip(axs,
                                   [img_np, otsu_mask, mask_dilated, mask_closed, final_mask],
                                   ["Asli", "Otsu", "Dilasi", "Erosi", "Masked"]):
            ax.imshow(data, cmap='gray')
            ax.set_title(title)
            ax.axis("off")
        st.pyplot()

        os.makedirs(masked_folder, exist_ok=True)
        for file in os.listdir(resized_folder):
            if file.endswith(".jpg"):
                img = Image.open(os.path.join(resized_folder, file)).convert("L")
                img_np = np.array(img)
                masked = np.where(polygon_mask > 0, img_np, 0)
                Image.fromarray(masked).save(os.path.join(masked_folder, file))

        st.success("Preprocessing selesai. Gambar telah disimpan ke folder.")