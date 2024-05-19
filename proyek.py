import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from scipy.stats import chisquare, pearsonr
import sys
sys.path.append('randomness_testsuite-master')
from Complexity import ComplexityTest as ct


def sine_map(x, y, a):
    xn = np.sin(np.pi * a * (y + 3) * x * (1 - x))
    yn = np.sin(np.pi * a * (x + 1 + 3) * y * (1 - y))
    return xn, yn

def generate_key(x0, y0, a, size):
    x = x0
    y = y0
    key = np.zeros(size, dtype=np.uint8)
    for i in range(size):
        x, y = sine_map(x, y, a)
        key[i] = int(x * 256) % 256  # Generate integer value between 0 and 255
    return key


def process_image(image_path, x0, a, mode):
    img = Image.open(image_path)
    img_array = np.array(img)
    flat_img = img_array.flatten()
    key = generate_key(x0, y0, a, flat_img.size)
    
    processed_img = np.bitwise_xor(flat_img, key)
    processed_img = processed_img.reshape(img_array.shape)
    
    return Image.fromarray(np.uint8(processed_img))

def load_image():
    global image_path, original_image, original_img_tk
    image_path = filedialog.askopenfilename()
    if image_path:
        original_image = Image.open(image_path)
        original_img_tk = ImageTk.PhotoImage(original_image)
        panel_original.configure(image=original_img_tk)
        panel_original.image = original_img_tk
        panel_processed.configure(image='')  # Clear the processed image panel

def encrypt_decrypt():
    start_time = time.time()
    progress_bar['value'] = 0
    root.update_idletasks()
    
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    a = float(entry_a.get())
    mode = var_mode.get()
    
    if image_path and mode:
        global processed_image, processed_img_tk, encrypted_image, decrypted_image
        
        original_image = Image.open(image_path)
        original_img_array = np.array(original_image)
        flat_original_img = original_img_array.flatten()
        
        key = generate_key(x0, y0, a, flat_original_img.size)
        
        # Enkripsi gambar
        encrypted_img_array = np.bitwise_xor(flat_original_img, key).reshape(original_img_array.shape)
        encrypted_image = Image.fromarray(np.uint8(encrypted_img_array))
        
        # Dekripsi gambar
        decrypted_img_array = np.bitwise_xor(encrypted_img_array.flatten(), key).reshape(original_img_array.shape)
        decrypted_image = Image.fromarray(np.uint8(decrypted_img_array))
        
        # Tampilkan gambar hasil enkripsi
        encrypted_img_tk = ImageTk.PhotoImage(encrypted_image)
        panel_processed.configure(image=encrypted_img_tk)
        panel_processed.image = encrypted_img_tk
        
        # Hitung MSE dan PSNR
        mse = np.mean((original_img_array - decrypted_img_array) ** 2)
        if mse == 0:
            psnr_value = float('inf')
        else:
            psnr_value = 10 * np.log10((255 ** 2) / mse)
        
        elapsed_time = time.time() - start_time
        progress_label.config(text=f"Elapsed time: {elapsed_time:.2f} seconds")
        messagebox.showinfo("Process Complete", f"Elapsed time: {elapsed_time:.2f} seconds\nPSNR: {psnr_value:.2f} dB")
        
        # Tampilkan gambar asli dan hasil dekripsi untuk perbandingan
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(original_image)
        ax[0].set_title("Original Image")
        ax[0].axis('off')
        
        ax[1].imshow(encrypted_image)
        ax[1].set_title("Encrypted Image")
        ax[1].axis('off')
        
        ax[2].imshow(decrypted_image)
        ax[2].set_title("Decrypted Image")
        ax[2].axis('off')
        
        plt.show()

    progress_bar['value'] = 100
    root.update_idletasks()

def array_to_binary(array):
    return ''.join(format(byte, '025b') for byte in array)

def save_binary_representation(image_path):
    binary_sequence = array_to_binary(np.array(processed_image).flatten())
    txt_file_path = image_path.rsplit('.', 1)[0] + '_binary.txt'
    with open(txt_file_path, 'w') as f:
        f.write(binary_sequence)
        
def save_image():
    if 'processed_image' in globals():
        file_path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[("PNG files", '*.png'), ("JPEG files", '*.jpg')])
        if file_path:
            processed_image.save(file_path)
            save_binary_representation(file_path)
            messagebox.showinfo("Save Image", "Image and binary representation successfully saved!")
    else:
        messagebox.showerror("Save Image", "No processed image to save.")

def sensitivity_analysis():
    x0 = float(entry_x0.get())
    a = float(entry_a.get())
    mode = var_mode.get()
    
    range_start = float(entry_range_start.get())
    range_end = float(entry_range_end.get())
    step = float(entry_step.get())
    
    variations = np.arange(range_start, range_end + step, step)
    
    result_images = []
    result_values = []
    
    for var in variations:
        result_images.append(process_image(image_path, var, a, mode))
        result_values.append(var)
    
    display_sensitivity_results(result_images, result_values)

def display_sensitivity_results(images, values):
    result_window = tk.Toplevel(root)
    result_window.title("Sensitivity Analysis Results")
    
    for i, (img, val) in enumerate(zip(images, values)):
        frame = ttk.Frame(result_window)
        frame.pack(side=tk.LEFT, padx=5, pady=5)

        img_tk = ImageTk.PhotoImage(img)
        panel = ttk.Label(frame, image=img_tk)
        panel.image = img_tk
        panel.pack()

        label = ttk.Label(frame, text=f"x0: {val:.5f}")
        label.pack()


def saveKey():
    try:
        x0 = float(entry_x0.get())
        y0 = float(entry_y0.get())
        a = float(entry_a.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for x0 and a.")
        return

    key_length = 10000  # Example length
    key = generate_key(x0, y0, a, key_length)
    
    # Ask the user where to save the file
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:  # Check if a path was selected
        with open(file_path, "w") as file:
            # Convert the numpy array to binary strings
            binary_strings = [format(byte, '025b') for byte in key]
            # Join all binary strings into a single string separated by newlines
            key_str = '\n'.join(binary_strings)
            file.write(key_str)
        print(f"Key saved to {file_path}")
    
def ergodicity_test():
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    a = float(entry_a.get())
    key_length = 10000  # Example length
    key = generate_key(x0, y0, a, key_length)

    plt.figure()
    plt.hist(key, bins=256, density=True, alpha=0.75, color='blue', label='Generated Key')
    
    expected_freq = [key_length / 256] * 256  # Expected frequency for uniform distribution
    observed_freq, _ = np.histogram(key, bins=256)
    chi2, p_value = chisquare(observed_freq, expected_freq)
    
    plt.title(f"Histogram of Generated Key\nChi-square test: chi2={chi2:.2f}, p-value={p_value:.2e}")
    plt.xlabel("Key Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    messagebox.showinfo("Ergodicity Test Result", f"Chi-square value: {chi2:.2f}, p-value: {p_value:.2e}")

def correlation_test():
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    a = float(entry_a.get())
    key_length = 10000  # Example length
    key = generate_key(x0, y0, a, key_length)
    key_shifted = np.roll(key, -1)  # Geser kunci dengan satu posisi
    corr, p_corr = pearsonr(key[:-1], key_shifted[:-1])  # Hitung korelasi Pearson
    messagebox.showinfo("Correlation Test Result", f"Pearson correlation: {corr:.2f}, p-value: {p_corr:.2e}")

def uaci_npcr_test():
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    a = float(entry_a.get())
    
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Tambahkan gangguan kecil pada gambar asli
    img_array_noisy = img_array.copy()
    img_array_noisy[0, 0] = (img_array_noisy[0, 0] + 1) % 256
    
    # Enkripsi kedua gambar
    flat_img = img_array.flatten()
    flat_img_noisy = img_array_noisy.flatten()
    
    key = generate_key(x0, y0, a, flat_img.size)
    
    encrypted_img = np.bitwise_xor(flat_img, key).reshape(img_array.shape)
    encrypted_img_noisy = np.bitwise_xor(flat_img_noisy, key).reshape(img_array.shape)
    
    # Hitung NPCR
    npcr = np.sum(encrypted_img != img_array) / encrypted_img.size * 100
    
    # Hitung UACI
    uaci = np.sum(np.abs(encrypted_img.astype(np.int16) - img_array.astype(np.int16))) / (encrypted_img.size * 255) * 100
    
    messagebox.showinfo("UACI and NPCR Test Result", f"NPCR: {npcr:.2f}%\nUACI: {uaci:.2f}%")

def psnr_test():
    x0 = float(entry_x0.get())
    y0 = float(entry_y0.get())
    a = float(entry_a.get())
    
    img = Image.open(image_path)
    img_array = np.array(img)
    flat_img = img_array.flatten()
    
    # Enkripsi gambar
    key = generate_key(x0, y0, a, flat_img.size)
    encrypted_img = np.bitwise_xor(flat_img, key).reshape(img_array.shape)
    
    # Dekripsi gambar
    decrypted_img = np.bitwise_xor(encrypted_img.flatten(), key).reshape(img_array.shape)
    
    # Hitung MSE
    mse = np.mean((img_array - decrypted_img) ** 2)
    if mse == 0:
        psnr_value = float('inf')
    else:
        psnr_value = 10 * np.log10((255 ** 2) / mse)
    
    messagebox.showinfo("PSNR Test Result", f"PSNR: {psnr_value:.2f} dB")

root = tk.Tk()
root.title("SINE MAP Image Encryption/Decryption")

# Styling
style = ttk.Style()
style.configure('TLabel', font=('Arial', 12), anchor='center')
style.configure('TButton', font=('Arial', 10))
style.configure('TRadiobutton', font=('Arial', 10))

# Variables
image_path = None
original_image = None
processed_image = None

# Set up frames
frame_controls = ttk.Frame(root, padding="10")
frame_controls.pack(fill=tk.X)

frame_images = ttk.Frame(root, padding="5")
frame_images.pack(fill=tk.BOTH, expand=True)

# Image display panels
panel_original = ttk.Label(frame_images, text="Image Loaded", compound=tk.TOP)
panel_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

panel_processed = ttk.Label(frame_images, text="Image Processed", compound=tk.TOP)
panel_processed.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Progress bar and time label
frame_progress = ttk.Frame(root, padding="3")
frame_progress.pack(fill=tk.X, side=tk.BOTTOM)
progress_bar = ttk.Progressbar(frame_progress, orient='horizontal', mode='determinate')
progress_bar.pack(fill=tk.X, padx=10, pady=2)
progress_label = ttk.Label(frame_progress, text="Elapsed time: 0.00 seconds")
progress_label.pack(fill=tk.X, pady=2)

# Controls for x0 and a
frame_params = ttk.Frame(frame_controls)
frame_params.pack(fill=tk.X, pady=5)

ttk.Label(frame_params, text="x0 (0 < x0 < 1):").pack(side=tk.LEFT)
entry_x0 = ttk.Entry(frame_params, font=('Arial', 10), width=8)
entry_x0.pack(side=tk.LEFT, padx=5)

ttk.Label(frame_params, text="y0 (0 < y0 < 1):").pack(side=tk.LEFT)
entry_y0 = ttk.Entry(frame_params, font=('Arial', 10), width=8)
entry_y0.pack(side=tk.LEFT, padx=5)

ttk.Label(frame_params, text="a (0 < a <= 1):").pack(side=tk.LEFT)
entry_a = ttk.Entry(frame_params, font=('Arial', 10), width=8)
entry_a.pack(side=tk.LEFT, padx=5)

var_mode = tk.StringVar(value="enkripsi")
ttk.Radiobutton(frame_params, text="Enkripsi", variable=var_mode, value="enkripsi").pack(side=tk.LEFT, padx=10)
ttk.Radiobutton(frame_params, text="Dekripsi", variable=var_mode, value="dekripsi").pack(side=tk.LEFT, padx=10)

# Controls for sensitivity analysis
frame_sensitivity = ttk.Frame(frame_controls)
frame_sensitivity.pack(fill=tk.X, pady=5)

ttk.Label(frame_sensitivity, text="Range Start:").pack(side=tk.LEFT)
entry_range_start = ttk.Entry(frame_sensitivity, font=('Arial', 10), width=8)
entry_range_start.pack(side=tk.LEFT, padx=5)

ttk.Label(frame_sensitivity, text="Range End:").pack(side=tk.LEFT)
entry_range_end = ttk.Entry(frame_sensitivity, font=('Arial', 10), width=8)
entry_range_end.pack(side=tk.LEFT, padx=5)

ttk.Label(frame_sensitivity, text="Step:").pack(side=tk.LEFT)
entry_step = ttk.Entry(frame_sensitivity, font=('Arial', 10), width=8)
entry_step.pack(side=tk.LEFT, padx=5)

# Buttons
frame_buttons = ttk.Frame(frame_controls)
frame_buttons.pack(fill=tk.X, pady=5)

ttk.Button(frame_buttons, text="Load Image", command=load_image).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Encrypt/Decrypt", command=encrypt_decrypt).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Save Image", command=save_image).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Generate Key", command=saveKey).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Sensitivity Analysis", command=sensitivity_analysis).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Ergodicity Test", command=ergodicity_test).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="Correlation Test", command=correlation_test).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="UACI and NPCR Test", command=uaci_npcr_test).pack(side=tk.LEFT, padx=10)
ttk.Button(frame_buttons, text="PSNR Test", command=psnr_test).pack(side=tk.LEFT, padx=10)

root.mainloop()
