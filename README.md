# SINE MAP Image Encryption/Decryption

This project demonstrates image encryption and decryption using a sine map. The application is built with Python and Tkinter for the graphical user interface (GUI), and it provides several functionalities, including encryption, decryption, sensitivity analysis, and statistical tests.

## Project Overview

This project is a final assignment for the Cryptography course. It aims to demonstrate the principles of image encryption and decryption using a chaotic sine map. The project showcases the application of cryptographic techniques to secure image data and includes various analyses and tests to evaluate the effectiveness of the encryption method.

## Features

- **Image Encryption/Decryption**: Encrypt and decrypt images using a sine map.
- **PSNR Calculation**: Calculate Peak Signal-to-Noise Ratio (PSNR) to measure the quality of the decrypted image.
- **Sensitivity Analysis**: Analyze how variations in initial conditions affect the encryption process.
- **Ergodicity Test**: Perform a chi-square test to check the uniform distribution of the generated key.
- **Correlation Test**: Perform a Pearson correlation test on the generated key.
- **UACI and NPCR Test**: Calculate the Unified Average Changing Intensity (UACI) and Number of Pixels Change Rate (NPCR).

## Prerequisites

- Python 3.x
- Required Python libraries: numpy, pillow, matplotlib, tkinter, scipy

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sine-map-encryption.git
    cd sine-map-encryption
    ```

2. **Install the required libraries**:
    ```bash
    pip install numpy pillow matplotlib scipy
    ```

## Usage

Run the `proyek.py` script to start the application:

```bash
python proyek.py
```

## Main GUI Elements

Main GUI Elements
1. Load Image: Load the image you want to encrypt/decrypt.
2. Encrypt/Decrypt: Perform encryption or decryption based on the selected mode.
3. Save Image: Save the processed image.
4. Sensitivity Analysis: Analyze the sensitivity of the encryption process to initial conditions.
5. Ergodicity Test: Perform an ergodicity test on the generated key.
6. Correlation Test: Perform a correlation test on the generated key.
7. UACI and NPCR Test: Calculate UACI and NPCR metrics.
8. PSNR Test: Calculate the PSNR of the decrypted image.


## Contributing
Fork the repository.
```
Create a new branch (git checkout -b feature-branch).

Commit your changes (git commit -am 'Add new feature').

Push to the branch (git push origin feature-branch).

Create a new Pull Request.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.