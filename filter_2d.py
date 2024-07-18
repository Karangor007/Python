import cv2
import numpy as np

def calc_psf(filter_size, angle):
    psf = np.zeros(filter_size)
    center = (filter_size[1] // 2, filter_size[0] // 2)
    cv2.ellipse(psf, center, (0, center[0] // 2), angle, 0, 360, 255, -1)
    psf /= psf.sum()
    return psf

def wiener_filter(input_img, psf, nsr):
    input_fft = np.fft.fft2(input_img)
    psf_fft = np.fft.fft2(psf, s=input_img.shape)
    psf_fft_conj = np.conj(psf_fft)
    psf_fft_abs2 = np.abs(psf_fft) ** 2
    wiener_filter = psf_fft_conj / (psf_fft_abs2 + nsr)
    result_fft = input_fft * wiener_filter
    result = np.fft.ifft2(result_fft)
    result = np.abs(result)
    return result

def main():
    img_path = 'images/image_blur_6.jpg'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    filter_size = (21, 21)
    motion_angle = 30
    nsr = 0.01  # Noise to Signal Ratio
    
    psf = calc_psf(filter_size, motion_angle)
    deblurred_img = wiener_filter(img, psf, nsr)
    
    cv2.imshow('Original', img)
    cv2.imshow('Deblurred', deblurred_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
