#!/usr/bin/env python3
"""
img_compare.py
--------------
Compare two images directly:
- Computes PSNR on RGB and on Y (luma)
- Computes SSIM(Y) if scikit-image is installed (optional)
- Saves: side-by-side image, absolute diff, amplified diff

Usage:
  python img_compare.py -a A.png -b B.png -o out_dir --resize-to-a

Dependencies: pillow, numpy (optional: scikit-image for SSIM)
  pip install pillow numpy
  pip install scikit-image   # (optional) for SSIM
"""
import argparse, os
import numpy as np
from PIL import Image

def to_np_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8), img

def to_luma_np(img_rgb_arr):
    # ITU-R BT.601 luma approx: Y = 0.299 R + 0.587 G + 0.114 B
    r = img_rgb_arr[...,0].astype(np.float32)
    g = img_rgb_arr[...,1].astype(np.float32)
    b = img_rgb_arr[...,2].astype(np.float32)
    y = 0.299*r + 0.587*g + 0.114*b
    return y

def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-12:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)

def side_by_side(imgA, imgB, out_path):
    W = imgA.width + imgB.width
    H = max(imgA.height, imgB.height)
    canvas = Image.new("RGB", (W, H), (0,0,0))
    canvas.paste(imgA, (0,0))
    canvas.paste(imgB, (imgA.width,0))
    canvas.save(out_path)

def abs_diff_img(A_rgb, B_rgb, out_path):
    diff = np.abs(A_rgb.astype(np.int16) - B_rgb.astype(np.int16)).astype(np.uint8)
    Image.fromarray(diff).save(out_path)

def amp_diff_img(A_rgb, B_rgb, out_path, factor=6):
    diff = (A_rgb.astype(np.int16) - B_rgb.astype(np.int16)) * factor + 128
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(out_path)

def maybe_ssim_y(yA, yB):
    try:
        from skimage.metrics import structural_similarity as ssim
        score, _ = ssim(yA, yB, data_range=255, full=True)
        return float(score)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--image-a", required=True, help="Reference image path")
    ap.add_argument("-b", "--image-b", required=True, help="Compared image path")
    ap.add_argument("-o", "--out-dir", required=True, help="Output directory")
    ap.add_argument("--resize-to-a", action="store_true",
                    help="Resize B to A's size (bicubic) before comparing")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    A_arr, A_img = to_np_rgb(args.image_a)
    B_arr, B_img = to_np_rgb(args.image_b)

    if args.resize_to_a and (A_img.size != B_img.size):
        B_img = B_img.resize(A_img.size, resample=Image.BICUBIC)
        B_arr = np.array(B_img, dtype=np.uint8)

    if A_img.size != B_img.size:
        raise SystemExit(f"Image sizes differ: {A_img.size} vs {B_img.size}. Use --resize-to-a if appropriate.")

    # Metrics
    psnr_rgb = psnr(A_arr, B_arr)
    yA = to_luma_np(A_arr)
    yB = to_luma_np(B_arr)
    psnr_y = psnr(yA, yB)
    ssim_y = maybe_ssim_y(yA, yB)

    # Outputs
    side_by_side(A_img, B_img, os.path.join(args.out_dir, "side_by_side.png"))
    abs_diff_img(A_arr, B_arr, os.path.join(args.out_dir, "diff_abs.png"))
    amp_diff_img(A_arr, B_arr, os.path.join(args.out_dir, "diff_amp_x6.png"))

    # Report
    print("Comparison report")
    print("-----------------")
    print(f"Image A: {args.image_a}")
    print(f"Image B: {args.image_b}")
    print(f"PSNR (RGB): {psnr_rgb:.2f} dB")
    print(f"PSNR (Y)  : {psnr_y:.2f} dB")
    if ssim_y is None:
        print("SSIM (Y)  : (skipped; install scikit-image to compute)")
    else:
        print(f"SSIM (Y)  : {ssim_y:.4f}")
    print(f"Outputs saved in: {args.out_dir}")
    print(" - side_by_side.png")
    print(" - diff_abs.png")
    print(" - diff_amp_x6.png")

if __name__ == "__main__":
    main()
