"""
Custom annotation function for 5-channel npy image channel annotation and visualization
Uses astropy's HistEqStretch for single-channel image processing and PIL for annotation
"""

import numpy as np
from astropy.io import fits
from astropy.visualization import ImageNormalize, MinMaxInterval, HistEqStretch
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import List, Dict
import glob
import os


def load_npy_image(npy_path: str) -> np.ndarray:
    """
    Load 5-channel npy image
    
    Args:
        npy_path: Path to npy file
        
    Returns:
        5-channel numpy array
    """
    data = np.load(npy_path)
    return data


def extract_channels(image: np.ndarray) -> List[np.ndarray]:
    """
    Extract individual channels from 5-channel image
    
    Args:
        image: 5-channel image array
        
    Returns:
        List containing 5 channels
    """
    channels = []
    for i in range(image.shape[0]):
        channels.append(image[i])
    return channels


def preprocess_channel_with_histeq(channel: np.ndarray) -> Image.Image:
    """
    Apply HistEqStretch to single channel and convert to PIL Image
    
    Args:
        channel: Single channel data
        
    Returns:
        PIL Image (RGB mode)
    """
    # Handle NaN values
    data = np.nan_to_num(channel, nan=np.nanmedian(channel))
    
    # Apply astropy's HistEqStretch for histogram equalization
    hist_eq_stretch = HistEqStretch(data)
    norm = ImageNormalize(interval=MinMaxInterval(), stretch=hist_eq_stretch)
    data_eq = norm(data)
    
    # Clip to 0-1 range
    data_eq = np.clip(data_eq, 0, 1)
    
    # Convert to 0-255 uint8
    img_uint8 = (data_eq * 255).astype(np.uint8)
    
    # Convert to PIL Image (grayscale)
    img = Image.fromarray(img_uint8, mode="L")
    
    # Convert to RGB for subsequent color box drawing
    return img.convert("RGB")


def parse_yolo_txt(txt_path: str, img_width: int, img_height: int) -> List[Dict]:
    """
    Parse YOLO format TXT annotation file
    
    Args:
        txt_path: Path to TXT file
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of detections, each containing bbox (absolute coordinates), conf, cls
    """
    detections = []
    
    if not Path(txt_path).exists():
        print(f"  Warning: TXT file not found: {txt_path}")
        return detections
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 6:
            continue
        
        cls_id = int(parts[0])
        # YOLO format: class_id x_center y_center width height (all normalized 0-1)
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        conf = float(parts[5]) if len(parts) > 5 else 1.0
        
        # Convert to absolute coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Convert to [x1, y1, x2, y2] format
        x1 = (x_center_abs - width_abs / 2)
        y1 = (y_center_abs - height_abs / 2)
        x2 = (x_center_abs + width_abs / 2)
        y2 = (y_center_abs + height_abs / 2)
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'conf': conf,
            'cls': cls_id
        })
    
    return detections


def draw_boxes_pil(img: Image.Image, detections: List[Dict], class_names: List[str]) -> Image.Image:
    """
    Draw detection boxes on image using PIL
    
    Args:
        img: PIL Image (RGB mode)
        detections: List of detections
        class_names: List of class names
        
    Returns:
        PIL Image with detections drawn
    """
    draw = ImageDraw.Draw(img)
    
    # Try to use default font, fallback to default size if fails
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default(size=26)
    
    for det in detections:
        bbox = det['bbox']  # [x1, y1, x2, y2]
        conf = det['conf']
        cls_id = det['cls']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box (green)
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # Draw label (green text)
        # label = f"{class_names[cls_id]}:{conf:.2f}"
        # text_y = max(0, y1 - 20)
        # draw.text((x1, text_y), label, fill="yellow", font=font)
    
    return img


def annotate_channels_from_txt(
    npy_path: str,
    txt_path: str,
    class_names: List[str] = None,
    save_dir: str = "./annotate_output",
    show_plot: bool = True
) -> Dict:
    """
    Read TXT annotation file and annotate each channel of 5-channel npy image (combined into one image)
    
    Args:
        npy_path: Path to npy file
        txt_path: Path to YOLO format TXT annotation file
        class_names: List of class names
        save_dir: Output directory
        show_plot: Whether to display image
        
    Returns:
        Dictionary containing all processing results
    """
    if class_names is None:
        class_names = ['Object']
    
    # Create output directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = load_npy_image(npy_path)
    
    # Extract channels
    channels = extract_channels(image)
    
    # Get image dimensions (assuming 640x640)
    img_height, img_width = channels[0].shape
    
    # Parse TXT file
    detections = parse_yolo_txt(txt_path, img_width, img_height)
    
    # Store results
    results = {
        'image_path': npy_path,
        'txt_path': txt_path,
        'channels': [],
        'detections': detections
    }
    
    # Channel name mapping
    channel_names = ['g', 'r', 'i', 'z', 'y']
    
    # Create image grid
    annotated_images = []
    
    for i, channel in enumerate(channels):
        channel_name = channel_names[i] if i < len(channel_names) else f"ch{i}"
        
        # Apply HistEqStretch and convert to PIL Image
        img_rgb = preprocess_channel_with_histeq(channel)
        
        # Draw annotations using PIL
        img_marked = draw_boxes_pil(img_rgb.copy(), detections, class_names)
        
        # Store results
        channel_result = {
            'channel_index': i,
            'channel_name': channel_name,
            'shape': channel.shape,
            'min_value': float(channel.min()),
            'max_value': float(channel.max()),
            'detections_count': len(detections)
        }
        results['channels'].append(channel_result)
        
        # Add to list
        annotated_images.append(img_marked)
    
    # Horizontally concatenate 5 channel images into one (with gaps)
    GAP = 20  # Gap between channels in pixels
    
    if len(annotated_images) == 5:
        # Create blank image for concatenation (white background)
        single_width, single_height = annotated_images[0].size
        total_width = single_width * 5 + GAP * 4
        combined = Image.new("RGB", (total_width, single_height), color="white")
        
        # Horizontally concatenate 5 channel images (with gaps)
        for i, img in enumerate(annotated_images):
            x_offset = i * (single_width + GAP)
            combined.paste(img, (x_offset, 0))
        
        # Save combined image
        output_path = save_path / f"{Path(npy_path).stem}_5channels_annotated.jpg"
        combined.save(str(output_path), quality=95)
        print(f"✓ Combined annotation saved to: {output_path}")
        
        # Display image in notebook
        if show_plot:
            plt.figure(figsize=(20, 3))
            plt.imshow(combined)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    else:
        print(f"Warning: Only processed {len(annotated_images)} channels, expected 5")
        # Still save processed channels
        if len(annotated_images) > 0:
            single_width, single_height = annotated_images[0].size
            total_width = single_width * len(annotated_images) + GAP * (len(annotated_images) - 1)
            combined = Image.new("RGB", (total_width, single_height), color="white")
            for i, img in enumerate(annotated_images):
                x_offset = i * (single_width + GAP)
                combined.paste(img, (x_offset, 0))
            output_path = save_path / f"{Path(npy_path).stem}_5channels_annotated.jpg"
            combined.save(str(output_path), quality=95)
            print(f"Saved partial processed image: {output_path}")
    
    return results


def batch_annotate_from_txt(
    npy_dir: str,
    txt_dir: str,
    class_names: List[str] = None,
    save_dir: str = "./annotate_output",
    show_plot: bool = False
) -> List[Dict]:
    """
    Batch processing: Read npy files and corresponding TXT annotation files from directories
    
    Args:
        npy_dir: npy file directory
        txt_dir: TXT annotation file directory
        class_names: List of class names
        save_dir: Output directory
        show_plot: Whether to display images
        
    Returns:
        List of processing results for all files
    """
    # Start from TXT files
    txt_files = glob.glob(f"{txt_dir}/*.txt")
    
    if len(txt_files) == 0:
        print(f"Warning: No TXT annotation files found in {txt_dir}")
        return []
    
    print(f"Found {len(txt_files)} TXT annotation files")
    
    all_results = []
    for txt_path in txt_files:
        txt_name = Path(txt_path).stem
        
        # Find corresponding npy file based on TXT filename
        npy_path = Path(npy_dir) / f"{txt_name}.npy"
        
        if not npy_path.exists():
            print(f"⚠ npy file not found: {npy_path}")
            continue
        
        # Process and annotate
        result = annotate_channels_from_txt(
            npy_path=str(npy_path),
            txt_path=str(txt_path),
            class_names=class_names,
            save_dir=save_dir,
            show_plot=show_plot
        )
        all_results.append(result)
    
    print(f"\nProcessing complete! Processed {len(all_results)} files")
    
    return all_results


# Usage example
if __name__ == "__main__":
    # Class names
    class_names = ['LT dwarf']
    
    # Batch process all files
    all_results = batch_annotate_from_txt(
        npy_dir="./images",
        txt_dir="./predict/labels",  # YOLO saved TXT file directory
        class_names=class_names,
        save_dir="./annotate_output",
        show_plot=False
    )
