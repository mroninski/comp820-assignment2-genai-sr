import json
import os

import cv2
import imagehash
import numpy as np
import torch
import typer
from PIL import Image
from pytorch_msssim import ms_ssim

app = typer.Typer()


def get_image_dimensions(image_path: str) -> tuple:
    """Return the dimensions of the image at the given path."""
    img = Image.open(image_path)
    return img.size  # Returns (width, height)


def resize_image_near_lossless(
    image_path: str, target_width: int, target_height: int
) -> np.ndarray:
    """
    Resize the image using a lossless method.

    For lossless resizing, we use Lanczos resampling which is one of the highest
    quality resampling methods, though true losslessness is technically not possible
    when downscaling. We use OpenCV for this operation.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize with Lanczos4 algorithm which provides high quality
    resized_img = cv2.resize(
        img_rgb, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4
    )

    return resized_img


def validate_resizing_similarity(
    image1_array: np.ndarray, image2_array: np.ndarray, threshold: float = 0.05
) -> float:
    image1_pil = Image.fromarray(image1_array)
    image2_pil = Image.fromarray(image2_array)

    # Calculate perceptual hashes
    phash1 = imagehash.phash(image1_pil)
    phash2 = imagehash.phash(image2_pil)
    ahash1 = imagehash.average_hash(image1_pil)
    ahash2 = imagehash.average_hash(image2_pil)
    dhash1 = imagehash.dhash(image1_pil)
    dhash2 = imagehash.dhash(image2_pil)
    whash1 = imagehash.whash(image1_pil)
    whash2 = imagehash.whash(image2_pil)

    # Calculate hash differences
    phash_diff = phash1 - phash2
    ahash_diff = ahash1 - ahash2
    dhash_diff = dhash1 - dhash2
    whash_diff = whash1 - whash2

    # Get average hash difference
    avg_diff = (phash_diff + ahash_diff + dhash_diff + whash_diff) / 4

    print(f"Averaged Hashed difference between images: {avg_diff}")

    # Determine if images are similar based on threshold
    are_similar = avg_diff <= threshold

    assert are_similar, (
        "The resized image is not similar to the smaller image, please validate they are the same image"
    )

    return avg_diff


def calculate_ms_ssim(
    image1_array: np.ndarray, image2_array: np.ndarray, grayscale: bool = False
):
    """
    Calculate the Multi-Scale Structural Similarity Index (MS-SSIM) between two images
    using pytorch-msssim.

    Parameters:
    image1_path (str): Path to the first image
    image2_path (str): Path to the second image
    grayscale (bool): Whether to convert images to grayscale before comparison

    Returns:
    float: MS-SSIM index value (1.0 means identical images)
    """
    # Load images
    # Convert BGR to RGB (OpenCV loads as BGR)
    if len(image1_array.shape) == 3 and image1_array.shape[2] == 3:
        image1_array = cv2.cvtColor(image1_array, cv2.COLOR_BGR2RGB)
        image2_array = cv2.cvtColor(image2_array, cv2.COLOR_BGR2RGB)

    # Apply grayscale if needed
    if grayscale:
        if len(image1_array.shape) == 3:
            image1_array = cv2.cvtColor(image1_array, cv2.COLOR_RGB2GRAY)
            image2_array = cv2.cvtColor(image2_array, cv2.COLOR_RGB2GRAY)

    # Convert to PyTorch tensors
    if grayscale or len(image1_array.shape) == 2:
        # Add batch and channel dimensions for grayscale [B, C, H, W]
        tensor1 = torch.from_numpy(image1_array).unsqueeze(0).unsqueeze(0).float()
        tensor2 = torch.from_numpy(image2_array).unsqueeze(0).unsqueeze(0).float()
    else:
        # For RGB, rearrange from [H, W, C] to [C, H, W], then add batch dimension
        tensor1 = torch.from_numpy(image1_array.transpose(2, 0, 1)).unsqueeze(0).float()
        tensor2 = torch.from_numpy(image2_array.transpose(2, 0, 1)).unsqueeze(0).float()

    # Normalize to [0, 1]
    tensor1 = tensor1 / 255.0
    tensor2 = tensor2 / 255.0

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor1 = tensor1.to(device)
    tensor2 = tensor2.to(device)

    # Calculate MS-SSIM
    with torch.no_grad():
        msssim_value = ms_ssim(tensor1, tensor2, data_range=1.0, size_average=True)

    return msssim_value.item()


@app.command()
def compare_images(
    image1_path: str = typer.Argument(..., help="Path to the first image file"),
    image2_path: str = typer.Argument(..., help="Path to the second image file"),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the output file with the calculated values",
    ),
):
    """
    Compare two images by resizing the larger one to match the smaller one's dimensions,
    then calculate their similarity using Peak Signal-to-Noise Ratio (PSNR).
    """

    # Validate file paths
    if not os.path.exists(image1_path):
        typer.echo(f"Error: Image file not found: {image1_path}")
        raise typer.Exit(code=1)

    if not os.path.exists(image2_path):
        typer.echo(f"Error: Image file not found: {image2_path}")
        raise typer.Exit(code=1)

    # Get dimensions of both images
    try:
        dim1 = get_image_dimensions(image1_path)
        dim2 = get_image_dimensions(image2_path)

        typer.echo(f"Image 1 dimensions: {dim1[0]}x{dim1[1]}")
        typer.echo(f"Image 2 dimensions: {dim2[0]}x{dim2[1]}")

        # Calculate areas to determine which is smaller
        area1 = dim1[0] * dim1[1]
        area2 = dim2[0] * dim2[1]

        # Determine which image is smaller
        if area1 <= area2:
            smaller_image_path = image1_path
            larger_image_path = image2_path
            target_width, target_height = dim1
        else:
            smaller_image_path = image2_path
            larger_image_path = image1_path
            target_width, target_height = dim2

        typer.echo(f"Smaller image: {os.path.basename(smaller_image_path)}")
        typer.echo(
            f"Resizing {os.path.basename(larger_image_path)} to {target_width}x{target_height}"
        )

        # Load the smaller image
        smaller_img = cv2.imread(smaller_image_path)
        smaller_img_rgb = cv2.cvtColor(smaller_img, cv2.COLOR_BGR2RGB)

        # Resize the larger image using lossless method
        resized_img = resize_image_near_lossless(
            larger_image_path, target_width, target_height
        )

        # Validate the similarity of the resized image
        resized_similarity = validate_resizing_similarity(
            smaller_img_rgb, resized_img, threshold=5.0
        )

        typer.echo(f"Resized similarity:` {resized_similarity}")

        # Calculate SSIM
        ms_ssim_value = calculate_ms_ssim(smaller_img_rgb, resized_img)

        output_data = {
            "ms_ssim": ms_ssim_value,
            "resized_similarity": resized_similarity,
        }

        typer.echo(
            f"Multi-Scale Structural Similarity Index Measure (MS-SSIM): {ms_ssim_value:.2f}"
        )  # Between 0 and 1. More is better -> 1 is identical, 0 is completely different

        if output_file:
            with open(output_file, "w") as f:
                json.dump(output_data, f)
                typer.echo(f"Output file saved to {output_file}")
        else:
            typer.echo(json.dumps(output_data, indent=4))

    except Exception as e:
        typer.echo(f"Error processing images: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
