import os
from pathlib import Path
from typing import Optional

import typer
from PIL import Image
from super_image import EdsrModel, ImageLoader

app = typer.Typer()


def create_output_dirs(input_folder: Path, scales: list[int]) -> dict[int, Path]:
    """
    Create output directories for each scale factor.
    
    Args:
        input_folder: Path to the input folder
        scales: List of scale factors (e.g., [2, 4])
        
    Returns:
        Dictionary mapping scale factors to their output directories
    """
    output_dirs = {}
    for scale in scales:
        output_dir = input_folder / f"SR_{scale}x"
        output_dir.mkdir(exist_ok=True)
        output_dirs[scale] = output_dir
    return output_dirs


def process_image(
    image_path: Path,
    model: EdsrModel,
    output_path: Path,
    scale: int,
    base_name: str
) -> None:
    """
    Process a single image with the given model and save the result.
    
    Args:
        image_path: Path to the input image
        model: EDSR model to use for upscaling
        output_path: Directory to save the output
        scale: Scale factor being applied
        base_name: Base name of the file without extension
    """
    image = Image.open(image_path)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)
    
    # Create new filename with scale factor
    new_name = f"{base_name}_SR_{scale}x{image_path.suffix}"
    output_file = output_path / new_name
    
    ImageLoader.save_image(preds, output_file)
    typer.echo(f"Processed {image_path.name} -> {new_name}")


@app.command()
def upscale_images(
    input_folder: Path = typer.Argument(
        ...,
        help="Path to the folder containing images to upscale",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    scales: list[int] = typer.Option(
        [2, 4],
        help="Scale factors to apply (e.g., 2 for 2x upscaling)",
    ),
    model_name: str = typer.Option(
        "eugenesiow/edsr-base",
        help="Name of the EDSR model to use from Hugging Face",
    ),
):
    """
    Upscale images in the input folder using the EDSR model.
    Creates separate output folders for each scale factor.
    """
    try:
        # Create output directories
        output_dirs = create_output_dirs(input_folder, scales)
        
        # Load models for each scale
        models = {
            scale: EdsrModel.from_pretrained(model_name, scale=scale)
            for scale in scales
        }
        
        # Process each image
        for image_path in input_folder.glob("*"):
            if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                base_name = image_path.stem.replace("SR_1024x1024", "")
                
                for scale, model in models.items():
                    process_image(
                        image_path,
                        model,
                        output_dirs[scale],
                        scale,
                        base_name
                    )
        
        typer.echo("Successfully processed all images.")
        
    except Exception as e:
        typer.echo(f"Error processing images: {str(e)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app() 