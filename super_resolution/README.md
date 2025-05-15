# Super Resolution

This module provides functionality for upscaling images using the EDSR (Enhanced Deep Super Resolution) model from Hugging Face.

## Features

- Upscale images by 2x and 4x using the EDSR model
- Process multiple images in a directory
- Create separate output folders for each scale factor
- Command-line interface for easy usage

## Installation

The module uses `uv` for dependency management. Install dependencies using:

```bash
uv sync
```

## Usage

Process images in a directory:

```bash
python main.py /path/to/your/images
```

### Options

- `--scales`: List of scale factors to apply (default: [2, 4])
- `--model-name`: Name of the EDSR model to use (default: "eugenesiow/edsr-base")

Example with custom options:

```bash
python main.py /path/to/your/images --scales 2 4 --model-name "eugenesiow/edsr-base"
```

## Output

The script creates separate folders for each scale factor:
- `SR_2x/` for 2x upscaled images
- `SR_4x/` for 4x upscaled images

Each output image is named with the pattern: `{original_name}_SR_{scale}x{extension}` 