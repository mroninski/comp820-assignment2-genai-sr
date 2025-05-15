import json
import os
from typing import Annotated

import cv2
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer()


def compare_images_sift(img1_path, img2_path, visualization=False):
    """
    Compare two images using SIFT features and return similarity metrics.

    Parameters:
    img1_path (str): Path to first image
    img2_path (str): Path to second image
    visualization (bool): Whether to show visualization of matches

    Returns:
    dict: Dictionary containing similarity metrics
    """

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Convert images to grayscale
    # The greyscale is a requirement for SIFT
    # Due to the nature of the SIFT algorithm
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, mask=None)
    kp2, des2 = sift.detectAndCompute(gray2, mask=None)

    # Print number of keypoints detected
    print(f"Keypoints in image 1: {len(kp1)}")
    print(f"Keypoints in image 2: {len(kp2)}")

    # Match descriptors using FLANN (Fast Library for Approximate Nearest Neighbors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate similarity metrics
    match_ratio = (
        len(good_matches) / min(len(kp1), len(kp2))
        if min(len(kp1), len(kp2)) > 0
        else 0
    )
    avg_distance = (
        np.mean([m.distance for m in good_matches]) if good_matches else float("inf")
    )

    results = {
        "num_keypoints1": len(kp1),
        "num_keypoints2": len(kp2),
        "num_good_matches": len(good_matches),
        "match_ratio": match_ratio,
        "avg_distance": avg_distance,
    }

    if visualization and good_matches:
        # Visualize matches
        # Create an output image to draw matches on
        img_matches_display = np.empty(
            (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3),
            dtype=np.uint8,
        )
        img_matches = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            img_matches_display,  # Pass the created output image
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(
            f"SIFT Matches: {len(good_matches)} good matches out of {min(len(kp1), len(kp2))} possible"
        )
        plt.show()

    return results


def interpret_similarity(results):
    """
    Interpret the similarity metrics and provide a human-readable assessment.

    Parameters:
    results (dict): Dictionary containing similarity metrics

    Returns:
    str: Interpretation of similarity
    """
    match_ratio = results["match_ratio"]
    avg_distance = results["avg_distance"]

    # Interpret match ratio
    if match_ratio > 0.4:
        similarity = "Very similar or identical (possibly the same image with different resolution)"
    elif match_ratio > 0.2:
        similarity = "Similar images (likely different views of the same object/scene)"
    elif match_ratio > 0.1:
        similarity = "Some similarities detected (possibly related content)"
    else:
        similarity = "Different images"

    # Additional information on average distance (lower is better)
    distance_quality = ""
    if avg_distance < 100:
        distance_quality = "High-quality matches"
    elif avg_distance < 200:
        distance_quality = "Medium-quality matches"
    else:
        distance_quality = "Low-quality matches"

    return f"{similarity} - {distance_quality} (Match ratio: {match_ratio:.3f}, Avg distance: {avg_distance:.2f})"


@app.command()
def compare_images(
    image1_path: Annotated[str, typer.Argument(help="Path to the first image file")],
    image2_path: Annotated[str, typer.Argument(help="Path to the second image file")],
    visualization: bool = typer.Option(
        False, "--visualization", help="Show visualization of matches"
    ),
    output_file: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to the output file with the calculated values",
    ),
) -> None:
    """
    Compare two images using SIFT features and return similarity metrics.
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
        # Compare the images using SIFT
        sift_results = compare_images_sift(
            image1_path, image2_path, visualization=visualization
        )

        typer.echo(f"SIFT results: {sift_results}")
        # Interpret the results
        # Greater than 0.4 is a very close match, closer to 1 is better
        # Avg distance is the average distance between the matches
        # Lower is better (goes from 0 to 300)
        interpretation = interpret_similarity(sift_results)
        typer.echo(f"Interpretation: {interpretation}")

        output_data = {
            "similarity_interpretation": interpretation,
        }

        for k, v in sift_results.items():
            output_data[k] = v

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
