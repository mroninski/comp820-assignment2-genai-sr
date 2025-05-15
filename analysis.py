import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# Set style for high-quality visualizations
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 120

# Colors for consistent visualization
COLOR_2X = "#1f77b4"  # Blue
COLOR_4X = "#ff7f0e"  # Orange
COLOR_CONTROL_2X = "#2ca02c"  # Green
COLOR_CONTROL_4X = "#d62728"  # Red


# Load the data
def load_data():
    """Load all 4 CSV files and return them as dataframes."""
    files = [
        "analysis/2x_results.csv",
        "analysis/4x_results.csv",
        "analysis/control_results_2x.csv",
        "analysis/control_results_4x.csv",
    ]
    dfs = {}

    for file in files:
        try:
            df = pl.read_csv(file)
            # Add a source column to identify the file
            file_key = file.replace(".csv", "")
            df = df.with_columns(
                pl.lit(file_key).alias("DataSource"),
                pl.col("FileKey").cast(pl.Utf8).alias("FileKey"),
            )
            dfs[file_key] = df
        except FileNotFoundError:
            print(
                f"Warning: {file} not found. Please ensure it exists in the working directory."
            )

    return dfs


# Prepare metrics for analysis
def prepare_metrics(dfs):
    """Extract and prepare metrics for analysis."""
    # Metrics to analyze (excluding text fields and file paths)
    numeric_metrics = {
        "scale-invariant-feature-transform_num_keypoints1": "SIFT Keypoints (Original)",
        "scale-invariant-feature-transform_num_keypoints2": "SIFT Keypoints (Super-Resolution)",
        "scale-invariant-feature-transform_num_good_matches": "SIFT Good Matches",
        "scale-invariant-feature-transform_match_ratio": "SIFT Match Ratio",
        "scale-invariant-feature-transform_avg_distance": "SIFT Avg Distance",
        "multi-scale-structural-similarity-index_ms_ssim": "MS-SSIM",
        "multi-scale-structural-similarity-index_resized_similarity": "Average Perceptual Hash Functions",
        "peak-signal-to-noise-ratio_psnr": "PSNR",
        "peak-signal-to-noise-ratio_nsr": "NSR",
        "learned-perceptual-image-patch-similarity_lpips": "LPIPS (Original)",
        "learned-perceptual-image-patch-similarity_resized_similarity": "LPIPS (Super-Resolution)",
    }

    # Create a combined dataframe with only the metrics we need
    combined_df = pl.DataFrame()

    base_cols = ["FileKey", "ImageSource", "Multiplier", "DataSource"]

    for key, df in dfs.items():
        # First we rename the columns to be more descriptive
        df = df.rename(numeric_metrics)

        # Then we select the base columns and the numeric metrics
        subset = df[base_cols + list(numeric_metrics.values())]
        combined_df = pl.concat([combined_df, subset], how="vertical")

    return combined_df, list(numeric_metrics.values())


# Visualization 1: Compare metrics across upscaling factors and control groups
def plot_metrics_comparison(df, metrics):
    """
    Create boxplots comparing each metric across the different datasets.
    This helps identify differences between 2x and 4x scaling, as well as control vs. experimental.
    """

    # Create a folder for outputs if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Create individual boxplots for each metric
    for metric in metrics:
        plt.figure(figsize=(14, 8))

        # Create boxplot with seaborn
        ax = sns.boxplot(
            x="DataSource",
            y=metric,
            data=df,
            palette={
                "analysis/2x_results": COLOR_2X,
                "analysis/4x_results": COLOR_4X,
                "analysis/control_results_2x": COLOR_CONTROL_2X,
                "analysis/control_results_4x": COLOR_CONTROL_4X,
            },
        )

        # Add individual data points for better visibility
        ax = sns.stripplot(
            x="DataSource", y=metric, data=df, color="black", alpha=0.5, jitter=True
        )

        # Customize the plot
        plt.title(f"Comparison of {metric} Across Datasets", fontsize=16)
        plt.xlabel("Dataset", fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"plots/{metric.replace(' ', '_')}_comparison.png")
        plt.close()

    print("Metric comparison plots created successfully.")


# Visualization 2: Correlation heatmap of metrics
def plot_correlation_heatmap(df, metrics):
    """
    Create a correlation heatmap to identify relationships between metrics.
    This helps determine which metrics provide redundant or unique information.
    """
    # Create a correlation matrix for numeric columns
    corr_matrix = df[metrics].corr()

    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create a custom colormap that highlights strong positive and negative correlations
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Plot the heatmap with seaborn
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 10},
    )

    plt.title("Correlation Between Image Quality Metrics", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/metric_correlation_heatmap.png")
    plt.close()

    print("Correlation heatmap created successfully.")


# Visualization 3: Distribution of metrics with z-score to identify outliers
def plot_metric_distributions(df, metrics):
    """
    Plot the distribution of each metric and highlight outliers.
    This helps identify values that deviate significantly from the norm.
    """
    # Calculate z-scores for each metric to identify outliers
    z_scores = pl.DataFrame()
    for metric in metrics:
        z_scores = z_scores.with_columns(
            pl.lit(stats.zscore(df[metric], nan_policy="omit")).alias(
                f"{metric}_zscore"
            )
        )

    # Remove the control from these charts
    df = df.filter(
        pl.col("DataSource").str.contains("analysis/2x_results")
        | pl.col("DataSource").str.contains("analysis/4x_results")
    )

    # Combine with original data
    df_with_zscores = pl.concat([df, z_scores], how="horizontal")

    # Create distribution plots for each metric
    for metric in metrics:
        plt.figure(figsize=(14, 8))

        # Create a GridSpec to arrange the plots
        gs = GridSpec(2, 1, height_ratios=[3, 1])

        # Plot the distribution with KDE
        ax1 = plt.subplot(gs[0])
        ax1 = sns.histplot(
            data=df,
            x=metric,
            hue="DataSource",
            kde=True,
            palette={
                "analysis/2x_results": COLOR_2X,
                "analysis/4x_results": COLOR_4X,
                # "analysis/control_results_2x": COLOR_CONTROL_2X,
                # "analysis/control_results_4x": COLOR_CONTROL_4X,
            },
            alpha=0.8,
        )

        plt.title(f"Distribution of {metric}", fontsize=16)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel("Count", fontsize=14)

        # Plot outliers (z-score > 2 or < -2) on a separate axis
        ax2 = plt.subplot(gs[1])
        outliers = df_with_zscores.filter(pl.col(f"{metric}_zscore").abs() > 2)

        if not outliers.is_empty():
            ax2 = sns.scatterplot(
                data=outliers,
                x=metric,
                y=outliers[f"{metric}_zscore"],
                hue="DataSource",
                palette={
                    "analysis/2x_results": COLOR_2X,
                    "analysis/4x_results": COLOR_4X,
                    "analysis/control_results_2x": COLOR_CONTROL_2X,
                    "analysis/control_results_4x": COLOR_CONTROL_4X,
                },
                s=100,
            )
            plt.ylabel("Z-Score", fontsize=12)
            plt.axhline(y=2, color="r", linestyle="--", alpha=0.5)
            plt.axhline(y=-2, color="r", linestyle="--", alpha=0.5)
            plt.title("Outliers (|Z-Score| > 2)", fontsize=14)
        else:
            plt.text(
                0.5,
                0.5,
                "No outliers detected",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax2.transAxes,
                fontsize=14,
            )

        plt.tight_layout()
        plt.savefig(f"plots/{metric}_distribution.png")
        plt.close()

    print("Metric distribution plots created successfully.")


# Visualization 4: Metrics across upscaling factors (2x vs 4x)
def plot_upscaling_comparison(df, metrics):
    """
    Compare metrics between 2x and 4x upscaling factors.
    This helps identify how image quality changes with higher upscaling factors.
    """
    # Filter for experiment data only (non-control)
    experiment_df = df.filter(
        pl.col("DataSource").str.contains("analysis/2x_results")
        | pl.col("DataSource").str.contains("analysis/4x_results")
    )

    # Extract the upscaling factor (2x or 4x)
    experiment_df = experiment_df.with_columns(
        pl.col("DataSource").str.replace("_results", "").alias("UpscalingFactor")
    )

    # Create comparison plots
    plt.figure(figsize=(16, 12))

    # Set up a grid for subplots
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // n_cols

    for i, metric in enumerate(metrics):
        plt.subplot(n_rows, n_cols, i + 1)

        # Create a box plot comparing 2x and 4x
        sns.boxplot(x="UpscalingFactor", y=metric, data=experiment_df)
        sns.stripplot(
            x="UpscalingFactor",
            y=metric,
            data=experiment_df,
            color="black",
            alpha=0.5,
            jitter=True,
        )

        plt.title(f"{metric.split('_')[-1].upper()}")
        if i % n_cols == 0:
            plt.ylabel("Metric Value")
        else:
            plt.ylabel("")

        if i >= n_metrics - n_cols:
            plt.xlabel("Upscaling Factor")
        else:
            plt.xlabel("")

    plt.tight_layout()
    plt.savefig("plots/upscaling_factor_comparison.png")
    plt.close()

    print("Upscaling factor comparison plot created successfully.")


# Visualization 5: Radar chart comparing key metrics across datasets
def plot_radar_chart(df, metrics):
    """
    Create a radar chart to visualize how all metrics compare across datasets.
    This provides a holistic view of image quality across different conditions.
    """
    # Normalize metrics to [0, 1] range for comparison

    df_normalized = deepcopy(df)
    for metric in metrics:
        if df[metric].min() != df[metric].max():  # Avoid division by zero
            df_normalized = df_normalized.with_columns(
                pl.lit(
                    (df[metric] - df[metric].min())
                    / (df[metric].max() - df[metric].min())
                ).alias(metric)
            )

        else:
            df_normalized = df_normalized.with_columns(pl.lit(0).alias(metric))

    # Calculate mean values for each metric by data source
    grouped = df_normalized.group_by("DataSource").mean()

    # Set up the radar chart
    categories = [m.split("_")[-1] for m in metrics]  # Use shortened metric names
    N = len(categories)

    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Add lines for each dataset
    for i, source in enumerate(grouped["DataSource"]):
        values = grouped[i, metrics].to_pandas().values.flatten().tolist()
        values += values[:1]  # Close the loop

        color = {
            "analysis/2x_results": COLOR_2X,
            "analysis/4x_results": COLOR_4X,
            "analysis/control_results_2x": COLOR_CONTROL_2X,
            "analysis/control_results_4x": COLOR_CONTROL_4X,
        }.get(source, "gray")

        ax.plot(
            angles, values, linewidth=2, linestyle="solid", label=source, color=color
        )
        ax.fill(angles, values, alpha=0.1, color=color)

    # Set category labels
    plt.xticks(angles[:-1], categories, size=12)

    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
    plt.ylim(0, 1)

    # Add legend
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.title("Comparison of Normalized Metrics Across Datasets", size=20, y=1.1)
    plt.tight_layout()
    plt.savefig("plots/radar_chart_comparison.png")
    plt.close()

    print("Radar chart created successfully.")


# Visualization 6: Identify metrics with lower fidelity
def plot_fidelity_analysis(df, metrics):
    """
    Analyze which metrics show lower fidelity across datasets.
    This helps identify where image quality is compromised during super-resolution.
    """
    # For comparison between original and super-resolution
    # Higher values are better for most metrics except:
    # - SIFT avg_distance (lower is better)
    # - LPIPS (lower is better)
    # - NSR (lower is better)

    # Define which metrics are "better" when higher or lower
    higher_better = {
        "SIFT Keypoints (Original)": True,
        "SIFT Keypoints (Super-Resolution)": True,
        "SIFT Good Matches": True,
        "SIFT Match Ratio": True,
        "SIFT Avg Distance": False,  # Lower is better
        "Average Perceptual Hash Functions": False,  # Lower is better
        "MS-SSIM": True,
        "PSNR": True,
        "NSR": False,  # Lower is better
        "LPIPS (Original)": False,  # Lower is better
        "LPIPS (Super-Resolution)": False,  # Lower is better
    }

    df = df.to_pandas()

    # Calculate relative performance scores
    grouped = df.groupby("DataSource")[metrics].mean()

    # Normalize scores from 0 to 1 based on whether higher or lower is better
    normalized_scores = pd.DataFrame(index=grouped.index, columns=metrics)

    for metric in metrics:
        if higher_better[metric]:
            normalized_scores[metric] = (grouped[metric] - grouped[metric].min()) / (
                grouped[metric].max() - grouped[metric].min()
            )
        else:
            # Invert for metrics where lower is better
            normalized_scores[metric] = 1 - (
                grouped[metric] - grouped[metric].min()
            ) / (grouped[metric].max() - grouped[metric].min())

    # Calculate average score across all metrics for each dataset
    normalized_scores["Average_Score"] = normalized_scores.mean(axis=1)

    # Plot average scores
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        normalized_scores.index, normalized_scores["Average_Score"], alpha=0.7
    )

    # Color the bars based on dataset
    for i, bar in enumerate(bars):
        source = normalized_scores.index[i]
        color = {
            "2x_results": COLOR_2X,
            "4x_results": COLOR_4X,
            "control_2x": COLOR_CONTROL_2X,
            "control_4x": COLOR_CONTROL_4X,
        }.get(source, "gray")
        bar.set_color(color)

    plt.title("Average Normalized Quality Score Across Datasets", fontsize=16)
    plt.ylabel("Average Score (Higher is Better)", fontsize=14)
    plt.xlabel("Dataset", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/average_quality_score.png")
    plt.close()

    # Plot individual metric scores to identify low fidelity metrics
    plt.figure(figsize=(14, 10))

    # Transpose for better visualization
    heatmap_data = normalized_scores[metrics].T

    # Create a heatmap
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)

    plt.title("Normalized Quality Scores by Metric and Dataset", fontsize=16)
    plt.ylabel("Metric", fontsize=14)
    plt.xlabel("Dataset", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/quality_score_heatmap.png")
    plt.close()

    # Identify metrics with lowest fidelity
    mean_scores = normalized_scores[metrics].mean()
    lowest_fidelity = mean_scores.nsmallest(3)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(mean_scores.index, mean_scores.values, alpha=0.7)

    # Highlight the lowest fidelity metrics
    for i, bar in enumerate(bars):
        if mean_scores.index[i] in lowest_fidelity.index:
            bar.set_color("red")
        else:
            bar.set_color("skyblue")

    plt.title("Average Quality Score by Metric (Across All Datasets)", fontsize=16)
    plt.ylabel("Average Score (Higher is Better)", fontsize=14)
    plt.xlabel("Metric", fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("plots/metric_quality_scores.png")
    plt.close()

    print("Fidelity analysis plots created successfully.")
    print(f"Metrics with lowest fidelity: {', '.join(lowest_fidelity.index)}")


def analyze_super_resolution_data(df: pl.DataFrame, metrics: list[str]) -> None:
    """
    Analyze super-resolution image data using Polars.

    Args:
        df: A Polars DataFrame containing super-resolution image metrics
    """
    # Print total number of rows
    print(f"Total number of rows: {len(df)}")

    # Print column names
    print("Column names:", df.columns)

    # Helper function to calculate average for a column
    def calculate_average(df, column_name):
        return df.select(pl.col(column_name).mean()).item()

    # Analyze SIFT metrics
    avg_sift_original_keypoints = calculate_average(df, "SIFT Keypoints (Original)")
    avg_sift_sr_keypoints = calculate_average(df, "SIFT Keypoints (Super-Resolution)")
    avg_sift_good_matches = calculate_average(df, "SIFT Good Matches")
    avg_sift_match_ratio = calculate_average(df, "SIFT Match Ratio")
    avg_sift_distance = calculate_average(df, "SIFT Avg Distance")

    print("\nSIFT Metrics:")
    print(f"Average SIFT Keypoints (Original): {avg_sift_original_keypoints:.2f}")
    print(f"Average SIFT Keypoints (Super-Resolution): {avg_sift_sr_keypoints:.2f}")
    print(f"Average SIFT Good Matches: {avg_sift_good_matches:.2f}")
    print(f"Average SIFT Match Ratio: {avg_sift_match_ratio:.4f}")
    print(f"Average SIFT Avg Distance: {avg_sift_distance:.4f}")

    ############################################################
    # Perceptual Quality Metrics
    ############################################################

    # Analyze perceptual quality metrics
    avg_ms_ssim = calculate_average(df, "MS-SSIM")
    avg_hash_functions = calculate_average(df, "Average Perceptual Hash Functions")
    avg_psnr = calculate_average(df, "PSNR")
    avg_nsr = calculate_average(df, "NSR")
    avg_lpips_orig = calculate_average(df, "LPIPS (Original)")
    avg_lpips_sr = calculate_average(df, "LPIPS (Super-Resolution)")

    print("\nPerceptual Quality Metrics:")
    print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")
    print(f"Average Perceptual Hash Functions: {avg_hash_functions:.4f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average NSR: {avg_nsr:.4f}")
    print(f"Average LPIPS (Original): {avg_lpips_orig:.4f}")
    print(f"Average LPIPS (Super-Resolution): {avg_lpips_sr:.4f}")

    ############################################################
    # Group data by multiplier to analyze scaling effects
    ############################################################
    print("\nAnalysis by Multiplier:")
    multiplier_analysis = df.group_by("Multiplier").agg([
        pl.count().alias("Count"),
        pl.col("SIFT Match Ratio").mean().alias("Avg SIFT Match Ratio"),
        pl.col("MS-SSIM").mean().alias("Avg MS-SSIM"),
        pl.col("PSNR").mean().alias("Avg PSNR"),
    ])

    for row in multiplier_analysis.iter_rows(named=True):
        print(f"\nMultiplier: {row['Multiplier']}, Count: {row['Count']}")
        print(f"  Avg SIFT Match Ratio: {row['Avg SIFT Match Ratio']:.4f}")
        print(f"  Avg MS-SSIM: {row['Avg MS-SSIM']:.4f}")
        print(f"  Avg PSNR: {row['Avg PSNR']:.2f} dB")

    # Group data by image source to analyze source effects
    print("\nAnalysis by Image Source:")
    source_analysis = df.group_by("ImageSource").agg([
        pl.count().alias("Count"),
        pl.col("SIFT Match Ratio").mean().alias("Avg SIFT Match Ratio"),
        pl.col("MS-SSIM").mean().alias("Avg MS-SSIM"),
        pl.col("PSNR").mean().alias("Avg PSNR"),
    ])

    for row in source_analysis.iter_rows(named=True):
        print(f"\nSource: {row['ImageSource']}, Count: {row['Count']}")
        print(f"  Avg SIFT Match Ratio: {row['Avg SIFT Match Ratio']:.4f}")
        print(f"  Avg MS-SSIM: {row['Avg MS-SSIM']:.4f}")
        print(f"  Avg PSNR: {row['Avg PSNR']:.2f} dB")

    ############################################################
    # Correlation
    ############################################################
    # Calculate correlation between metrics to identify relationships
    def calculate_correlation(df, field1, field2):
        # Filter out null and NaN values
        valid_pairs = df.filter(
            pl.col(field1).is_not_null()
            & ~pl.col(field1).is_nan()
            & pl.col(field2).is_not_null()
            & ~pl.col(field2).is_nan()
        )

        if len(valid_pairs) < 2:
            return 0

        # Convert to NumPy for correlation calculation
        x = valid_pairs.select(field1).to_numpy().flatten()
        y = valid_pairs.select(field2).to_numpy().flatten()

        # Calculate correlation coefficient
        n = len(x)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_xx = (x * x).sum()
        sum_yy = (y * y).sum()

        numerator = n * sum_xy - sum_x * sum_y
        denominator = np.sqrt(
            (n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)
        )

        return 0 if denominator == 0 else numerator / denominator

    print("\nCorrelation Analysis:")
    print(
        f"SIFT Match Ratio vs MS-SSIM: {calculate_correlation(df, 'SIFT Match Ratio', 'MS-SSIM'):.4f}"
    )
    print(
        f"SIFT Match Ratio vs PSNR: {calculate_correlation(df, 'SIFT Match Ratio', 'PSNR'):.4f}"
    )
    print(f"MS-SSIM vs PSNR: {calculate_correlation(df, 'MS-SSIM', 'PSNR'):.4f}")
    print(f"PSNR vs NSR: {calculate_correlation(df, 'PSNR', 'NSR'):.4f}")


def analyze_genai_super_resolution(df: pl.DataFrame) -> None:
    """
    Analyze quality of GenAI super-resolution images.

    Args:
        df: A Polars DataFrame containing super-resolution image metrics
    """
    # Filter for GenAI images
    genai_images = df.filter(pl.col("ImageSource") == "GenAI")

    # Log the number of GenAI images
    print(f"Number of GenAI images: {len(genai_images)}")

    # Check column names to confirm we have all metrics
    print("Column names:", df.columns)

    # Define metrics and whether higher or lower values are better
    metrics = {
        "SIFT Match Ratio": {"better": "higher"},
        "MS-SSIM": {"better": "higher"},
        "PSNR": {"better": "higher"},
        "Average Perceptual Hash Functions": {"better": "higher"},
        "SIFT Avg Distance": {"better": "lower"},
        "NSR": {"better": "lower"},
        "LPIPS (Super-Resolution)": {"better": "lower"},
    }

    # Find min and max for each metric
    for metric in metrics:
        if metric in genai_images.columns:
            metrics[metric]["min"] = genai_images.select(pl.col(metric).min()).item()
            metrics[metric]["max"] = genai_images.select(pl.col(metric).max()).item()

    print("Metric ranges:", metrics)

    # Convert the DataFrame to a list of dictionaries for easier processing
    genai_images_list = genai_images.to_dicts()

    # Calculate a quality score for each image
    for image in genai_images_list:
        score_sum = 0
        weight_sum = 0

        for metric, info in metrics.items():
            if (
                metric in image
                and image[metric] is not None
                and not np.isnan(image[metric])
            ):
                min_val = info["min"]
                max_val = info["max"]

                # Skip if min equals max (no variation in this metric)
                if min_val == max_val:
                    continue

                # Normalize the score based on whether higher or lower is better
                if info["better"] == "higher":
                    normalized_score = (image[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_score = 1 - (
                        (image[metric] - min_val) / (max_val - min_val)
                    )

                # Weight assignment based on metric importance
                weight = 1
                if metric == "MS-SSIM":
                    weight = 2  # Emphasizing perceptual metrics
                if metric == "PSNR":
                    weight = 1.5  # Common baseline metric

                score_sum += normalized_score * weight
                weight_sum += weight

        # Calculate weighted average score
        image["qualityScore"] = score_sum / weight_sum if weight_sum > 0 else 0

    # Sort by quality score
    genai_images_list.sort(key=lambda x: x["qualityScore"])

    # Get worst and best images
    worst_image = genai_images_list[0]
    best_image = genai_images_list[-1]

    print("\nWORST GenAI Super-Resolution Image:")
    print({
        "FileKey": worst_image["FileKey"],
        "Multiplier": worst_image["Multiplier"],
        "DataSource": worst_image["DataSource"],
        "QualityScore": worst_image["qualityScore"],
        "SIFT Match Ratio": worst_image["SIFT Match Ratio"],
        "MS-SSIM": worst_image["MS-SSIM"],
        "PSNR": worst_image["PSNR"],
        "NSR": worst_image["NSR"],
        "Average Perceptual Hash Functions": worst_image[
            "Average Perceptual Hash Functions"
        ],
        "LPIPS (Super-Resolution)": worst_image["LPIPS (Super-Resolution)"],
    })

    print("\nBEST GenAI Super-Resolution Image:")
    print({
        "FileKey": best_image["FileKey"],
        "Multiplier": best_image["Multiplier"],
        "DataSource": best_image["DataSource"],
        "QualityScore": best_image["qualityScore"],
        "SIFT Match Ratio": best_image["SIFT Match Ratio"],
        "MS-SSIM": best_image["MS-SSIM"],
        "PSNR": best_image["PSNR"],
        "NSR": best_image["NSR"],
        "Average Perceptual Hash Functions": best_image[
            "Average Perceptual Hash Functions"
        ],
        "LPIPS (Super-Resolution)": best_image["LPIPS (Super-Resolution)"],
    })

    # Calculate distribution of quality scores
    scores = [img["qualityScore"] for img in genai_images_list]
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std_dev = np.sqrt(variance)

    print("\nQuality Score Distribution:")
    print({"min": min(scores), "max": max(scores), "mean": mean, "stdDev": std_dev})

    # Get the top 3 worst and best for more context
    print("\nTop 3 WORST GenAI Super-Resolution Images:")
    for img in genai_images_list[:3]:
        print({
            "FileKey": img["FileKey"],
            "QualityScore": img["qualityScore"],
            "Multiplier": img["Multiplier"],
            "MS-SSIM": img["MS-SSIM"],
            "PSNR": img["PSNR"],
        })

    print("\nTop 3 BEST GenAI Super-Resolution Images:")
    for img in reversed(genai_images_list[-3:]):
        print({
            "FileKey": img["FileKey"],
            "QualityScore": img["qualityScore"],
            "Multiplier": img["Multiplier"],
            "MS-SSIM": img["MS-SSIM"],
            "PSNR": img["PSNR"],
        })


# Main function to run all visualizations
def main():
    """Main function to run the analysis and create visualizations."""
    print("Starting analysis of super-resolution image quality metrics...")

    # Load data
    print("Loading data files...")
    dfs = load_data()

    # Check if files were loaded successfully
    if not dfs:
        print("No data files were loaded. Please check file paths and try again.")
        return

    print(f"Successfully loaded {len(dfs)} data files.")

    # Prepare metrics for analysis
    print("Preparing metrics for analysis...")
    combined_df, metrics = prepare_metrics(dfs)

    # Print basic statistics
    print(f"Total number of samples: {combined_df.shape[0]}")
    for source in combined_df["DataSource"].unique():
        count = combined_df.filter(pl.col("DataSource") == source).shape[0]
        print(f"  {source}: {count} samples")

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Compare metrics across datasets
    plot_metrics_comparison(combined_df, metrics)

    # 2. Create correlation heatmap
    plot_correlation_heatmap(combined_df, metrics)

    # 3. Plot metric distributions and identify outliers
    plot_metric_distributions(combined_df, metrics)

    # 4. Compare metrics across upscaling factors
    plot_upscaling_comparison(combined_df, metrics)

    # 5. Create radar chart for overall comparison
    plot_radar_chart(combined_df, metrics)

    # 6. Analyze metrics with lower fidelity
    plot_fidelity_analysis(combined_df, metrics)

    # 7. Print out some basic statistics for the super-resolution data
    analyze_super_resolution_data(combined_df, metrics)

    # 8. Print out

    print(
        "\nAnalysis complete! All visualizations have been saved to the 'plots' directory."
    )

    combined_df.write_csv("analysis/combined_df.csv")


if __name__ == "__main__":
    main()
