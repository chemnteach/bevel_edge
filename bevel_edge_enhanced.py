import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OriginalSimpleHDC:
    THRESHOLD = 50
    DIMENSIONS = 10000

    def __init__(self):
        """Initialize original simple HDC."""
        self.good_images = []
        self.good_filenames = []
        self.bad_images = []
        self.bad_filenames = []
        self.image_size = None
        self.pixel_hypervectors = self.generate_pixel_hypervectors()

    def log_time(self, start_time, operation_name):
        """Log elapsed time for an operation."""
        elapsed = time.time() - start_time
        message = f"{operation_name} completed in {elapsed:.2f} seconds"
        logging.info(message)
        print(message)
        return time.time()

    def generate_pixel_hypervectors(self):
        """Generate random hypervectors for each possible pixel value."""
        np.random.seed(42)
        return {i: np.random.choice([-1, 1], self.DIMENSIONS) for i in range(256)}

    def crop_upper_right_quadrant(self, image):
        """Crop the upper right quadrant of the image."""
        width, height = image.size
        return image.crop((width // 2, 0, width, height // 2))

    def segment_image(self, image):
        """Segment the image to get the wafer region."""
        gray_image = image.convert("L")
        image_array = np.array(gray_image)

        # Find boundaries
        top_boundary = 0
        for y in range(image_array.shape[0]):
            if np.mean(image_array[y, :]) > self.THRESHOLD:
                top_boundary = y
                break

        bottom_boundary = image_array.shape[0]
        for y in range(image_array.shape[0] - 1, -1, -1):
            if np.mean(image_array[y, :]) > self.THRESHOLD:
                bottom_boundary = y
                break

        return image.crop((0, top_boundary, image.width, bottom_boundary))

    def encode_image_simple(self, image):
        """Simple HDC encoding - back to basics."""
        image_array = np.array(image)
        hypervector = np.zeros(self.DIMENSIONS)

        for pixel in image_array.flatten():
            hypervector += self.pixel_hypervectors[pixel]

        return np.sign(hypervector)

    def process_image(self, image_path, edge_folder_path):
        """Process single image - right quadrant only."""
        try:
            image = Image.open(image_path)
        except IOError:
            logging.error(f"Error opening image {image_path}")
            return None, None

        filename = os.path.basename(image_path)

        # Right quadrant only
        right_quadrant = self.crop_upper_right_quadrant(image)
        wafer_region = self.segment_image(right_quadrant)

        # Save processed image
        save_path = os.path.join(edge_folder_path, filename)
        wafer_region.save(save_path)

        return wafer_region.convert("L"), filename

    def calculate_review_bands(self, good_distances, bad_distances):
        """Calculate review bands using your original approach."""
        good_distances = np.array(good_distances)
        bad_distances = np.array(bad_distances)

        # Statistics
        good_mean = np.mean(good_distances)
        good_std = np.std(good_distances)
        good_max = np.max(good_distances)
        bad_min = np.min(bad_distances)

        logging.info("=== ORIGINAL SIMPLE DISTANCE ANALYSIS ===")
        logging.info(
            f"Good images: mean={good_mean:.1f}, std={good_std:.1f}, "
            f"max={good_max:.1f}"
        )
        logging.info(f"Bad images: min={bad_min:.1f}")

        # Use your original approach
        sorted_good = sorted(good_distances, reverse=True)
        if len(sorted_good) >= 4:
            confident_good_threshold = sorted_good[3]  # 4th highest good
        else:
            confident_good_threshold = np.percentile(good_distances, 85)

        # Bad threshold: Use 2nd lowest bad
        sorted_bad = sorted(bad_distances)
        if len(sorted_bad) >= 2:
            second_lowest_bad = sorted_bad[1]
        else:
            second_lowest_bad = np.min(bad_distances)

        # Create review band
        if confident_good_threshold >= second_lowest_bad:
            # Overlap exists - create minimal band
            overlap_center = (confident_good_threshold + second_lowest_bad) / 2
            band_width = min(200, (good_max - good_mean))
            confident_good_threshold = overlap_center - band_width / 2
            confident_bad_threshold = overlap_center + band_width / 2
        else:
            confident_bad_threshold = second_lowest_bad
            gap = confident_bad_threshold - confident_good_threshold
            if gap > 400:
                midpoint = (confident_good_threshold + confident_bad_threshold) / 2
                confident_good_threshold = midpoint - 100
                confident_bad_threshold = midpoint + 100

        # Safety check: ensure no bad images pass as good
        bad_in_good_zone = np.sum(bad_distances <= confident_good_threshold)
        if bad_in_good_zone > 0:
            bad_in_good = bad_distances[bad_distances <= confident_good_threshold]
            confident_good_threshold = np.max(bad_in_good) + 10
            logging.warning(
                f"Adjusted good threshold to {confident_good_threshold:.1f} "
                "to avoid false negatives"
            )

        # Final validation
        if confident_good_threshold >= confident_bad_threshold:
            confident_bad_threshold = confident_good_threshold + 50

        logging.info("=== ORIGINAL REVIEW BANDS ===")
        logging.info(f"Confident GOOD threshold: {confident_good_threshold:.1f}")
        logging.info(f"Confident BAD threshold: {confident_bad_threshold:.1f}")
        logging.info(
            f"Review band width: "
            f"{confident_bad_threshold - confident_good_threshold:.1f}"
        )

        # Verify no false negatives
        false_negatives = np.sum(bad_distances <= confident_good_threshold)
        if false_negatives > 0:
            logging.warning(
                f"⚠️ WARNING: {false_negatives} bad images would pass " "as good!"
            )

        return confident_good_threshold, confident_bad_threshold

    def classify_with_bands(self, distances, filenames, good_threshold, bad_threshold):
        """Classify images using review bands."""
        results = []

        for distance, filename in zip(distances, filenames):
            if distance <= good_threshold:
                classification = "ASSUME_GOOD"
                color = "green"
            elif distance >= bad_threshold:
                classification = "ASSUME_BAD"
                color = "red"
            else:
                classification = "MANUAL_REVIEW"
                color = "orange"

            results.append(
                {
                    "filename": filename,
                    "distance": distance,
                    "classification": classification,
                    "color": color,
                }
            )

        return results

    def run_original_analysis(self, good_folder_path, bad_folder_path):
        """Run original simple HDC analysis."""
        total_start = time.time()

        # Process good images
        logging.info("Processing good images with original simple encoding...")
        edge_folder_path = os.path.join(good_folder_path, "Edge_Original")
        os.makedirs(edge_folder_path, exist_ok=True)

        image_paths = [
            os.path.join(good_folder_path, filename)
            for filename in os.listdir(good_folder_path)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]

        good_images = []
        good_filenames = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, path, edge_folder_path)
                for path in image_paths
            ]
            for future in as_completed(futures):
                processed_image, filename = future.result()
                if processed_image is not None:
                    good_images.append(processed_image)
                    good_filenames.append(filename)

        # Resize images
        max_width = max(image.width for image in good_images)
        max_height = max(image.height for image in good_images)
        self.image_size = (max_width, max_height)
        good_images = [img.resize(self.image_size) for img in good_images]

        # Create HDC prototype
        logging.info("Creating original HDC prototype...")
        good_hvs = [self.encode_image_simple(img) for img in good_images]
        prototype = np.sign(np.sum(good_hvs, axis=0))

        # Calculate distances for good images
        good_distances = []
        for img in good_images:
            hv = self.encode_image_simple(img)
            distance = np.sum(hv != prototype)
            good_distances.append(distance)

        # Process bad images
        logging.info("Processing bad images with original simple encoding...")
        edge_folder_path = os.path.join(bad_folder_path, "Edge_Original")
        os.makedirs(edge_folder_path, exist_ok=True)

        image_paths = [
            os.path.join(bad_folder_path, filename)
            for filename in os.listdir(bad_folder_path)
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]

        bad_images = []
        bad_filenames = []

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_image, path, edge_folder_path)
                for path in image_paths
            ]
            for future in as_completed(futures):
                processed_image, filename = future.result()
                if processed_image is not None:
                    bad_images.append(processed_image.resize(self.image_size))
                    bad_filenames.append(filename)

        # Calculate distances for bad images
        bad_distances = []
        for img in bad_images:
            hv = self.encode_image_simple(img)
            distance = np.sum(hv != prototype)
            bad_distances.append(distance)

        # Calculate review bands
        good_threshold, bad_threshold = self.calculate_review_bands(
            good_distances, bad_distances
        )

        # Classify all images
        good_results = self.classify_with_bands(
            good_distances, good_filenames, good_threshold, bad_threshold
        )
        bad_results = self.classify_with_bands(
            bad_distances, bad_filenames, good_threshold, bad_threshold
        )

        # Generate report
        self.generate_original_report(
            good_results, bad_results, good_threshold, bad_threshold
        )

        # Visualize
        self.visualize_original_results(
            good_results, bad_results, good_threshold, bad_threshold
        )

        self.log_time(total_start, "*** ORIGINAL ANALYSIS COMPLETE ***")

    def generate_original_report(
        self, good_results, bad_results, good_threshold, bad_threshold
    ):
        """Generate report for original analysis."""

        # Count classifications
        good_assume_good = sum(
            1 for r in good_results if r["classification"] == "ASSUME_GOOD"
        )
        good_manual_review = sum(
            1 for r in good_results if r["classification"] == "MANUAL_REVIEW"
        )
        good_assume_bad = sum(
            1 for r in good_results if r["classification"] == "ASSUME_BAD"
        )

        bad_assume_good = sum(
            1 for r in bad_results if r["classification"] == "ASSUME_GOOD"
        )
        bad_manual_review = sum(
            1 for r in bad_results if r["classification"] == "MANUAL_REVIEW"
        )
        bad_assume_bad = sum(
            1 for r in bad_results if r["classification"] == "ASSUME_BAD"
        )

        total_good = len(good_results)
        total_bad = len(bad_results)

        logging.info("=== ORIGINAL SIMPLE HDC CLASSIFICATION REPORT ===")
        logging.info(f"Review Band: {good_threshold:.1f} to {bad_threshold:.1f}")
        logging.info("")
        logging.info("GOOD IMAGES CLASSIFICATION:")
        logging.info(
            f"  Assume Good: {good_assume_good}/{total_good} "
            f"({good_assume_good/total_good*100:.1f}%)"
        )
        logging.info(
            f"  Manual Review: {good_manual_review}/{total_good} "
            f"({good_manual_review/total_good*100:.1f}%)"
        )
        logging.info(
            f"  Assume Bad: {good_assume_bad}/{total_good} "
            f"({good_assume_bad/total_good*100:.1f}%)"
        )
        logging.info("")
        logging.info("BAD IMAGES CLASSIFICATION:")
        logging.info(
            f"  Assume Good: {bad_assume_good}/{total_bad} "
            f"({bad_assume_good/total_bad*100:.1f}%) ⚠️ FALSE NEGATIVES"
        )
        logging.info(
            f"  Manual Review: {bad_manual_review}/{total_bad} "
            f"({bad_manual_review/total_bad*100:.1f}%)"
        )
        logging.info(
            f"  Assume Bad: {bad_assume_bad}/{total_bad} "
            f"({bad_assume_bad/total_bad*100:.1f}%)"
        )
        logging.info("")
        logging.info("PERFORMANCE METRICS:")
        logging.info(f"  False Negative Rate: {bad_assume_good/total_bad*100:.1f}%")
        logging.info(f"  False Positive Rate: {good_assume_bad/total_good*100:.1f}%")
        total_manual = good_manual_review + bad_manual_review
        total_images = total_good + total_bad
        logging.info(f"  Manual Review Rate: {total_manual/total_images*100:.1f}%")

        # Save results
        all_results = []
        for r in good_results:
            r["true_label"] = "GOOD"
            all_results.append(r)
        for r in bad_results:
            r["true_label"] = "BAD"
            all_results.append(r)

        df = pd.DataFrame(all_results)
        df.to_csv("original_simple_hdc_results.csv", index=False)
        logging.info("Results saved to original_simple_hdc_results.csv")

    def visualize_original_results(
        self, good_results, bad_results, good_threshold, bad_threshold
    ):
        """Visualize original results."""
        plt.figure(figsize=(16, 10))

        # Plot good images
        for i, result in enumerate(good_results):
            plt.scatter(
                result["distance"],
                i,
                color=result["color"],
                s=100,
                alpha=0.7,
                edgecolors="blue",
                linewidth=2,
            )
            plt.annotate(
                result["filename"].replace(".jpg", ""),
                (result["distance"], i),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
            )

        # Plot bad images
        for i, result in enumerate(bad_results):
            plt.scatter(
                result["distance"],
                len(good_results) + i,
                color=result["color"],
                s=100,
                alpha=0.7,
                edgecolors="darkorange",
                linewidth=2,
            )
            plt.annotate(
                result["filename"].replace(".jpg", ""),
                (result["distance"], len(good_results) + i),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7),
            )

        # Add threshold lines
        plt.axvline(
            x=good_threshold,
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Assume Good Threshold ({good_threshold:.1f})",
        )
        plt.axvline(
            x=bad_threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Assume Bad Threshold ({bad_threshold:.1f})",
        )

        # Shade review band
        plt.axvspan(
            good_threshold,
            bad_threshold,
            alpha=0.2,
            color="orange",
            label="Manual Review Band",
        )

        plt.xlabel("Original Simple HDC Distance")
        plt.ylabel("Image Index")
        plt.title("Original Simple HDC Classification - Baseline")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    good_folder = "C:/Users/cdbrown/Scripts/Bevel Roughness/Bevel Damage/Good"
    bad_folder = "C:/Users/cdbrown/Scripts/Bevel Roughness/Bevel Damage/Bad"

    analyzer = OriginalSimpleHDC()
    analyzer.run_original_analysis(good_folder, bad_folder)
