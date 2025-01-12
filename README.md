# Image Depth Estimation

This repository explores fundamental techniques for estimating depth from stereo image pairs. It focuses on extracting disparity information from two grayscale images to generate accurate depth maps. The project implements both pixel-wise and window-based matching algorithms, utilizing different distance metrics and similarity measures.

## Implemented Methods

This project includes the following depth estimation methods:

### 1. Pixel-Wise Matching

This method computes the disparity map by comparing individual pixel intensities between the left and right images.  It calculates the cost for each possible disparity by measuring the pixel-wise distance between corresponding pixels, and then chooses the disparity with the minimum cost.

- **Distance Metrics:**
    - **L1 Distance (Manhattan Distance):**  Calculates the absolute difference between pixel intensities.
    - **L2 Distance (Euclidean Distance):** Calculates the squared difference between pixel intensities.

### 2. Window-Based Matching

This approach calculates disparities by comparing small image patches (windows) centered around each pixel. It improves the robustness to noise by considering a neighborhood of pixels instead of only individual pixels.

- **Distance Metrics:**
  -   **L1 Distance (Manhattan Distance):**  Calculates the absolute difference between pixels in the window.
  -   **L2 Distance (Euclidean Distance):** Calculates the squared difference between pixels in the window.
- **Similarity Measure:**
    -   **Cosine Similarity:** Measures the cosine of the angle between the vectors formed by pixel values within the corresponding windows.

## Files

The project consists of the following Python files:

- **`README.md`**: This file, providing an overview of the project.
- **`pixel_wise_matching.py`**: Implements pixel-wise matching with L1 and L2 distance metrics.
- **`window_based_matching.py`**: Implements window-based matching with L1 and L2 distance metrics.
- **`window_based_cosine_similarity.py`**: Implements window-based matching using cosine similarity as a matching metric.

## Usage

To use this project:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd image_depth_estimation
    ```

2.  **Ensure you have the required libraries:**
    -   `opencv-python`
    -   `numpy`
    Install these libraries using pip:

    ```bash
    pip install opencv-python numpy
    ```

3.  **Run the Python scripts:**

    - To run the pixel-wise matching methods:

      ```bash
      python pixel_wise_matching.py
      ```

    - To run the window-based matching methods with L1 and L2 distance:

        ```bash
        python window_based_matching.py
        ```

    - To run the window-based matching method with cosine similarity:

        ```bash
        python window_based_cosine_similarity.py
        ```

4. **View results:**

    The generated disparity maps (both grayscale and colorized) will be saved in the respective `result` subdirectories:

    - `result/pixel_wise/` for pixel-wise matching results.
    - `result/window_based/` for window-based matching results.

## Datasets

The project uses the following image pairs:

-   **Tsukuba:** Used for pixel-wise matching, located in `assets/tsukuba/`.
-   **Aloe:** Used for window-based matching, located in `assets/Aloe/`.

## Results

The scripts will output the resulting disparity maps as grayscale images and colorized depth maps using the `COLORMAP_JET` color map.

## Notes
-  The `scale` variable controls the scaling of the calculated disparity before converting it to an 8-bit grayscale image.
-   The kernel size is a crucial parameter that influences the performance of window-based matching.
-   The disparity range should be adjusted based on the specific image pair being used.

## Further Improvements

-   Implement additional matching algorithms (e.g. graph cuts, semi-global matching).
-   Experiment with different window sizes and shapes for window-based methods.
-   Apply post-processing techniques (e.g. median filtering) to reduce noise and improve the visual quality of depth maps.
-   Implement a more robust cost aggregation method.

This project serves as a basic foundation for exploring and understanding stereo depth estimation. It can be extended to more sophisticated methods and used as a base for creating various applications that require depth information.
