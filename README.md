# BasicFlow

Course repository for CS7.404 Digital Image Processing.

## Overview

BasicFlow is an alternative, computationally light implementation of the pipeline for SteadyFlow based on "SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization" (CVPR 2014) by Liu et al.

SteadyFlow stabilizes videos by enforcing spatial smoothness on optical flow, removing discontinuous motion, and smoothing per-pixel motion profiles instead of feature trajectories.


### 1. The SteadyFlow Method

**SteadyFlow** is a video stabilization technique designed to handle spatially variant motion (e.g., parallax, rolling shutter) without relying on brittle feature tracking or simple 2D homography models.

* **Core Concept (Pixel Profiles):** Instead of tracking specific features (like corners) which can be lost, SteadyFlow analyzes "Pixel Profiles". A pixel profile is the collection of motion vectors at a fixed pixel location $(x, y)$ over time.
* **Motion Model:** It uses a dense optical flow field. However, raw optical flow is often noisy or discontinuous (e.g., at object boundaries).
* **Spatially Smooth Flow:** The method modifies raw optical flow into "SteadyFlow" by identifying discontinuous motion vectors (using spatial-temporal analysis) and "inpainting" (completing) the motion from neighbors to enforce strong spatial coherence.
* **Smoothing:** Once the flow is spatially smooth, the video is stabilized by smoothing each pixel profile independently using an optimization function.

### 2. MeshFlow Improvements upon SteadyFlow

**MeshFlow** improves upon SteadyFlow primarily by addressing computational efficiency and latency, making online stabilization possible.

* **Sparse vs. Dense:** SteadyFlow calculates dense optical flow for every pixel, which is computationally expensive. MeshFlow replaces this with a **sparse motion model** defined only at mesh vertexes. It tracks sparse features (FAST/KLT) and propagates their motion to nearby mesh vertexes.
* **Median Filtering:** To maintain the spatial smoothness required for stabilization (which SteadyFlow achieved via inpainting), MeshFlow uses two median filters on the vertex motions to reject outliers and smooth the field.
* **Predicted Adaptive Path Smoothing (PAPS):** SteadyFlow requires future frames and iterative optimization to determine the smoothing strength ($\lambda_t$). MeshFlow introduces PAPS, which predicts the optimal smoothing strength based only on past motion (Translation and Affine components), allowing for online processing with only one frame of latency.

### 3. Analysis of BasicFlow and Modifications from Papers

The provided code, BasicFlow, implements a hybrid approach. It constructs a **SteadyFlow** pipeline but significantly simplifies the complex iterative components for performance, relying on the superior accuracy of modern Deep Learning optical flow (RAFT) to compensate for the simpler logic.



#### A. Simplified Discontinuity Analysis (Spatial vs. Temporal)
* **Original Paper:** The SteadyFlow paper uses a complex "Iterative Refinement" loop. It first identifies discontinuities using **spatial analysis** (gradient thresholds), performs motion completion, stabilizes the video, and *then* uses **temporal analysis** (checking if accumulated motion is smooth over time) to refine the outlier mask and repeat the process.
* **BasicFlow Implementation:** The code implements a **single-pass spatial analysis only**.
    * It calculates the gradient magnitude of the flow field using Sobel operators (`dx`, `dy`).
    * It identifies discontinuities purely based on a spatial threshold (`grad_mag_sum > 2.0`).
    * **Why/Result:** It completely skips the temporal analysis and the outer refinement loop. This avoids the problem where temporal analysis requires an already-stable video to work well. By trusting the single-pass spatial check, the pipeline becomes linear and significantly faster.

#### B. Removal of Complex Motion Completion
* **Original Paper:** To fix discontinuities (holes in the flow), the paper uses "As-Similar-As-Possible" (ASAP) warping. This involves setting up a grid of control points and solving a sparse linear system to minimize an energy function that balances data fidelity with grid rigidity. This is mathematically heavy and computationally expensive.
* **BasicFlow Implementation:** The code replaces the linear system solver with **Iterative Diffusion (Blurring)**.
    * It identifies the "bad" pixels (edges/discontinuities) using the spatial mask.
    * It runs a loop (10 iterations) where it applies a simple box blur (`cv2.blur`) to the flow field, but *only updates the pixels inside the mask* with the blurred values.
    * **Why/Result:** This acts as a diffusion process, filling in the discontinuities with smooth values from the valid neighbors. Because RAFT provides a very dense and high-quality flow initially (unlike the methods in 2014), we do not need to reconstruct large missing chunks of data (which ASAP is good for). We only need to smooth sharp edges. This simple blurring is orders of magnitude faster and yields cleaner results for high-quality input flow.



#### C. Exact Deviations from Literature
The implementation is not *exactly* SteadyFlow or MeshFlow. It is a modernized, simplistic variant. The specific differences are:

1.  **No Temporal Outlier Detection:** The code does not check `flow_acc` for temporal smoothness to update the mask as described in Equation 1 of the SteadyFlow paper.
2.  **No Iterative Refinement Loop:** The global process (Estimate -> Inpaint -> Stabilize -> Refine) described in Section 4.4 of the SteadyFlow paper is removed. The code runs strictly sequentially once.
3.  **Different Motion Completion:** The ASAP energy minimization (Equation 2 in SteadyFlow) is replaced by the iterative `cv2.blur` loop.
4.  **Hybrid Smoothing Parameters:** It uses the **SteadyFlow** solver (Jacobi iteration on dense pixels) but drives it with **MeshFlow** parameters. Instead of iteratively searching for $\lambda_t$ (SteadyFlow), it predicts $\lambda_t$ using the MeshFlow linear regression formulas based on Homography translation and affine components.
5.  **RAFT Integration:** The use of `torchvision.models.optical_flow.raft_large` replaces the optical flow and feature tracking methods of both papers, serving as the foundational accuracy boost that allows the other simplifications to work effectively.

### 4. Solving Issues Presented in the Papers

The implementation in BasicFLow addresses several key limitations identified in the source texts:

| Issue in Paper | Context | How BasicFlow Solves It |
| :--- | :--- | :--- |
| **Computational Cost** | SteadyFlow is slow (1.5s/frame) due to ASAP warping and iterative parameter search. | **Solution:** By replacing ASAP warping with **iterative blurring** and parameter search with **MeshFlow prediction**, the per-frame overhead is drastically reduced. |
| **Feature Tracking Brittleness** | Feature tracks (used in MeshFlow) are sparse, uneven, and hard to maintain in textureless regions. | **Solution:** By sticking to the **SteadyFlow dense model** (Pixel Profiles) but powering it with **RAFT**, the code avoids the need for sparse feature tracking entirely. RAFT is highly robust in textureless regions compared to traditional KLT. |
| **Parameter Tuning Latency** | SteadyFlow requires iterative refinement to find the smoothing weight $\lambda_t$, which is "impractical for the online scenario". | **Solution:** The code imports the **PAPS (Predicted Adaptive Path Smoothing)** logic from MeshFlow. It calculates $\lambda_t$ instantly using the global homography, avoiding the costly iterative search while keeping the high-quality dense warp. |


# References
- [OpticalFlow-Visualization, MATLAB optical flow visualization following Baker et al. (ICCV 2007) as used by the MPI-Sintel challenge](https://in.mathworks.com/matlabcentral/fileexchange/175668-opticalflow-visualization)
- [For understanding optical flow visualisation](https://medium.com/@ml6vq/understanding-optical-flow-visualization-293471c97456)
- 
