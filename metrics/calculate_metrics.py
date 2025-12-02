import cv2
import numpy as np
from numpy.fft import rfft
import argparse

class VideoMetrics:
    def __init__(self, input_path, stab_path):
        self.input_path = input_path
        self.stab_path = stab_path

    def get_homography_and_motion(self, img1, img2):
        """
        Returns:
            H: Homography matrix (3x3) mapping img1 -> img2
            M: Affine matrix (2x3) mapping img1 -> img2 (for trajectory)
        """
        # ORB is faster than SURF and unpatented. Increase features for robustness.
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Keep top 50% matches to remove outliers
        matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches)*0.5)]

        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 1. Homography (Exact warp for Cropping/Distortion)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 2. Affine Partial (Rotation + Translation for Stability Trajectory)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        return H, M

    def calculate_metrics(self):
        cap_orig = cv2.VideoCapture(self.input_path)
        cap_stab = cv2.VideoCapture(self.stab_path)

        crops = []
        distortions = []
        
        # Trajectory accumulators (start at zero)
        path_x, path_y, path_a = [0.0], [0.0], [0.0]
        
        prev_stab_gray = None

        print("Processing video frames...")
        
        frame_count = 0
        while True:
            ret1, frame_orig = cap_orig.read()
            ret2, frame_stab = cap_stab.read()

            if not ret1 or not ret2:
                break
            
            frame_count += 1
            
            # Ensure dimensions match for feature matching stability
            h_orig, w_orig = frame_orig.shape[:2]
            h_stab, w_stab = frame_stab.shape[:2]
            
            # Use the smaller dimension for matching resize
            h = min(h_orig, h_stab)
            w = min(w_orig, w_stab)
            
            img_orig_small = cv2.resize(frame_orig, (w, h))
            img_stab_small = cv2.resize(frame_stab, (w, h))

            gray_orig = cv2.cvtColor(img_orig_small, cv2.COLOR_BGR2GRAY)
            gray_stab = cv2.cvtColor(img_stab_small, cv2.COLOR_BGR2GRAY)

            # --- METRICS 1 & 2: Cropping & Distortion (Input vs Output) ---
            H_io, _ = self.get_homography_and_motion(gray_orig, gray_stab)
            
            if H_io is not None:
                # -- Distortion Score --
                # "Ratio of two largest eigenvalues of the affine part... choose their minimum"
                # Ideally 1.0 (Isotropic scaling).
                try:
                    # Top-left 2x2 is the affine part (Rotation + Scale + Shear)
                    A = H_io[:2, :2]
                    # SVD singular values (s1 >= s2)
                    _, s, _ = np.linalg.svd(A)
                    if s[0] > 1e-6:
                        # Ratio min/max. If perfectly isotropic, s1=s2 -> ratio=1.
                        distortions.append(s[1] / s[0])
                except np.linalg.LinAlgError:
                    pass

                # -- Cropping Ratio --
                # "Ratio of overlapping area and original frame area"
                # We project the Stabilized Frame corners BACK to the Original Domain
                try:
                    H_inv = np.linalg.inv(H_io)
                    
                    # Define corners of the stabilized frame (in small resized coords)
                    stab_corners = np.array([
                        [0, 0], 
                        [w, 0], 
                        [w, h], 
                        [0, h]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    
                    # Project corners back to Original Image coordinate space
                    orig_projected_corners = cv2.perspectiveTransform(stab_corners, H_inv)
                    
                    # Calculate area of this projected polygon
                    visible_area = cv2.contourArea(orig_projected_corners)
                    total_area = w * h
                    
                    crops.append(visible_area / total_area)
                except np.linalg.LinAlgError:
                    pass

            # --- METRIC 3: Stability Score (Output Trajectory Analysis) ---
            # We track the path of the STABILIZED video itself (Frame t -> Frame t+1)
            if prev_stab_gray is not None:
                _, M_stab = self.get_homography_and_motion(prev_stab_gray, gray_stab)
                
                dx, dy, da = 0, 0, 0
                if M_stab is not None:
                    dx = M_stab[0, 2]
                    dy = M_stab[1, 2]
                    da = np.arctan2(M_stab[1, 0], M_stab[0, 0])
                
                # Accumulate motion to get absolute path
                path_x.append(path_x[-1] + dx)
                path_y.append(path_y[-1] + dy)
                path_a.append(path_a[-1] + da)

            prev_stab_gray = gray_stab

        cap_orig.release()
        cap_stab.release()

        # --- Aggregation ---
        def get_freq_ratio(trajectory):
            signal = np.array(trajectory)
            if len(signal) < 10: return 1.0
            
            # FFT
            fft_magnitude = np.abs(rfft(signal))
            energy = fft_magnitude ** 2
            
            # Indices: 0=DC, 1=Fundamental (Motion), 2-6=Low Freqs
            # Paper: "2nd to 6th frequencies" (1-based index) -> Python indices 1 to 6
            # This captures smooth panning (idx 1) + gentle wobble (idx 2-6) as "good" energy.
            low_freq_energy = np.sum(energy[1:7]) 
            
            # Total energy excluding DC
            total_energy = np.sum(energy[1:]) + 1e-10
            
            return low_freq_energy / total_energy

        # 1. Stability: Min of Translation(x,y) and Rotation scores
        s_x = get_freq_ratio(path_x)
        s_y = get_freq_ratio(path_y)
        s_a = get_freq_ratio(path_a)
        final_stability = min(s_x, s_y, s_a)

        # 2. Distortion: Minimum (Worst case) frame score
        final_distortion = np.min(distortions) if distortions else 0.0

        # 3. Cropping: Average frame score
        final_cropping = np.mean(crops) if crops else 0.0

        return {
            "Stability Score": final_stability,
            "Distortion Score": final_distortion,
            "Cropping Ratio": final_cropping
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original shaky video")
    parser.add_argument("--stab", required=True, help="Path to stabilized output video")
    args = parser.parse_args()
    
    metrics = VideoMetrics(args.input, args.stab)
    results = metrics.calculate_metrics()
    
    print("\n=== Video Stabilization Evaluation ===")
    print(f"Stability Score:  {results['Stability Score']:.4f}")
    print(f"Distortion Score: {results['Distortion Score']:.4f}")
    print(f"Cropping Ratio:   {results['Cropping Ratio']:.4f}")
    print("======================================\n")