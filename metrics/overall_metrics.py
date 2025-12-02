import cv2
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
        orb = cv2.ORB_create(nfeatures=5000)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None, None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches)*0.5)]

        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

        return H, M

    def calculate_metrics(self):
        cap_orig = cv2.VideoCapture(self.input_path)
        cap_stab = cv2.VideoCapture(self.stab_path)

        crops = []
        distortions = []
        
        path_x, path_y, path_a = [0.0], [0.0], [0.0]
        
        prev_stab_gray = None
        
        frame_count = 0
        while True:
            ret1, frame_orig = cap_orig.read()
            ret2, frame_stab = cap_stab.read()

            if not ret1 or not ret2:
                break
            
            frame_count += 1
            
            h_orig, w_orig = frame_orig.shape[:2]
            h_stab, w_stab = frame_stab.shape[:2]
            
            h = min(h_orig, h_stab)
            w = min(w_orig, w_stab)
            
            img_orig_small = cv2.resize(frame_orig, (w, h))
            img_stab_small = cv2.resize(frame_stab, (w, h))

            gray_orig = cv2.cvtColor(img_orig_small, cv2.COLOR_BGR2GRAY)
            gray_stab = cv2.cvtColor(img_stab_small, cv2.COLOR_BGR2GRAY)

            H_io, _ = self.get_homography_and_motion(gray_orig, gray_stab)
            
            if H_io is not None:
                try:
                    A = H_io[:2, :2]
                    _, s, _ = np.linalg.svd(A)
                    if s[0] > 1e-6:
                        distortions.append(s[1] / s[0])
                except np.linalg.LinAlgError:
                    pass

                try:
                    H_inv = np.linalg.inv(H_io)
                    
                    stab_corners = np.array([
                        [0, 0], 
                        [w, 0], 
                        [w, h], 
                        [0, h]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    
                    orig_projected_corners = cv2.perspectiveTransform(stab_corners, H_inv)
                    
                    visible_area = cv2.contourArea(orig_projected_corners)
                    total_area = w * h
                    
                    crops.append(visible_area / total_area)
                except np.linalg.LinAlgError:
                    pass

            if prev_stab_gray is not None:
                _, M_stab = self.get_homography_and_motion(prev_stab_gray, gray_stab)
                
                dx, dy, da = 0, 0, 0
                if M_stab is not None:
                    dx = M_stab[0, 2]
                    dy = M_stab[1, 2]
                    da = np.arctan2(M_stab[1, 0], M_stab[0, 0])
                
                path_x.append(path_x[-1] + dx)
                path_y.append(path_y[-1] + dy)
                path_a.append(path_a[-1] + da)

            prev_stab_gray = gray_stab

        cap_orig.release()
        cap_stab.release()

        def get_freq_ratio(trajectory):
            signal = np.array(trajectory)
            if len(signal) < 10: return 1.0
            
            fft_magnitude = np.abs(rfft(signal))
            energy = fft_magnitude ** 2
            
            low_freq_energy = np.sum(energy[1:7]) 
            total_energy = np.sum(energy[1:]) + 1e-10
            
            return low_freq_energy / total_energy

        s_x = get_freq_ratio(path_x)
        s_y = get_freq_ratio(path_y)
        s_a = get_freq_ratio(path_a)
        final_stability = min(s_x, s_y, s_a)

        final_distortion = np.min(distortions) if distortions else 0.0
        final_cropping = np.mean(crops) if crops else 0.0

        return {
            "Stability Score": final_stability,
            "Distortion Score": final_distortion,
            "Cropping Ratio": final_cropping
        }


def get_first_frame(video_path):
    """Extract and return the first frame of a video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def find_video_sets(base_dir='./test_videos'):
    """Find all matching video sets across the three folders."""
    inputs_dir = Path(base_dir) / 'inputs'
    basicflow_dir = Path(base_dir) / 'basicflow'
    steadyflow_dir = Path(base_dir) / 'steadyflow'
    
    input_videos = sorted(list(inputs_dir.glob('*.avi')) + list(inputs_dir.glob('*.mp4')))
    
    video_sets = []
    
    for input_video in input_videos:
        input_stem = input_video.stem
        
        # Skip static_easy if it has no matches
        if input_stem == 'static_easy':
            continue
        
        patterns = []
        
        if '_input' in input_stem:
            base = input_stem.replace('_input', '')
            patterns.append(base)
            
            import re
            match = re.match(r'(.+)_input(\d+)$', input_stem)
            if match:
                patterns.append(f"{match.group(1)}{match.group(2)}")
        else:
            base = input_stem
            patterns.append(base)
        
        basicflow_match = None
        steadyflow_match = None
        
        for pattern in patterns:
            if not basicflow_match:
                candidates = (
                    list(basicflow_dir.glob(f'{pattern}_result_ours.mp4')) +
                    list(basicflow_dir.glob(f'{pattern}_result_outs.mp4'))
                )
                if candidates:
                    basicflow_match = candidates[0]
            
            if not steadyflow_match:
                candidates = (
                    list(steadyflow_dir.glob(f'{pattern}_result.avi')) +
                    list(steadyflow_dir.glob(f'{input_stem}_result.avi'))
                )
                if candidates:
                    steadyflow_match = candidates[0]
        
        if basicflow_match and steadyflow_match:
            display_name = input_stem.replace('_input', '')
            
            video_sets.append({
                'base_name': display_name,
                'input': str(input_video),
                'basicflow': str(basicflow_match),
                'steadyflow': str(steadyflow_match)
            })
    
    return sorted(video_sets, key=lambda x: x['base_name'])


def create_metrics_table(base_dir='./test_videos', output_path='metrics_comparison.png'):
    """Create a comprehensive metrics comparison table."""
    
    video_sets = find_video_sets(base_dir)
    
    if not video_sets:
        print("No matching video sets found!")
        return
    
    n_videos = len(video_sets)
    print(f"Found {n_videos} video sets to process\n")
    
    # Calculate metrics for all videos
    all_metrics = []
    first_frames = []
    
    for i, video_set in enumerate(video_sets, 1):
        print(f"[{i}/{n_videos}] Processing: {video_set['base_name']}")
        
        # Get first frame
        frame = get_first_frame(video_set['input'])
        first_frames.append(frame)
        
        # Calculate metrics for BasicFlow
        print(f"  Calculating BasicFlow metrics...")
        bf_metrics = VideoMetrics(video_set['input'], video_set['basicflow'])
        bf_results = bf_metrics.calculate_metrics()
        
        # Calculate metrics for SteadyFlow
        print(f"  Calculating SteadyFlow metrics...")
        sf_metrics = VideoMetrics(video_set['input'], video_set['steadyflow'])
        sf_results = sf_metrics.calculate_metrics()
        
        all_metrics.append({
            'name': video_set['base_name'],
            'bf': bf_results,
            'sf': sf_results
        })
        print()
    
    # Create the figure
    fig = plt.figure(figsize=(max(20, n_videos * 2.5), 12))
    
    # Create gridspec: 1 row for frames + 3 rows for metrics
    gs = fig.add_gridspec(4, n_videos, height_ratios=[2, 1, 1, 1], 
                          hspace=0.05, wspace=0.05,
                          left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # Add title
    fig.suptitle('Video Stabilization Metrics Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Row 0: First frames
    for col, (video_set, frame) in enumerate(zip(video_sets, first_frames)):
        ax = fig.add_subplot(gs[0, col])
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(video_set['base_name'], fontsize=10, fontweight='bold', pad=5)
        ax.axis('off')
    
    # Metric rows
    metric_names = ['Stability Score', 'Distortion Score', 'Cropping Ratio']
    metric_colors = {
        'bf': '#3498db',  # Blue
        'sf': '#e74c3c'   # Red
    }
    
    for row_idx, metric_name in enumerate(metric_names, start=1):
        for col, metrics_data in enumerate(all_metrics):
            ax = fig.add_subplot(gs[row_idx, col])
            
            # Get metric values
            bf_val = metrics_data['bf'][metric_name]
            sf_val = metrics_data['sf'][metric_name]
            
            # Create bar chart
            x_pos = [0, 1]
            values = [bf_val, sf_val]
            colors = [metric_colors['bf'], metric_colors['sf']]
            
            bars = ax.bar(x_pos, values, color=colors, width=0.7, alpha=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Styling
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(['BF', 'SF'], fontsize=9)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            
            # Only show y-axis label on leftmost column
            if col == 0:
                ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
                ax.tick_params(axis='y', labelsize=9)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(axis='x', labelsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
    
    # Add legend
    bf_patch = mpatches.Patch(color=metric_colors['bf'], label='BasicFlow', alpha=0.8)
    sf_patch = mpatches.Patch(color=metric_colors['sf'], label='SteadyFlow', alpha=0.8)
    
    fig.legend(handles=[bf_patch, sf_patch], 
              loc='upper right', fontsize=12, 
              bbox_to_anchor=(0.98, 0.96))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Metrics comparison table saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_metrics_table(base_dir='./test_videos', 
                        output_path='metrics_comparison.png')