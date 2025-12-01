import os
import subprocess
from pathlib import Path
import re

def normalize_base_name(filename):
    """Extract and normalize base name from various filename patterns."""
    # Remove file extensions
    name = Path(filename).stem
    
    # Remove common suffixes
    name = re.sub(r'_input$', '', name)
    name = re.sub(r'_result(_ours)?$', '', name)
    name = re.sub(r'_result_outs$', '', name)  # Handle typo in limitation1
    
    # Normalize numbered variations (e.g., quick_zoom1 -> quick_zoom_input1)
    # This handles cases where input has "input1" but others have just "1"
    name = re.sub(r'(\d+)$', r'_input\1', name)
    name = re.sub(r'_input_input', '_input', name)  # Fix double input
    
    return name

def find_matching_videos(base_dir='./test_videos'):
    """Find all matching video sets by intelligently matching base names."""
    inputs_dir = Path(base_dir) / 'inputs'
    basicflow_dir = Path(base_dir) / 'basicflow'
    steadyflow_dir = Path(base_dir) / 'steadyflow'
    
    # Get all input videos
    input_videos = sorted(list(inputs_dir.glob('*.avi')) + list(inputs_dir.glob('*.mp4')))
    
    video_sets = []
    
    for input_video in input_videos:
        # Extract base name from input
        input_stem = input_video.stem
        
        # Create possible base name patterns to search for
        patterns = []
        
        if '_input' in input_stem:
            # Remove _input suffix
            base = input_stem.replace('_input', '')
            patterns.append(base)
            
            # Handle numbered cases: quick_zoom_input1 -> quick_zoom1
            if base.endswith(('1', '2', '3', '4', '5', '6', '7', '8', '9')):
                # For "large_foreground_input1" -> "large_foreground1"
                patterns.append(base)
            
            # Also try with number moved: quick_zoom_input1 -> quick_zoom1
            match = re.match(r'(.+)_input(\d+)$', input_stem)
            if match:
                patterns.append(f"{match.group(1)}{match.group(2)}")
        else:
            # No _input suffix (like limitation1, limitation2)
            base = input_stem
            patterns.append(base)
        
        # Search for matching files in basicflow and steadyflow
        basicflow_match = None
        steadyflow_match = None
        
        for pattern in patterns:
            if not basicflow_match:
                # Try various basicflow naming patterns
                candidates = (
                    list(basicflow_dir.glob(f'{pattern}_result_ours.mp4')) +
                    list(basicflow_dir.glob(f'{pattern}_result_outs.mp4'))  # Handle typo
                )
                if candidates:
                    basicflow_match = candidates[0]
            
            if not steadyflow_match:
                # Try various steadyflow naming patterns
                candidates = (
                    list(steadyflow_dir.glob(f'{pattern}_result.avi')) +
                    list(steadyflow_dir.glob(f'{input_stem}_result.avi'))  # Try with original input name
                )
                if candidates:
                    steadyflow_match = candidates[0]
        
        if basicflow_match and steadyflow_match:
            # Use the original input stem (without _input) as display name
            display_name = input_stem.replace('_input', '')
            
            video_sets.append({
                'base_name': display_name,
                'input': str(input_video),
                'basicflow': str(basicflow_match),
                'steadyflow': str(steadyflow_match)
            })
            print(f"✓ Matched: {display_name}")
            print(f"    Input:      {input_video.name}")
            print(f"    BasicFlow:  {basicflow_match.name}")
            print(f"    SteadyFlow: {steadyflow_match.name}")
        else:
            print(f"✗ No match for: {input_stem}")
            if not basicflow_match:
                print(f"    Missing in basicflow/ (searched: {patterns})")
            if not steadyflow_match:
                print(f"    Missing in steadyflow/ (searched: {patterns})")
    
    print()
    return video_sets

def create_comparison_video(video_set, output_dir):
    """Create a comparison video with title and three videos side by side."""
    base_name = video_set['base_name']
    output_path = Path(output_dir) / f'{base_name}_comparison.mp4'
    
    # FFmpeg filter complex for:
    # 1. Scale all videos to same height (480p for efficiency)
    # 2. Add labels below each video
    # 3. Stack horizontally
    # 4. Add title at top
    
    filter_complex = (
        # Scale all inputs to same height with even width (required for H.264)
        # scale=-2:480 ensures width is divisible by 2
        "[0:v]scale=-2:480,drawtext=text='Input':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h-40:box=1:boxcolor=black@0.5:boxborderw=5[v0]; "
        "[1:v]scale=-2:480,drawtext=text='BasicFlow':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h-40:box=1:boxcolor=black@0.5:boxborderw=5[v1]; "
        "[2:v]scale=-2:480,drawtext=text='SteadyFlow':fontsize=24:fontcolor=white:x=(w-text_w)/2:y=h-40:box=1:boxcolor=black@0.5:boxborderw=5[v2]; "
        # Stack three videos horizontally
        f"[v0][v1][v2]hstack=inputs=3[stacked]; "
        # Add title bar at top with the base name
        f"[stacked]drawtext=text='{base_name}':fontsize=32:fontcolor=white:x=(w-text_w)/2:y=20:box=1:boxcolor=black@0.7:boxborderw=10[out]"
    )
    
    command = [
        'ffmpeg',
        '-i', video_set['input'],
        '-i', video_set['basicflow'],
        '-i', video_set['steadyflow'],
        '-filter_complex', filter_complex,
        '-map', '[out]',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-y',  # Overwrite output file if exists
        str(output_path)
    ]
    
    print(f"Creating comparison video for: {base_name}")
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"✓ Successfully created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating {base_name}: {e.stderr.decode()}")

def main():
    base_dir = './test_videos'
    output_dir = Path(base_dir) / 'comparison'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video sets
    print("Scanning for matching video sets...\n")
    video_sets = find_matching_videos(base_dir)
    
    if not video_sets:
        print("No matching video sets found!")
        return
    
    print(f"Found {len(video_sets)} video sets to process\n")
    print("="*60)
    
    # Process each video set
    for i, video_set in enumerate(video_sets, 1):
        print(f"\n[{i}/{len(video_sets)}] Processing: {video_set['base_name']}")
        create_comparison_video(video_set, output_dir)
    
    print("\n" + "="*60)
    print(f"All done! Comparison videos saved to: {output_dir}")

if __name__ == '__main__':
    main()