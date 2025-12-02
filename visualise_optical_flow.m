%% ============================================================
%   Compute RAFT Optical Flow Between Two Frames of a Video
%   Taken from https://in.mathworks.com/matlabcentral/fileexchange/175668-opticalflow-visualization
% ============================================================

%% ----------- USER INPUTS -----------------
videoPath = "test_videos/basicflow/quick_zoom1_result_ours.mp4";
frameA = 20;     % First frame number
frameB = 21;     % Second frame number (must be > frameA)
decim = [10 10]; % Arrow spacing for vector plot
scale = 0.5;     % Arrow scale factor
%% -----------------------------------------------------------

%% Load video
vid = VideoReader(videoPath);

numFrames = floor(vid.FrameRate * vid.Duration);
assert(frameA >= 1 && frameA <= numFrames, "frameA out of range.");
assert(frameB >= 1 && frameB <= numFrames, "frameB out of range.");
assert(frameB > frameA, "frameB must be > frameA.");

%% Read frame A
vid.CurrentTime = (frameA-1) / vid.FrameRate;
frame1 = readFrame(vid);

%% Read frame B
vid.CurrentTime = (frameB-1) / vid.FrameRate;
frame2 = readFrame(vid);

%% Ensure RGB
if size(frame1,3)==1, frame1 = repmat(frame1,1,1,3); end
if size(frame2,3)==1, frame2 = repmat(frame2,1,1,3); end

%% RAFT optical flow
flowModel = opticalFlowRAFT;

estimateFlow(flowModel, frame1);   % initialize RAFT
flow = estimateFlow(flowModel, frame2);

%% ---- Visualization: Flow Vectors Overlay on Frame A ----
figure;
imshow(frame1); hold on;
plot(flow, DecimationFactor=decim, ScaleFactor=scale, color="g");
title(sprintf("RAFT Optical Flow: Frame %d → %d", frameA, frameB));
hold off;

%% ---- Visualization: flow2rgb ----
flowImage = flow2rgb(flow);

figure;
imshow(flowImage);
title("flow2rgb Visualization");


%% ============================================================
%                 flow2rgb FUNCTION
% ============================================================
function flowImage = flow2rgb(flow)
arguments
    flow (1,1) opticalFlow
end

magnitude = flow.Magnitude;
angle = flow.Orientation;

% Normalize angle to [0,1]
angle_normalized = (angle + pi) ./ (2*pi);

% Normalize magnitude
maxMag = max(magnitude(:));
if maxMag == 0
    magnitude_normalized = zeros(size(magnitude));
else
    magnitude_normalized = magnitude ./ maxMag;
end

% Get color wheel
mapColor = make_colorwheel();
numColors = size(mapColor,1);

x = linspace(0,1,numColors)';
rgbImage = interp1(x,mapColor,angle_normalized,"linear");

% Combine magnitude × color
flowImage = 1 - magnitude_normalized .* (1 - rgbImage);
flowImage = im2single(flowImage);

end


%% ============================================================
%               make_colorwheel FUNCTION
% ============================================================
function colorwheel = make_colorwheel()

RY = 15; YG = 6; GC = 4; CB = 11; BM = 13; MR = 6;
ncols = RY + YG + GC + CB + BM + MR;
colorwheel = zeros(ncols, 3);
col = 1;

% RY
colorwheel(1:RY,1) = 255;
colorwheel(1:RY,2) = floor(255*(0:RY-1)/RY);
col = col+RY;

% YG
colorwheel(col:col+YG-1,1) = 255 - floor(255*(0:YG-1)/YG);
colorwheel(col:col+YG-1,2) = 255;
col = col+YG;

% GC
colorwheel(col:col+GC-1,2) = 255;
colorwheel(col:col+GC-1,3) = floor(255*(0:GC-1)/GC);
col = col+GC;

% CB
colorwheel(col:col+CB-1,2) = 255 - floor(255*(0:CB-1)/CB);
colorwheel(col:col+CB-1,3) = 255;
col = col+CB;

% BM
colorwheel(col:col+BM-1,3) = 255;
colorwheel(col:col+BM-1,1) = floor(255*(0:BM-1)/BM);
col = col+BM;

% MR
colorwheel(col:col+MR-1,3) = 255 - floor(255*(0:MR-1)/MR);
colorwheel(col:col+MR-1,1) = 255;

colorwheel = colorwheel ./ 255;

end
