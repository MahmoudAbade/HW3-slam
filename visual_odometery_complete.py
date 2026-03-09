"""
Complete Visual Odometry Pipeline (RGB-D)
- ORB feature detection with CLAHE preprocessing
- FLANN-based matching with Lowe's ratio test
- PnP pose estimation using depth data (metric scale)
- Real-time frame processing visualization
- Ground truth comparison and ATE calculation
- 3D point cloud map with Pangolin viewer

Students:
- Genan Abdallah, 323033035
- Muhammad Swalha, 314932211
- Mahmoud Abade, 206773756
"""

import cv2
import numpy as np
import os
import math
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
import pypangolin as pangolin
import OpenGL.GL as gl


# ====================================================================================
# DATASET LOADER
# ====================================================================================

class DatasetLoader:
    """Load and synchronize TUM RGB-D dataset (RGB + Depth + Ground Truth)."""
    def __init__(self, data_path):
        self.data_path = data_path
        self.rgb_file = os.path.join(data_path, 'rgb.txt')
        self.depth_file = os.path.join(data_path, 'depth.txt')
        self.gt_file = os.path.join(data_path, 'groundtruth.txt')

    def _parse_file(self, filepath):
        entries = []
        if not os.path.exists(filepath):
            return entries
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    entries.append((float(parts[0]), parts[1:]))
        return entries

    def _find_closest(self, target_ts, entries, max_diff=0.02):
        best_diff = float('inf')
        best = None
        for ts, data in entries:
            diff = abs(target_ts - ts)
            if diff < best_diff:
                best_diff = diff
                best = (ts, data)
        if best_diff <= max_diff:
            return best
        return None

    def _parse_gt_pose(self, data):
        """Convert ground truth line [tx ty tz qx qy qz qw] to 4x4 pose matrix."""
        tx, ty, tz = float(data[0]), float(data[1]), float(data[2])
        qx, qy, qz, qw = float(data[3]), float(data[4]), float(data[5]), float(data[6])
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def get_synchronized_frames(self):
        rgb_entries = self._parse_file(self.rgb_file)
        depth_entries = self._parse_file(self.depth_file)
        gt_entries = self._parse_file(self.gt_file)

        frames = []
        for ts_rgb, rgb_data in rgb_entries:
            # Find matching depth
            depth_match = self._find_closest(ts_rgb, depth_entries)
            if depth_match is None:
                continue

            # Find matching ground truth
            gt_match = self._find_closest(ts_rgb, gt_entries, max_diff=0.05)
            gt_pose = self._parse_gt_pose(gt_match[1]) if gt_match else None

            frames.append({
                'timestamp': ts_rgb,
                'rgb': os.path.join(self.data_path, rgb_data[0]),
                'depth': os.path.join(self.data_path, depth_match[1][0]),
                'gt_pose': gt_pose
            })

        print(f"[DatasetLoader] Synchronized {len(frames)} RGB-D frames "
              f"({sum(1 for f in frames if f['gt_pose'] is not None)} with GT).")
        return frames


# ====================================================================================
# VISUAL ODOMETRY (PnP with Depth)
# ====================================================================================

class VisualOdometry:
    """
    RGB-D Visual Odometry using PnP with depth for metric-scale pose estimation.
    """
    def __init__(self, fx, fy, cx, cy, depth_scale=5000.0, visualize=True):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.visualize = visualize

        # ORB feature detector
        self.orb = cv2.ORB_create(
            nfeatures=3000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=12
        )

        # FLANN matcher for binary descriptors (LSH)
        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Camera intrinsic matrix
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        self.last_keypoints = None
        self.last_descriptors = None
        self.last_image = None
        self.last_depth = None
        self.current_pose = np.eye(4)

        if self.visualize:
            cv2.namedWindow('Visual Odometry - Frame Processing', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Visual Odometry - Frame Processing', 1280, 480)

    def _extract_features(self, image):
        """Extract ORB features with CLAHE preprocessing."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return self.orb.detectAndCompute(gray, None)

    def _robust_matching(self, desc1, desc2):
        """Feature matching with Lowe's ratio test."""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            return []

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        return good_matches

    def _unproject(self, u, v, z):
        """Convert pixel (u,v) + depth z to 3D point in camera frame."""
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z])

    def _estimate_motion_pnp(self, kp1, kp2, matches, depth1):
        """
        Estimate camera motion using PnP with depth from previous frame.
        Returns R, t, num_inliers.
        """
        pts_3d = []
        pts_2d = []

        for m in matches:
            u1, v1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
            u2, v2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])

            # Bounds check
            if v1 >= depth1.shape[0] or u1 >= depth1.shape[1]:
                continue

            d_val = depth1[v1, u1]
            if d_val == 0:
                continue

            z = float(d_val) / self.depth_scale
            if 0.3 < z < 6.0:
                pts_3d.append(self._unproject(u1, v1, z))
                pts_2d.append([u2, v2])

        if len(pts_3d) < 15:
            return None, None, 0

        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, None,
            iterationsCount=300, reprojectionError=2.0, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP
        )

        if success and inliers is not None and len(inliers) >= 15:
            # Refine with iterative method using inliers only
            pts_3d_in = pts_3d[inliers.flatten()]
            pts_2d_in = pts_2d[inliers.flatten()]
            _, rvec, tvec = cv2.solvePnP(
                pts_3d_in, pts_2d_in, self.K, None,
                rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec, len(inliers)

        return None, None, 0

    def process_frame(self, image, depth_image, frame_num=0):
        """Process an RGB-D frame and estimate camera pose."""
        keypoints, descriptors = self._extract_features(image)

        if self.last_keypoints is None or descriptors is None:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            self.last_image = image.copy()
            self.last_depth = depth_image.copy()
            if self.visualize:
                self._visualize_keypoints(image, keypoints, frame_num)
            return self.current_pose.copy()

        if len(descriptors) < 8 or len(self.last_descriptors) < 8:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            self.last_image = image.copy()
            self.last_depth = depth_image.copy()
            return self.current_pose.copy()

        # Match features
        good_matches = self._robust_matching(self.last_descriptors, descriptors)

        if len(good_matches) < 15:
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            self.last_image = image.copy()
            self.last_depth = depth_image.copy()
            return self.current_pose.copy()

        # Estimate motion using PnP with depth
        R, tvec, num_inliers = self._estimate_motion_pnp(
            self.last_keypoints, keypoints, good_matches, self.last_depth
        )

        # Visualize
        if self.visualize:
            self._visualize_matching(self.last_image, image, self.last_keypoints,
                                     keypoints, good_matches, R, tvec, num_inliers, frame_num)

        if R is not None and num_inliers >= 15:
            # Build delta transform
            delta_T = np.eye(4)
            delta_T[:3, :3] = R
            delta_T[:3, 3] = tvec.flatten()

            # Accumulate pose: PnP gives transform from prev camera to curr camera
            # Inverse gives world motion
            candidate_pose = self.current_pose @ np.linalg.inv(delta_T)

            # Motion validation - reject implausible jumps
            delta_trans = np.linalg.norm(candidate_pose[:3, 3] - self.current_pose[:3, 3])
            R_delta = self.current_pose[:3, :3].T @ candidate_pose[:3, :3]
            trace_val = max(-1.0, min(3.0, np.trace(R_delta)))
            angle_delta = abs(math.acos(max(-1.0, min(1.0, (trace_val - 1) / 2.0))))

            if delta_trans < 0.15 and angle_delta < math.radians(15):
                self.current_pose = candidate_pose

        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        self.last_image = image.copy()
        self.last_depth = depth_image.copy()

        return self.current_pose.copy()

    def _visualize_keypoints(self, image, keypoints, frame_num):
        vis_img = cv2.drawKeypoints(image, keypoints, None,
                                    color=(0, 255, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        info_text = [
            f"Frame: {frame_num}",
            f"Keypoints: {len(keypoints)}",
            "Status: Initializing"
        ]
        y_offset = 30
        for text in info_text:
            cv2.putText(vis_img, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        cv2.imshow('Visual Odometry - Frame Processing', vis_img)
        cv2.waitKey(1)

    def _visualize_matching(self, img1, img2, kp1, kp2, matches, R, t, inliers, frame_num):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        vis_img[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

        for match in matches[:50]:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2_shifted = (pt2[0] + w1, pt2[1])
            cv2.line(vis_img, pt1, pt2_shifted, (0, 255, 0), 1)
            cv2.circle(vis_img, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis_img, pt2_shifted, 3, (255, 0, 0), -1)

        info_text = [
            f"Frame: {frame_num}",
            f"Matches: {len(matches)}",
            f"PnP Inliers: {inliers}",
            f"Status: {'OK' if R is not None else 'FAILED'}"
        ]
        if R is not None and t is not None:
            info_text.append(f"Translation: {np.linalg.norm(t):.4f}m")
            trace_val = max(-1.0, min(3.0, np.trace(R)))
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (trace_val - 1) / 2.0))))
            info_text.append(f"Rotation: {angle:.2f} deg")

        pos = self.current_pose[:3, 3]
        info_text.append(f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        y_offset = 30
        for text in info_text:
            cv2.putText(vis_img, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        cv2.imshow('Visual Odometry - Frame Processing', vis_img)
        cv2.waitKey(1)


# ====================================================================================
# MAP BUILDER
# ====================================================================================

class MapBuilder:
    """Builds a sparse 3D point cloud from RGB-D frames."""
    def __init__(self, fx, fy, cx, cy, depth_scale=5000.0, voxel_size=0.05):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.voxel_size = voxel_size
        self.global_points = []
        self.global_colors = []
        self.voxel_set = set()

    def integrate_frame(self, rgb_img, depth_img, pose):
        """Add points from a frame to the global map with voxel deduplication."""
        step = 20
        h, w = depth_img.shape
        R = pose[:3, :3]
        t = pose[:3, 3]

        for v in range(0, h, step):
            for u in range(0, w, step):
                d_val = depth_img[v, u]
                if d_val == 0:
                    continue
                z = float(d_val) / self.depth_scale
                if 0.5 < z < 5.0:
                    x_c = (u - self.cx) * z / self.fx
                    y_c = (v - self.cy) * z / self.fy
                    p_c = np.array([x_c, y_c, z])
                    p_w = R @ p_c + t

                    # Voxel grid deduplication
                    voxel_key = (int(round(p_w[0] / self.voxel_size)),
                                 int(round(p_w[1] / self.voxel_size)),
                                 int(round(p_w[2] / self.voxel_size)))
                    if voxel_key in self.voxel_set:
                        continue
                    self.voxel_set.add(voxel_key)

                    color = rgb_img[v, u] / 255.0
                    self.global_points.append(p_w)
                    self.global_colors.append([color[2], color[1], color[0]])  # BGR to RGB


# ====================================================================================
# ATE CALCULATION
# ====================================================================================

def compute_ate(vo_poses, gt_poses):
    """
    Compute Absolute Trajectory Error (ATE) after aligning VO to GT.
    Alignment uses the first frame as reference.
    """
    # Align VO trajectory to GT using first pose
    T_align = gt_poses[0] @ np.linalg.inv(vo_poses[0])

    errors = []
    aligned_positions = []
    gt_positions = []

    for vo_pose, gt_pose in zip(vo_poses, gt_poses):
        aligned = T_align @ vo_pose
        aligned_pos = aligned[:3, 3]
        gt_pos = gt_pose[:3, 3]
        error = np.linalg.norm(aligned_pos - gt_pos)
        errors.append(error)
        aligned_positions.append(aligned_pos)
        gt_positions.append(gt_pos)

    errors = np.array(errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    mean_err = np.mean(errors)
    max_err = np.max(errors)

    print(f"\n{'='*50}")
    print(f"Absolute Trajectory Error (ATE)")
    print(f"{'='*50}")
    print(f"  RMSE:    {rmse:.4f} m")
    print(f"  Mean:    {mean_err:.4f} m")
    print(f"  Max:     {max_err:.4f} m")
    print(f"  Frames:  {len(errors)}")
    print(f"{'='*50}")

    return np.array(aligned_positions), np.array(gt_positions), rmse


# ====================================================================================
# SMOOTHING
# ====================================================================================

def smooth_path(points, window_size=11, poly_order=3):
    """Smooth trajectory using Savitzky-Golay filter."""
    if len(points) < window_size:
        window_size = len(points) if len(points) % 2 == 1 else len(points) - 1
        if window_size < 3:
            return points
    if window_size % 2 == 0:
        window_size += 1
    poly_order = min(poly_order, window_size - 1)

    smoothed = np.copy(points)
    for i in range(3):
        smoothed[:, i] = savgol_filter(points[:, i], window_size, poly_order, mode='interp')
    return smoothed


# ====================================================================================
# PANGOLIN VIEWER
# ====================================================================================

def draw_camera(pose, size=0.1, color=(0.0, 1.0, 0.0)):
    w = size * 0.75
    h = size * 0.5
    z = size
    gl.glPushMatrix()
    gl.glMultMatrixf(pose.T.flatten())
    gl.glLineWidth(1)
    gl.glColor3f(*color)
    gl.glBegin(gl.GL_LINES)
    for x, y in [(-w, -h), (w, -h), (w, h), (-w, h)]:
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(x, y, z)
    corners = [(-w, -h, z), (w, -h, z), (w, h, z), (-w, h, z)]
    for i in range(4):
        gl.glVertex3f(*corners[i])
        gl.glVertex3f(*corners[(i + 1) % 4])
    gl.glEnd()
    gl.glPopMatrix()


def draw_trajectory(points, color=(0.0, 0.0, 1.0), width=2):
    if len(points) < 2:
        return
    gl.glLineWidth(width)
    gl.glColor3f(*color)
    gl.glBegin(gl.GL_LINE_STRIP)
    for p in points:
        gl.glVertex3f(p[0], p[1], p[2])
    gl.glEnd()


def pangolin_viewer(vo_path, gt_path, poses, map_points=None, map_colors=None):
    """Interactive Pangolin viewer showing VO trajectory, GT, and point cloud map."""
    w, h = 1024, 768
    pangolin.CreateWindowAndBind('Visual Odometry - RGB-D SLAM', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    # Compute view from combined trajectories
    all_pts = np.vstack([vo_path, gt_path]) if gt_path is not None and len(gt_path) > 0 else vo_path
    center = all_pts.mean(axis=0)
    extent = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    view_dist = max(extent * 1.5, 2.0)

    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.1, 1000),
        pangolin.ModelViewLookAt(
            center[0] - view_dist, center[1] - view_dist, center[2] + view_dist,
            center[0], center[1], center[2],
            pangolin.AxisZ
        )
    )
    handler = pangolin.Handler3D(scam)
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(
        pangolin.Attach(0.0), pangolin.Attach(1.0),
        pangolin.Attach(0.0), pangolin.Attach(1.0),
        -w / h
    )
    dcam.SetHandler(handler)

    print("\nPangolin viewer:")
    print("  Blue  = VO trajectory (smoothed)")
    print("  Green = Ground Truth trajectory")
    print("  Orange = Camera frustums")
    print("  Colored points = 3D map")
    print("  Left-click+drag: Rotate | Right-click+drag: Translate | Scroll: Zoom\n")

    step = max(1, len(poses) // 50)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.05, 0.05, 0.1, 1.0)
        dcam.Activate(scam)

        # Ground grid
        gl.glLineWidth(1)
        gl.glColor4f(0.3, 0.3, 0.3, 0.5)
        gl.glBegin(gl.GL_LINES)
        grid_range = int(extent * 2) + 5
        for i in range(-grid_range, grid_range + 1):
            gl.glVertex3f(i, -grid_range, 0)
            gl.glVertex3f(i, grid_range, 0)
            gl.glVertex3f(-grid_range, i, 0)
            gl.glVertex3f(grid_range, i, 0)
        gl.glEnd()

        # Draw GT trajectory (green)
        if gt_path is not None and len(gt_path) > 1:
            draw_trajectory(gt_path, color=(0.0, 1.0, 0.3), width=3)

        # Draw VO trajectory (blue)
        draw_trajectory(vo_path, color=(0.2, 0.5, 1.0), width=2)

        # Start/end markers
        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(*vo_path[0])
        gl.glEnd()
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glBegin(gl.GL_POINTS)
        gl.glVertex3f(*vo_path[-1])
        gl.glEnd()

        # Camera frustums
        for i in range(0, len(poses), step):
            color = (0.0, 1.0, 0.0) if i == 0 else (0.8, 0.4, 0.0)
            draw_camera(poses[i], size=extent * 0.02, color=color)
        draw_camera(poses[-1], size=extent * 0.03, color=(1.0, 0.0, 0.0))

        # Point cloud map
        if map_points and len(map_points) > 0:
            gl.glPointSize(2)
            gl.glBegin(gl.GL_POINTS)
            for p, c in zip(map_points, map_colors):
                gl.glColor3f(c[0], c[1], c[2])
                gl.glVertex3f(p[0], p[1], p[2])
            gl.glEnd()

        pangolin.FinishFrame()


# ====================================================================================
# MAIN PIPELINE
# ====================================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'rgbd_dataset_freiburg2_pioneer_slam3')

    if not os.path.exists(data_path):
        # Fallback to symlink
        data_path = os.path.join(script_dir, 'Dataset_VO')
    if not os.path.exists(data_path):
        print(f"Dataset not found.")
        return

    print("Loading and synchronizing RGB-D dataset...")
    loader = DatasetLoader(data_path)
    frames = loader.get_synchronized_frames()
    if not frames:
        print("No synchronized frames found.")
        return

    # TUM FR2 camera intrinsics
    fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
    depth_scale = 5000.0

    print("Initializing Visual Odometry (PnP + Depth)...")
    vo = VisualOdometry(fx, fy, cx, cy, depth_scale=depth_scale, visualize=True)
    mapper = MapBuilder(fx, fy, cx, cy, depth_scale=depth_scale, voxel_size=0.05)

    vo_poses = []
    gt_poses = []

    print("Processing frames...")
    print("Press 'q' on the visualization window to skip visualization.\n")

    for i, frame in enumerate(frames):
        rgb = cv2.imread(frame['rgb'])
        depth = cv2.imread(frame['depth'], cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue

        pose = vo.process_frame(rgb, depth, frame_num=i)
        vo_poses.append(pose.copy())

        if frame['gt_pose'] is not None:
            gt_poses.append(frame['gt_pose'].copy())
        else:
            gt_poses.append(None)

        # Build map every 10 frames
        if i % 10 == 0:
            mapper.integrate_frame(rgb, depth, pose)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(frames)} frames")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Visualization skipped")
            vo.visualize = False
            cv2.destroyAllWindows()

    print(f"\nProcessed all {len(frames)} frames")
    cv2.destroyAllWindows()

    # Extract positions
    vo_positions = np.array([p[:3, 3] for p in vo_poses])

    # Filter frames that have ground truth
    valid_idx = [i for i, g in enumerate(gt_poses) if g is not None]
    vo_poses_with_gt = [vo_poses[i] for i in valid_idx]
    gt_poses_valid = [gt_poses[i] for i in valid_idx]

    # Compute ATE
    gt_path = None
    if len(gt_poses_valid) > 0:
        aligned_positions, gt_positions, ate_rmse = compute_ate(vo_poses_with_gt, gt_poses_valid)
        gt_path = gt_positions

        # Use aligned trajectory for display
        T_align = gt_poses_valid[0] @ np.linalg.inv(vo_poses_with_gt[0])
        vo_positions_aligned = np.array([(T_align @ p)[:3, 3] for p in vo_poses])
        aligned_poses = [T_align @ p for p in vo_poses]

        # Also transform map points
        aligned_map_points = [T_align[:3, :3] @ p + T_align[:3, 3] for p in mapper.global_points]
    else:
        vo_positions_aligned = vo_positions
        aligned_poses = vo_poses
        aligned_map_points = mapper.global_points

    # Smooth VO trajectory
    smoothed_vo = smooth_path(vo_positions_aligned, window_size=11, poly_order=3)

    # Launch Pangolin viewer
    print("\nLaunching Pangolin 3D viewer...")
    pangolin_viewer(smoothed_vo, gt_path, aligned_poses,
                    map_points=aligned_map_points, map_colors=mapper.global_colors)

    print("\nDone!")


if __name__ == '__main__':
    main()
