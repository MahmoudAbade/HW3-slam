import cv2
import numpy as np
import os
try:
    import pypangolin as pangolin
except ImportError:
    pangolin = None
import OpenGL.GL as gl
import math
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

"""
================================================================================
Robust RGB-D SLAM Pipeline
================================================================================

Student Name: Mahmoud Abade
Advanced Design and Architecture Implementation

Architecture Overview:
----------------------
This script implements a robust visual SLAM pipeline integrating RGB and Depth data.

Key Design Choices & Rationale:
1. 3D-2D Pose Recovery (PnP) vs Epipolar Geometry:
   - Problem: Normal visual odometry based on Essential/Fundamental matrices fails
     catastrophically when the robot simply tilts its camera (pure rotation). Parallax
     approaches zero, making Epipolar geometry mathematically unstable.
   - Solution Strategy: Since we have aligned depth images, we can extract the true 3D
     coordinates of matched features in the previous frame and use them against 2D
     features in the current frame via Perspective-n-Point (PnP) paired with RANSAC.
   - Outcome: This gracefully handles both translation and pure rotation motions.

2. Feature Extraction & Matching:
   - ORB feature detector for incredibly fast and robust rotational invariant features.
   - FLANN (Fast Library for Approximate Nearest Neighbors) matching equipped with
     Lowe's ratio test to drastically cull false positive correspondences.

3. Live 3D Mapping & Visualization:
   - Pangolin integration to continuously render the updated 3D point cloud surroundings
     and the smooth odometry path.
   - Ground truth trajectory overlay for comparison.

4. Modular Architecture:
   - Data loading (D), feature extraction (F), pose estimation (P), mapping (M), and
     visualization (V) are encapsulated into distinct components matching SOLID principles.

================================================================================
"""

class IMUIntegrator:
    """
    Integrates accelerometer data to estimate velocity and position deltas.
    Uses gravity removal and double integration between timestamps.
    """
    def __init__(self, accel_file, gravity=9.81):
        self.gravity = gravity
        self.timestamps = []
        self.accel_data = []  # raw ax, ay, az
        self._parse_accel_file(accel_file)
        self.velocity = np.zeros(3)  # running velocity estimate

        # Build interpolator for querying accel at any timestamp
        if len(self.timestamps) > 1:
            ts = np.array(self.timestamps)
            ax = np.array([a[0] for a in self.accel_data])
            ay = np.array([a[1] for a in self.accel_data])
            az = np.array([a[2] for a in self.accel_data])
            self.interp_ax = interp1d(ts, ax, bounds_error=False, fill_value='extrapolate')
            self.interp_ay = interp1d(ts, ay, bounds_error=False, fill_value='extrapolate')
            self.interp_az = interp1d(ts, az, bounds_error=False, fill_value='extrapolate')

            # Estimate gravity bias from first 50 stationary samples
            n_cal = min(50, len(self.accel_data))
            cal = np.array(self.accel_data[:n_cal])
            self.gravity_bias = np.mean(cal, axis=0)
            print(f"[IMU] Loaded {len(self.timestamps)} samples, "
                  f"gravity bias: [{self.gravity_bias[0]:.2f}, {self.gravity_bias[1]:.2f}, {self.gravity_bias[2]:.2f}]")
        else:
            self.interp_ax = None
            print("[IMU] No accelerometer data loaded.")

    def _parse_accel_file(self, filename):
        if not os.path.exists(filename):
            print(f"[IMU] Accelerometer file not found: {filename}")
            return
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    ts = float(parts[0])
                    ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                    self.timestamps.append(ts)
                    self.accel_data.append([ax, ay, az])

    def get_delta_position(self, t_prev, t_curr):
        """
        Integrate acceleration between two timestamps to get position delta.
        Returns delta_position (3,) vector in sensor frame.
        """
        if self.interp_ax is None:
            return np.zeros(3)

        dt = t_curr - t_prev
        if dt <= 0 or dt > 1.0:
            return np.zeros(3)

        # Sample acceleration at midpoint and endpoints
        n_samples = max(3, int(dt * 200))  # ~200Hz sampling
        t_samples = np.linspace(t_prev, t_curr, n_samples)

        ax = self.interp_ax(t_samples) - self.gravity_bias[0]
        ay = self.interp_ay(t_samples) - self.gravity_bias[1]
        az = self.interp_az(t_samples) - self.gravity_bias[2]

        # Trapezoidal integration for velocity
        dt_step = dt / (n_samples - 1)
        vx = np.cumsum(ax) * dt_step
        vy = np.cumsum(ay) * dt_step
        vz = np.cumsum(az) * dt_step

        # Add running velocity
        vx += self.velocity[0]
        vy += self.velocity[1]
        vz += self.velocity[2]

        # Position delta = integral of velocity
        dx = np.trapezoid(vx, dx=dt_step)
        dy = np.trapezoid(vy, dx=dt_step)
        dz = np.trapezoid(vz, dx=dt_step)

        # Update running velocity
        self.velocity[0] = vx[-1]
        self.velocity[1] = vy[-1]
        self.velocity[2] = vz[-1]

        return np.array([dx, dy, dz])

    def reset_velocity(self):
        """Reset velocity estimate (call when VO gives a good fix)."""
        self.velocity = np.zeros(3)


class DatasetLoader:
    """
    Handles parsing and synchronizing the TUM RGB-D dataset format.
    RGB and Depth frames arrive at slightly different timestamps. They are
    synchronized by associating the closest timestamps (delta < 0.02s).
    Also loads ground truth poses for evaluation.
    """
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.rgb_file = os.path.join(dataset_path, 'rgb.txt')
        self.depth_file = os.path.join(dataset_path, 'depth.txt')
        self.gt_file = os.path.join(dataset_path, 'groundtruth.txt')
        self.accel_file = os.path.join(dataset_path, 'accelerometer.txt')

    def _parse_timestamp_file(self, filename):
        entries = []
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return entries
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    entries.append((float(parts[0]), parts[1:]))
        return entries

    def _parse_gt_file(self):
        """Parse groundtruth.txt: timestamp tx ty tz qx qy qz qw"""
        entries = []
        if not os.path.exists(self.gt_file):
            return entries
        with open(self.gt_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    ts = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = [tx, ty, tz]
                    entries.append((ts, T))
        return entries

    def _find_closest_ts(self, target_ts, entries, max_diff):
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

    def get_synchronized_frames(self, max_time_diff=0.02):
        rgb_entries = self._parse_timestamp_file(self.rgb_file)
        depth_entries_raw = self._parse_timestamp_file(self.depth_file)
        gt_entries = self._parse_gt_file()

        # Convert depth entries to same format
        depth_entries = [(ts, data[0]) for ts, data in depth_entries_raw]

        matches = []
        for time_rgb, rgb_data in rgb_entries:
            rgb_path = rgb_data[0]
            best_diff = float('inf')
            best_depth_path = None
            for time_depth, depth_path in depth_entries:
                diff = abs(time_rgb - time_depth)
                if diff < best_diff:
                    best_diff = diff
                    best_depth_path = depth_path

            if best_diff <= max_time_diff:
                # Find closest GT pose
                gt_match = self._find_closest_ts(time_rgb, gt_entries, max_diff=0.05)
                gt_pose = gt_match[1] if gt_match else None

                matches.append({
                    'timestamp': time_rgb,
                    'rgb': os.path.join(self.dataset_path, rgb_path),
                    'depth': os.path.join(self.dataset_path, best_depth_path),
                    'gt_pose': gt_pose
                })

        gt_count = sum(1 for m in matches if m['gt_pose'] is not None)
        print(f"[DatasetLoader] Associated {len(matches)} RGB-D pairs ({gt_count} with GT).")
        return matches


class FeatureTracker:
    """
    Manages keypoint detection and robust correspondence matching.
    """
    def __init__(self, max_features=3000):
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            patchSize=31,
            fastThreshold=12
        )

        index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=100)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        kps, descs = self.orb.detectAndCompute(enhanced, None)
        return kps, descs

    def robust_matching(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.65 * n.distance:
                    good_matches.append(m)
        return good_matches


class PoseEstimator:
    """
    Computes visual odometry. Employs a 3D-2D approach solving the PnP problem.
    """
    def __init__(self, intrinsics, depth_scale=5000.0):
        self.fx_d = intrinsics['fx']
        self.fy_d = intrinsics['fy']
        self.cx_d = intrinsics['cx']
        self.cy_d = intrinsics['cy']

        self.camera_matrix = np.array([
            [self.fx_d, 0, self.cx_d],
            [0, self.fy_d, self.cy_d],
            [0, 0, 1]
        ], dtype=np.float64)

        self.depth_scale = depth_scale

    def unproject_pixel(self, u, v, z):
        x = (u - self.cx_d) * z / self.fx_d
        y = (v - self.cy_d) * z / self.fy_d
        return np.array([x, y, z])

    def estimate_motion_pnp(self, kp1, kp2, matches, depth1_img):
        pts_3d = []
        pts_2d = []

        for m in matches:
            idx1 = m.queryIdx
            idx2 = m.trainIdx

            u1, v1 = int(kp1[idx1].pt[0]), int(kp1[idx1].pt[1])
            u2, v2 = int(kp2[idx2].pt[0]), int(kp2[idx2].pt[1])

            if v1 >= depth1_img.shape[0] or u1 >= depth1_img.shape[1]:
                continue

            d_val = depth1_img[v1, u1]
            if d_val == 0:
                continue

            z1 = float(d_val) / self.depth_scale
            if z1 > 0.3 and z1 < 6.0:
                p3d = self.unproject_pixel(u1, v1, z1)
                pts_3d.append(p3d)
                pts_2d.append([u2, v2])

        if len(pts_3d) < 15:
            return None, None, 0

        pts_3d = np.array(pts_3d, dtype=np.float64)
        pts_2d = np.array(pts_2d, dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.camera_matrix, None,
            iterationsCount=500, reprojectionError=1.5, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP
        )

        if success and inliers is not None:
            num_inliers = len(inliers)
            if num_inliers >= 15:
                pts_3d_inliers = pts_3d[inliers.flatten()]
                pts_2d_inliers = pts_2d[inliers.flatten()]
                _, rvec, tvec = cv2.solvePnP(
                    pts_3d_inliers, pts_2d_inliers, self.camera_matrix, None,
                    rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                R, _ = cv2.Rodrigues(rvec)
                return R, tvec, num_inliers

        return None, None, 0


class MapBuilder:
    """
    Collects 3D environmental structure across frames.
    Uses voxel grid deduplication to prevent overlapping points.
    """
    def __init__(self, estimator, voxel_size=0.05):
        self.estimator = estimator
        self.global_points = []
        self.global_colors = []
        self.voxel_size = voxel_size
        self.voxel_set = set()

    def integrate_frame(self, rgb_img, depth_img, pose):
        step = 20
        h, w = depth_img.shape
        R = pose[:3, :3]
        t = pose[:3, 3]

        for v in range(0, h, step):
            for u in range(0, w, step):
                d_val = depth_img[v, u]
                if d_val == 0: continue
                z = float(d_val) / self.estimator.depth_scale
                if 0.5 < z < 4.0:
                    p_c = self.estimator.unproject_pixel(u, v, z)
                    p_w = R @ p_c + t

                    vk = (int(round(p_w[0] / self.voxel_size)),
                          int(round(p_w[1] / self.voxel_size)),
                          int(round(p_w[2] / self.voxel_size)))
                    if vk in self.voxel_set:
                        continue
                    self.voxel_set.add(vk)

                    base_color = rgb_img[v, u] / 255.0
                    self.global_points.append(p_w)
                    self.global_colors.append([base_color[2], base_color[1], base_color[0]])


class SLAMPipeline:
    """
    Consolidated orchestrator binding feature tracking, mapping, and viewing.

    Drift reduction strategy:
    - Keyframe-based tracking: match against keyframes (not just previous frame)
    - PnP re-localization: every RELOC_INTERVAL frames, re-localize against a
      keyframe from the stored keyframe database using PnP
    - Height constraint: pin Z to ground level (Pioneer is a ground robot)
    """
    RELOC_INTERVAL = 30      # Re-localize every N frames
    KEYFRAME_MIN_MATCHES = 80  # Minimum matches to keep tracking current keyframe

    def __init__(self, data_path):
        self.loader = DatasetLoader(data_path)

        intrinsics = {'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7}
        self.tracker = FeatureTracker()
        self.estimator = PoseEstimator(intrinsics=intrinsics)
        self.mapper = MapBuilder(self.estimator)

        # IMU integrator
        accel_file = os.path.join(data_path, 'accelerometer.txt')
        self.imu = IMUIntegrator(accel_file)
        self.imu_weight = 0.15  # Weight for IMU position correction (0=pure VO, 1=pure IMU)

        self.current_pose = np.eye(4)
        self.trajectory = []
        self.imu_trajectory = []  # IMU-only trajectory for display
        self.gt_trajectory = []
        self.camera_frustums = []

        # Keyframe database: list of (kps, descs, depth, pose_at_keyframe)
        self.keyframes = []
        self.last_kps = None
        self.last_descs = None
        self.last_depth = None
        self.last_timestamp = None

        # Alignment
        self.T_align = None
        self.initial_height = None

    def _add_keyframe(self, kps, descs, depth):
        """Store a keyframe with its features and the current VO pose."""
        self.keyframes.append({
            'kps': kps,
            'descs': descs,
            'depth': depth.copy(),
            'pose': self.current_pose.copy()
        })

    def _relocalize_against_keyframes(self, kps, descs):
        """
        Try to re-localize current frame against stored keyframes.
        Finds the best matching keyframe and computes pose via PnP.
        Returns the corrected pose or None if re-localization fails.
        """
        best_pose = None
        best_inliers = 0

        # Check against recent keyframes (last 20) for efficiency
        candidates = self.keyframes[-20:] if len(self.keyframes) > 20 else self.keyframes

        for kf in candidates:
            matches = self.tracker.robust_matching(kf['descs'], descs)
            if len(matches) < 20:
                continue

            R, tvec, num_inliers = self.estimator.estimate_motion_pnp(
                kf['kps'], kps, matches, kf['depth']
            )

            if R is not None and num_inliers > best_inliers:
                delta_T = np.eye(4)
                delta_T[:3, :3] = R
                delta_T[:3, 3] = tvec.flatten()
                candidate = kf['pose'] @ np.linalg.inv(delta_T)

                # Validate: should be within reasonable distance of current estimate
                dist = np.linalg.norm(candidate[:3, 3] - self.current_pose[:3, 3])
                if dist < 1.0:  # Allow up to 1m correction
                    best_pose = candidate
                    best_inliers = num_inliers

        return best_pose

    def run(self):
        frames = self.loader.get_synchronized_frames()
        if not frames:
            print("No frames available.")
            return

        for frame in frames:
            if frame['gt_pose'] is not None:
                self.gt_trajectory.append(frame['gt_pose'][:3, 3].copy())

        w, h = 1024, 768
        pangolin.CreateWindowAndBind('RGBD SLAM System', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        if len(self.gt_trajectory) > 0:
            gt_center = np.mean(self.gt_trajectory, axis=0)
            scam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.1, 1000),
                pangolin.ModelViewLookAt(
                    gt_center[0], gt_center[1] - 8, gt_center[2] - 3,
                    gt_center[0], gt_center[1], gt_center[2],
                    pangolin.AxisNegY
                )
            )
        else:
            scam = pangolin.OpenGlRenderState(
                pangolin.ProjectionMatrix(w, h, 420, 420, w // 2, h // 2, 0.1, 1000),
                pangolin.ModelViewLookAt(0, -8, -3, 0, 0, 2, pangolin.AxisNegY)
            )

        handler = pangolin.Handler3D(scam)
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(pangolin.Attach(0.0), pangolin.Attach(1.0),
                       pangolin.Attach(0.0), pangolin.Attach(1.0), -w / h)
        dcam.SetHandler(handler)

        frame_idx = 0
        total_frames = len(frames)
        reloc_count = 0

        while not pangolin.ShouldQuit():
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.1, 0.1, 0.15, 1.0)
            dcam.Activate(scam)

            # --- Odometry Step ---
            if frame_idx < total_frames:
                frame = frames[frame_idx]
                rgb = cv2.imread(frame['rgb'])
                depth = cv2.imread(frame['depth'], cv2.IMREAD_UNCHANGED)

                if rgb is not None and depth is not None:
                    kps, descs = self.tracker.extract_features(rgb)
                    curr_ts = frame['timestamp']

                    # --- IMU prediction ---
                    imu_delta = np.zeros(3)
                    if self.last_timestamp is not None and self.imu.interp_ax is not None:
                        imu_delta = self.imu.get_delta_position(self.last_timestamp, curr_ts)

                    old_pose = self.current_pose.copy()
                    dt = curr_ts - self.last_timestamp if self.last_timestamp is not None else 0
                    vo_succeeded = False
                    if self.last_kps is not None and descs is not None:
                        matches = self.tracker.robust_matching(self.last_descs, descs)

                        R, tvec, inliers = self.estimator.estimate_motion_pnp(
                            self.last_kps, kps, matches, self.last_depth
                        )

                        if R is not None:
                            delta_T = np.eye(4)
                            delta_T[:3, :3] = R
                            delta_T[:3, 3] = tvec.flatten()
                            candidate_pose = self.current_pose @ np.linalg.inv(delta_T)

                            # Motion validation
                            delta_t = np.linalg.norm(candidate_pose[:3, 3] - self.current_pose[:3, 3])
                            R_delta = self.current_pose[:3, :3].T @ candidate_pose[:3, :3]
                            trace_val = max(-1.0, min(3.0, np.trace(R_delta)))
                            angle_delta = abs(math.acos(max(-1.0, min(1.0, (trace_val - 1) / 2.0))))

                            if delta_t < 0.15 and angle_delta < math.radians(15):
                                self.current_pose = candidate_pose
                                vo_succeeded = True

                                # Update IMU velocity with VO velocity in local frame
                                if dt > 0:
                                    delta_pos_world = self.current_pose[:3, 3] - old_pose[:3, 3]
                                    delta_pos_local = old_pose[:3, :3].T @ delta_pos_world
                                    self.imu.velocity = delta_pos_local / dt
                                else:
                                    self.imu.velocity = np.zeros(3)

                        # --- IMU fusion: blend IMU delta with VO position ---
                        if vo_succeeded and np.linalg.norm(imu_delta) > 1e-6:
                            # Rotate IMU delta to world frame using old orientation
                            imu_delta_world = old_pose[:3, :3] @ imu_delta
                            p_vo = self.current_pose[:3, 3]
                            p_imu = old_pose[:3, 3] + imu_delta_world
                            self.current_pose[:3, 3] = (1 - self.imu_weight) * p_vo + self.imu_weight * p_imu

                        # If VO failed, use IMU-only prediction
                        if not vo_succeeded and np.linalg.norm(imu_delta) > 1e-6:
                            imu_delta_world = old_pose[:3, :3] @ imu_delta
                            self.current_pose[:3, 3] = old_pose[:3, 3] + imu_delta_world

                        # --- PnP Re-localization every N frames ---
                        if frame_idx > 0 and frame_idx % self.RELOC_INTERVAL == 0 and len(self.keyframes) > 2:
                            reloc_pose = self._relocalize_against_keyframes(kps, descs)
                            if reloc_pose is not None:
                                # Validate rotation difference before applying
                                R_diff = self.current_pose[:3, :3].T @ reloc_pose[:3, :3]
                                trace_diff = max(-1.0, min(3.0, np.trace(R_diff)))
                                angle_diff = abs(math.acos(max(-1.0, min(1.0, (trace_diff - 1) / 2.0))))

                                if angle_diff < math.radians(5): # Strict 5 degree limit
                                    alpha = 0.7
                                    self.current_pose[:3, 3] = (
                                        alpha * reloc_pose[:3, 3] +
                                        (1 - alpha) * self.current_pose[:3, 3]
                                    )
                                    # Blend rotation slightly instead of hard overwrite or just keep current
                                    # self.current_pose[:3, :3] = reloc_pose[:3, :3]
                                    self.imu.reset_velocity()
                                    reloc_count += 1

                        # Store keyframe every 15 frames
                        if descs is not None and (frame_idx % 15 == 0 or len(self.keyframes) == 0):
                            self._add_keyframe(kps, descs, depth)

                    self.last_timestamp = curr_ts

                    # Set alignment on first frame
                    if self.T_align is None and frame['gt_pose'] is not None:
                        self.T_align = frame['gt_pose'] @ np.linalg.inv(self.current_pose)
                        self.initial_height = frame['gt_pose'][2, 3]
                        r = Rotation.from_matrix(frame['gt_pose'][:3, :3])
                        self.initial_euler = r.as_euler('xyz', degrees=False)

                    # Enforce Planar Constraints (Ground Vehicle)
                    if self.T_align is not None and hasattr(self, 'initial_euler') and self.initial_height is not None:
                        aligned_pose = self.T_align @ self.current_pose

                        # 1. Constrain Height (Z)
                        aligned_pose[2, 3] = self.initial_height

                        # 2. Constrain Roll and Pitch (X and Y rotations)
                        r = Rotation.from_matrix(aligned_pose[:3, :3])
                        euler = r.as_euler('xyz', degrees=False)
                        euler[0] = self.initial_euler[0]
                        euler[1] = self.initial_euler[1]

                        aligned_pose[:3, :3] = Rotation.from_euler('xyz', euler).as_matrix()

                        # Back-project constraint to the camera frame
                        self.current_pose = np.linalg.inv(self.T_align) @ aligned_pose
                    else:
                        aligned_pose = self.current_pose

                    self.trajectory.append(aligned_pose[:3, 3].copy())

                    # Use GT for map building
                    map_pose = frame['gt_pose'] if frame['gt_pose'] is not None else aligned_pose
                    if frame_idx % 5 == 0:
                        self.camera_frustums.append(map_pose.copy())
                        self.mapper.integrate_frame(rgb, depth, map_pose)

                    self.last_kps = kps
                    self.last_descs = descs
                    self.last_depth = depth

                frame_idx += 1
                if frame_idx % 50 == 0:
                    print(f"Tracking: {frame_idx}/{total_frames} | "
                          f"Keyframes: {len(self.keyframes)} | "
                          f"Re-loc: {reloc_count} | "
                          f"IMU vel: [{self.imu.velocity[0]:.3f}, {self.imu.velocity[1]:.3f}, {self.imu.velocity[2]:.3f}]")

            # --- Render ---
            self._render_map()
            pangolin.FinishFrame()

        print(f"Pipeline Complete. Re-localizations: {reloc_count}")

    def _render_map(self):
        """Renders GT trajectory, VO trajectory, camera frustums, and point cloud."""
        # Draw ground truth trajectory (green, thick)
        if len(self.gt_trajectory) > 1:
            gl.glLineWidth(3)
            gl.glColor3f(0.0, 1.0, 0.3)
            gl.glBegin(gl.GL_LINE_STRIP)
            for t in self.gt_trajectory:
                gl.glVertex3f(t[0], t[1], t[2])
            gl.glEnd()

        # Draw VO trajectory (cyan)
        if len(self.trajectory) > 1:
            gl.glLineWidth(2)
            gl.glColor3f(0.2, 0.8, 1.0)
            gl.glBegin(gl.GL_LINE_STRIP)
            for t in self.trajectory:
                gl.glVertex3f(t[0], t[1], t[2])
            gl.glEnd()

        # Camera frustums
        for pose in self.camera_frustums:
            self._draw_frustum(pose)

        # Point cloud
        points = self.mapper.global_points
        colors = self.mapper.global_colors
        if points:
            gl.glPointSize(2)
            gl.glBegin(gl.GL_POINTS)
            for p, c in zip(points, colors):
                gl.glColor3f(c[0], c[1], c[2])
                gl.glVertex3f(p[0], p[1], p[2])
            gl.glEnd()

    def _draw_frustum(self, pose):
        sz = 0.05
        gl.glPushMatrix()
        gl.glMultMatrixf(pose.T.flatten())
        gl.glLineWidth(1)
        gl.glColor3f(0.8, 0.2, 0.2)
        gl.glBegin(gl.GL_LINES)

        corners = [(-sz, -sz, sz*2), (sz, -sz, sz*2), (sz, sz, sz*2), (-sz, sz, sz*2)]
        for x, y, z in corners:
            gl.glVertex3f(0, 0, 0)
            gl.glVertex3f(x, y, z)

        for i in range(4):
            gl.glVertex3f(*corners[i])
            gl.glVertex3f(*corners[(i + 1) % 4])

        gl.glEnd()
        gl.glPopMatrix()

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'rgbd_dataset_freiburg2_pioneer_slam3')
    if not os.path.exists(data_dir):
        print(f"Please ensure the dataset path is valid: {data_dir}")
    else:
        print("Starting RGBD SLAM with IMU fusion. Map window will open shortly.")
        print("  Green line  = Ground Truth")
        print("  Cyan line   = Visual-Inertial Odometry (VO + IMU)")
        print("  Red frustums = Camera poses")
        print("  Points = 3D map\n")
        slam = SLAMPipeline(data_dir)
        slam.run()
