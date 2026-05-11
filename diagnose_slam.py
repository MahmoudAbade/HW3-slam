"""
Diagnostic: frame-to-frame VO + height constraint + tuned loop closure.
No GT dependency.
"""
import cv2
import numpy as np
import os
import math
from scipy.spatial.transform import Rotation

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'rgbd_dataset_freiburg2_pioneer_slam3')

def parse_file(filepath):
    entries = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'): continue
            parts = line.strip().split()
            if len(parts) >= 2:
                entries.append((float(parts[0]), parts[1:]))
    return entries

def find_closest(target_ts, entries, max_diff):
    best_diff, best = float('inf'), None
    for ts, data in entries:
        diff = abs(target_ts - ts)
        if diff < best_diff:
            best_diff, best = diff, (ts, data)
    return best if best_diff <= max_diff else None

def load_frames():
    rgb_entries = parse_file(os.path.join(DATA_DIR, 'rgb.txt'))
    depth_entries = parse_file(os.path.join(DATA_DIR, 'depth.txt'))
    gt_entries = parse_file(os.path.join(DATA_DIR, 'groundtruth.txt'))
    frames = []
    for ts_rgb, rgb_data in rgb_entries:
        d = find_closest(ts_rgb, depth_entries, 0.02)
        if d is None: continue
        g = find_closest(ts_rgb, gt_entries, 0.05)
        gt_pose = None
        if g:
            p = g[1]
            tx,ty,tz = float(p[0]),float(p[1]),float(p[2])
            qx,qy,qz,qw = float(p[3]),float(p[4]),float(p[5]),float(p[6])
            R = Rotation.from_quat([qx,qy,qz,qw]).as_matrix()
            T = np.eye(4); T[:3,:3] = R; T[:3,3] = [tx,ty,tz]
            gt_pose = T
        frames.append({
            'timestamp': ts_rgb,
            'rgb': os.path.join(DATA_DIR, rgb_data[0]),
            'depth': os.path.join(DATA_DIR, d[1][0]),
            'gt_pose': gt_pose
        })
    return frames

fx, fy, cx, cy = 520.9, 521.0, 325.1, 249.7
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
DEPTH_SCALE = 5000.0

orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8,
                     edgeThreshold=15, patchSize=31, fastThreshold=12)
index_params = dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
flann = cv2.FlannBasedMatcher(index_params, dict(checks=100))
clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return orb.detectAndCompute(clahe_obj.apply(gray), None)

def match_features(d1, d2, ratio=0.75):
    if d1 is None or d2 is None or len(d1)<2 or len(d2)<2: return []
    try: ms = flann.knnMatch(d1, d2, k=2)
    except: return []
    good = []
    for pair in ms:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

def estimate_pnp(kp1, kp2, matches, depth1):
    pts3, pts2 = [], []
    for m in matches:
        u1,v1 = int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])
        u2,v2 = int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])
        if v1>=depth1.shape[0] or u1>=depth1.shape[1]: continue
        d = depth1[v1,u1]
        if d==0: continue
        z = float(d)/DEPTH_SCALE
        if 0.3<z<6.0:
            pts3.append([(u1-cx)*z/fx, (v1-cy)*z/fy, z])
            pts2.append([u2,v2])
    if len(pts3)<15: return None, None, 0
    pts3 = np.array(pts3, dtype=np.float64)
    pts2 = np.array(pts2, dtype=np.float64)
    ok, rv, tv, inl = cv2.solvePnPRansac(pts3, pts2, K, None,
        iterationsCount=500, reprojectionError=1.5, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl)<15: return None, None, 0
    p3i, p2i = pts3[inl.flatten()], pts2[inl.flatten()]
    _, rv, tv = cv2.solvePnP(p3i, p2i, K, None, rvec=rv, tvec=tv,
                             useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rv)
    return R, tv, len(inl)


def main():
    print("Loading frames...")
    frames = load_frames()
    print(f"  {len(frames)} frames, {sum(1 for f in frames if f['gt_pose'] is not None)} with GT")

    KEYFRAME_INTERVAL = 10
    LOOP_CHECK_INTERVAL = 10
    LOOP_MIN_FRAME_GAP = 150   # ~15 seconds at 10Hz
    LOOP_MIN_INLIERS = 30
    LOOP_MAX_DRIFT = 2.5

    print("\nRunning VO + height constraint + loop closure...")
    pose = np.eye(4)
    prev_kps, prev_desc, prev_depth = None, None, None
    T_align = None
    initial_height = None
    keyframes = []
    vo_trajectory, gt_trajectory = [], []
    vo_ok, vo_fail, loop_count = 0, 0, 0

    for i, f in enumerate(frames):
        rgb = cv2.imread(f['rgb'])
        depth = cv2.imread(f['depth'], cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None: continue

        kps, desc = extract(rgb)

        if prev_kps is not None and desc is not None:
            ms = match_features(prev_desc, desc)
            if len(ms) >= 15:
                R, tv, ninl = estimate_pnp(prev_kps, kps, ms, prev_depth)
                if R is not None:
                    dT = np.eye(4); dT[:3,:3]=R; dT[:3,3]=tv.flatten()
                    cand = pose @ np.linalg.inv(dT)
                    dt = np.linalg.norm(cand[:3,3] - pose[:3,3])
                    Rd = pose[:3,:3].T @ cand[:3,:3]
                    tr = max(-1.0, min(3.0, np.trace(Rd)))
                    ang = abs(math.acos(max(-1.0, min(1.0, (tr-1)/2.0))))
                    if dt < 0.5 and ang < math.radians(30):
                        pose = cand; vo_ok += 1
                    else: vo_fail += 1
                else: vo_fail += 1
            else: vo_fail += 1

            # Height constraint
            if T_align is not None and initial_height is not None:
                al = T_align @ pose; al[2,3] = initial_height
                pose = np.linalg.inv(T_align) @ al

            # Store keyframe
            if desc is not None and (i % KEYFRAME_INTERVAL == 0 or len(keyframes) == 0):
                keyframes.append({'kps':kps,'descs':desc,'depth':depth.copy(),
                                  'pose':pose.copy(),'frame_idx':i})

            # Loop closure
            if i % LOOP_CHECK_INTERVAL == 0 and len(keyframes) > 3:
                cands = [(ki,kf) for ki,kf in enumerate(keyframes)
                         if i - kf['frame_idx'] >= LOOP_MIN_FRAME_GAP]
                # Sample for speed but keep all if few
                if len(cands) > 15:
                    cands = cands[::max(1, len(cands)//15)]

                best_kf, best_inl, best_dT = None, 0, None
                for ki, kf in cands:
                    kms = match_features(kf['descs'], desc, ratio=0.80)
                    if len(kms) < 20: continue
                    kR, ktv, kninl = estimate_pnp(kf['kps'], kps, kms, kf['depth'])
                    if kR is not None and kninl > best_inl and kninl >= LOOP_MIN_INLIERS:
                        kdT = np.eye(4); kdT[:3,:3]=kR; kdT[:3,3]=ktv.flatten()
                        best_kf, best_inl, best_dT = ki, kninl, kdT

                if best_kf is not None:
                    corr = keyframes[best_kf]['pose'] @ np.linalg.inv(best_dT)
                    drift = corr[:3,3] - pose[:3,3]
                    dn = np.linalg.norm(drift)
                    if 0.01 < dn < LOOP_MAX_DRIFT:
                        cur_ki = len(keyframes) - 1
                        n_loop = cur_ki - best_kf
                        if n_loop > 0:
                            for j in range(best_kf+1, len(keyframes)):
                                a = min((j-best_kf)/n_loop, 1.0)
                                keyframes[j]['pose'][:3,3] += a * drift
                        pose[:3,3] = corr[:3,3]
                        pose[:3,:3] = corr[:3,:3]
                        if T_align is not None and initial_height is not None:
                            al = T_align @ pose; al[2,3] = initial_height
                            pose = np.linalg.inv(T_align) @ al
                        loop_count += 1
                        fg = i - keyframes[best_kf]['frame_idx']
                        print(f"  [LOOP] Frame {i}: kf {best_kf} "
                              f"(gap={fg}f, inliers={best_inl}, drift={dn:.3f}m)")

        if T_align is None and f['gt_pose'] is not None:
            T_align = f['gt_pose'] @ np.linalg.inv(pose)
            initial_height = f['gt_pose'][2, 3]

        if f['gt_pose'] is not None:
            ap = (T_align @ pose) if T_align is not None else pose
            vo_trajectory.append(ap[:3,3].copy())
            gt_trajectory.append(f['gt_pose'][:3,3].copy())

        prev_kps, prev_desc, prev_depth = kps, desc, depth
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(frames)}")

    print(f"\n  ok={vo_ok} fail={vo_fail} loops={loop_count} kfs={len(keyframes)}")

    vo_pos = np.array(vo_trajectory); gt_pos = np.array(gt_trajectory)
    errors = np.linalg.norm(vo_pos - gt_pos, axis=1)
    rmse = np.sqrt(np.mean(errors**2))
    print(f"\n  ATE: RMSE={rmse:.4f}m  mean={errors.mean():.4f}m  max={errors.max():.4f}m")
    vc=vo_pos-vo_pos.mean(0); gc=gt_pos-gt_pos.mean(0)
    vs=np.sqrt(np.mean(np.sum(vc**2,1))); gs=np.sqrt(np.mean(np.sum(gc**2,1)))
    print(f"  Scale: VO={vs:.3f} GT={gs:.3f} ratio={vs/gs:.3f}")
    n=len(errors); seg=n//5
    for s in range(5):
        se=errors[s*seg:(s+1)*seg]
        print(f"  Seg {s+1}: {np.sqrt(np.mean(se**2)):.4f}m")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].plot(gt_pos[:,0], gt_pos[:,1], 'g-', label='GT', linewidth=2)
        axes[0].plot(vo_pos[:,0], vo_pos[:,1], 'b-', label='VO', linewidth=1, alpha=0.7)
        axes[0].legend(); axes[0].set_title('XY Trajectory'); axes[0].axis('equal')
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
        axes[1].plot(errors,'r-',linewidth=0.5)
        axes[1].set_title(f'ATE (RMSE={rmse:.3f}m)')
        axes[1].set_xlabel('Frame'); axes[1].set_ylabel('Error (m)')
        axes[2].plot(gt_pos[:,0], gt_pos[:,2], 'g-', label='GT', linewidth=2)
        axes[2].plot(vo_pos[:,0], vo_pos[:,2], 'b-', label='VO', linewidth=1, alpha=0.7)
        axes[2].legend(); axes[2].set_title('XZ Trajectory'); axes[2].axis('equal')
        axes[2].set_xlabel('X'); axes[2].set_ylabel('Z')
        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagnosis_loop_closure.png')
        plt.savefig(out, dpi=150); print(f"\n  Saved: {out}")
    except ImportError: pass

if __name__ == '__main__':
    main()
