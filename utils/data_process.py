import os
import numpy as np
import pickle
import re
from tqdm import tqdm
from collections import defaultdict
from config import cfg


# ==========================================
# 1. File Scanner
# ==========================================
class DatasetScanner:
    def __init__(self, data_root):
        self.data_root = data_root
        self.pattern = re.compile(r".*(Town\d+)_type(\d+)_subtype(\d+)_scenario(\d+).*")

    def scan_files(self):
        file_list = []
        if not os.path.exists(self.data_root):
            print(f"Error: Data root {self.data_root} not found.")
            return []

        print(f"Scanning directory: {self.data_root} ...")
        for root, dirs, files in os.walk(self.data_root):
            if "label" in root:
                # calib is needed to confirm frame existence, though we rely less on ego matrix now
                calib_dir = root.replace("label", "calib")
                for f in files:
                    if f.endswith('.txt'):
                        label_path = os.path.join(root, f)
                        # Try finding corresponding calib file just to be sure frame is valid
                        calib_name = f.replace('.txt', '.pkl')
                        calib_path = os.path.join(calib_dir, calib_name)

                        if os.path.exists(calib_path):
                            info = self._parse_info(label_path)
                            if info['town'] != 'Unknown':
                                file_list.append({
                                    'label': label_path,
                                    'calib': calib_path,
                                    'info': info,
                                    'timestamp': f  # Filename as timestamp
                                })
        print(f"Found {len(file_list)} valid frames.")
        return file_list

    def _parse_info(self, path):
        match = self.pattern.search(path)
        if match:
            return {
                'town': match.group(1),
                'type_id': int(match.group(2)),
                'subtype_id': int(match.group(3)),
                'scenario_id': int(match.group(4)),
                'is_accident': "accident" in path.lower()
            }
        return {'town': 'Unknown', 'is_accident': False}


# ==========================================
# 2. Map Processor (Agent-Centric)
# ==========================================
class MapProcessor:
    def __init__(self, map_root):
        self.map_root = map_root
        self.cache = {}

    def get_agent_centric_map(self, town_name, anchor_pose):
        """
        获取相对于 Agent 的局部地图。
        anchor_pose: (x, y, yaw) of the agent at T=0 in World Frame.
        """
        # 1. Load Raw Map (World Frame)
        if town_name not in self.cache:
            map_path = os.path.join(self.map_root, f"{town_name}.npz")
            try:
                if os.path.exists(map_path):
                    data = np.load(map_path)
                    if 'polylines' in data:
                        self.cache[town_name] = data['polylines']
                    else:
                        self.cache[town_name] = None
                else:
                    self.cache[town_name] = None
            except:
                self.cache[town_name] = None

        raw_map = self.cache[town_name]

        # Init Output
        output = np.zeros((cfg.MAP_MAX_LINES, cfg.MAP_POINTS_PER_LINE, cfg.MAP_DIM))
        if raw_map is None: return output

        # 2. Coordinate Transformation (World -> Agent-Centric)
        # Anchor
        ax, ay, ayaw = anchor_pose

        # Rotation Matrix elements for Inverse Rotation (rotating world to align with agent)
        # We want Agent Yaw -> 0 degrees (Positive X axis)
        c, s = np.cos(ayaw), np.sin(ayaw)

        # Prepare Map Points
        src_dim = raw_map.shape[-1]
        # Assume first 2 cols are x, y
        map_x = raw_map[..., 0]
        map_y = raw_map[..., 1]

        # Translate
        tx = map_x - ax
        ty = map_y - ay

        # Rotate: x' = x*c + y*s, y' = -x*s + y*c
        local_x = tx * c + ty * s
        local_y = -tx * s + ty * c

        # 3. Spatial Query (Filter lines far away)
        # Calculate center of each polyline in local frame
        line_centers_x = local_x.mean(axis=1)
        line_centers_y = local_y.mean(axis=1)
        dists = np.sqrt(line_centers_x ** 2 + line_centers_y ** 2)

        # Top K closest lines
        indices = np.argsort(dists)[:cfg.MAP_MAX_LINES]

        # Select features
        sel_local_x = local_x[indices]
        sel_local_y = local_y[indices]
        sel_raw = raw_map[indices]

        n_l = min(len(indices), cfg.MAP_MAX_LINES)
        n_p = min(sel_local_x.shape[1], cfg.MAP_POINTS_PER_LINE)

        # 4. Fill Output
        # [0:2] -> Local X, Local Y
        output[:n_l, :n_p, 0] = sel_local_x[:n_l, :n_p]
        output[:n_l, :n_p, 1] = sel_local_y[:n_l, :n_p]

        # [2] -> Local Z (if available, mostly 0 for HDMap)
        if src_dim >= 3:
            # Z usually doesn't change much with rotation, just translation if needed
            # Here we simplify and just copy raw Z if it exists, or 0
            output[:n_l, :n_p, 2] = sel_raw[:n_l, :n_p, 2]

        # [3:] -> Attributes (Type, etc.)
        # Source starts at 3 if it has Z, or 2 if only XY
        src_attr_start = 3 if src_dim >= 3 else 2
        tgt_attr_start = 3  # We filled x,y,z

        copy_len = min(cfg.MAP_DIM - tgt_attr_start, src_dim - src_attr_start)
        if copy_len > 0:
            output[:n_l, :n_p, tgt_attr_start:tgt_attr_start + copy_len] = \
                sel_raw[:n_l, :n_p, src_attr_start:src_attr_start + copy_len]

        return output


# ==========================================
# 3. Data Processor (Trajectory-First & Agent-Centric)
# ==========================================
class DataProcessor:
    def __init__(self):
        data_root = getattr(cfg, 'RAW_DATA_ROOT', './data')
        map_root = getattr(cfg, 'MAP_ROOT', './map_features')
        # Robust map path check
        if not os.path.exists(map_root):
            if os.path.exists('../map_features'): map_root = '../map_features'

        self.scanner = DatasetScanner(data_root)
        self.map_proc = MapProcessor(map_root)

        # For calculating global stats (though Agent-Centric is mostly normalized by design)
        self.scalers = {'pos_mean': 0, 'pos_std': 1, 'vel_mean': 0, 'vel_std': 1}

    def parse_label(self, path):
        """Parse one label file into a list of agent dicts."""
        agents = []
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 10: continue

                # Filter object type
                obj_type = parts[0].lower()
                valid = ['car', 'van', 'truck', 'vehicle']
                if not any(v in obj_type for v in valid): continue

                try:
                    info = {
                        'id': int(parts[10]),
                        'type': parts[0],
                        # Raw World Coordinates
                        'pos': np.array([float(parts[1]), float(parts[2]), float(parts[3])]),
                        'yaw': float(parts[7]),
                        'vel': np.array([float(parts[8]), float(parts[9])])
                    }
                    agents.append(info)
                except:
                    continue
        except:
            pass
        return agents

    def to_agent_centric(self, traj_world_pos, anchor_pose):
        """
        Batch Transform trajectory points to Agent-Centric Frame.

        Args:
            traj_world_pos: [T, 2] or [T, 3] World coordinates
            anchor_pose: (ax, ay, ayaw) Anchor state

        Returns:
            traj_local: [T, 2] Local coordinates
        """
        ax, ay, ayaw = anchor_pose
        c, s = np.cos(ayaw), np.sin(ayaw)

        # 1. Translate
        px = traj_world_pos[:, 0] - ax
        py = traj_world_pos[:, 1] - ay

        # 2. Rotate (Inverse)
        # We rotate the *world* by -ayaw so the agent faces 0
        new_x = px * c + py * s
        new_y = -px * s + py * c

        return np.stack([new_x, new_y], axis=-1)

    def run(self):
        all_files = self.scanner.scan_files()
        if not all_files: return

        # Group files by Scenario (Parent Directory)
        scenarios = defaultdict(list)
        for f in all_files:
            # Use parent dir as scenario key to group frames
            parent_dir = os.path.dirname(f['label'])
            scenarios[parent_dir].append(f)

        samples = []
        all_pos_local = []
        all_vel_local = []

        print(f"Processing {len(scenarios)} scenarios (Agent-Centric Mode)...")

        total_saved = 0

        for s_path, frames in tqdm(scenarios.items()):
            # Sort frames by time/filename
            frames.sort(key=lambda x: x['label'])

            if len(frames) < cfg.OBS_LEN + cfg.PRED_LEN: continue

            # [Step 1] Load entire scene into memory to minimize IO
            scene_data = []

            # Get scenario static info
            first_frame_info = frames[0]['info']
            town_name = first_frame_info['town']

            for frame in frames:
                agents = self.parse_label(frame['label'])
                scene_data.append({
                    'agents': agents,
                    'info': frame['info']  # meta info
                })

            # [Step 2] Build Tracks (Agent ID -> List of Data)
            agent_tracks = defaultdict(list)
            for idx, data in enumerate(scene_data):
                for ag in data['agents']:
                    # Store (Frame Index, Agent Data)
                    agent_tracks[ag['id']].append((idx, ag))

            window_size = cfg.OBS_LEN + cfg.PRED_LEN

            # [Step 3] Slice & Transform
            for aid, track in agent_tracks.items():
                track.sort(key=lambda x: x[0])
                if len(track) < window_size: continue

                # Iterate sliding windows on this specific agent's track
                for i in range(len(track) - window_size + 1):
                    segment = track[i: i + window_size]

                    # Continuity Check
                    if segment[-1][0] - segment[0][0] != window_size - 1:
                        continue

                    # Define "Current" Frame (T=0)
                    # Data: [Hist (0...29), Curr (29), Fut (30...49)]
                    curr_idx_in_seg = cfg.OBS_LEN - 1
                    curr_data = segment[curr_idx_in_seg][1]

                    # === ESTABLISH ANCHOR ===
                    # Anchor is the agent's state at T=0
                    ax, ay = curr_data['pos'][0], curr_data['pos'][1]
                    ayaw = curr_data['yaw']
                    anchor_pose = (ax, ay, ayaw)

                    # Extract World Trajectory for the whole window
                    # Shape: [Window_Size, 2]
                    traj_world = np.array([item[1]['pos'][:2] for item in segment])

                    # === TRANSFORM TO AGENT-CENTRIC ===
                    traj_local = self.to_agent_centric(traj_world, anchor_pose)

                    # Calculate Local Velocities / Accelerations
                    # v = (p_t - p_{t-1}) / dt
                    # Note: We rely on position differentiation to ensure kinematic consistency in local frame
                    dt = 0.1
                    vel_local = np.zeros_like(traj_local)
                    # Forward diff for 0 to N-1
                    vel_local[1:] = (traj_local[1:] - traj_local[:-1]) / dt
                    vel_local[0] = vel_local[1]  # Repeat first velocity

                    # Acc
                    acc_local = np.zeros_like(vel_local)
                    acc_local[1:] = (vel_local[1:] - vel_local[:-1]) / dt

                    # Split into History and Future
                    # Hist: [0, OBS_LEN)
                    hist_arr = np.zeros((cfg.OBS_LEN, cfg.INPUT_DIM))
                    # x, y
                    hist_arr[:, 0:2] = traj_local[:cfg.OBS_LEN]
                    # vx, vy
                    hist_arr[:, 2:4] = vel_local[:cfg.OBS_LEN]
                    # ax, ay
                    hist_arr[:, 4:6] = acc_local[:cfg.OBS_LEN]
                    # yaw (relative to anchor, so roughly 0, but we can compute heading from vel)
                    # For simplicity, we can set yaw to 0 or atan2(vy, vx)
                    # Since we rotated the world, the agent's yaw at T=0 is exactly 0.

                    # Fut: [OBS_LEN, End)
                    fut_arr = np.zeros((cfg.PRED_LEN, 4))
                    fut_arr[:, 0:2] = traj_local[cfg.OBS_LEN:]
                    fut_arr[:, 2:4] = vel_local[cfg.OBS_LEN:]

                    # Sanity Check: Outliers (Teleportation check)
                    # If velocity > 100 m/s (360 km/h), likely data noise
                    if np.max(np.abs(vel_local)) > 100: continue

                    # === GET MAP (Agent-Centric) ===
                    map_feat = self.map_proc.get_agent_centric_map(town_name, anchor_pose)

                    # Save
                    samples.append({
                        'hist': hist_arr,
                        'future': fut_arr,
                        'map': map_feat,
                        'meta': scene_data[0]['info'],
                        'agent_id': aid,
                        'scenario': s_path
                    })

                    all_pos_local.append(hist_arr[:, :2])
                    all_vel_local.append(hist_arr[:, 2:4])
                    total_saved += 1

        print(f"\n[Summary] Total Samples Saved: {total_saved}")

        if len(samples) == 0:
            print("❌ No samples. Check data paths.")
            return

        # Calculate Scalers (on Local Data)
        # Expectation: Mean should be very close to 0, Std should be small (e.g., 10-30m)
        if len(all_pos_local) > 0:
            all_pos_cat = np.concatenate(all_pos_local)
            all_vel_cat = np.concatenate(all_vel_local)

            self.scalers['pos_mean'] = np.mean(all_pos_cat, axis=0)
            self.scalers['pos_std'] = np.std(all_pos_cat, axis=0) + 1e-6
            self.scalers['vel_mean'] = np.mean(all_vel_cat, axis=0)
            self.scalers['vel_std'] = np.std(all_vel_cat, axis=0) + 1e-6

            print(f"Scalers Computed (Local Frame):")
            print(f"  Pos Mean: {self.scalers['pos_mean']} (Should be ~0)")
            print(f"  Pos Std:  {self.scalers['pos_std']}  (Should be small, e.g. <50)")

        os.makedirs(os.path.dirname(cfg.PROCESSED_DATA_PATH), exist_ok=True)
        with open(cfg.PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(samples, f)
        with open(cfg.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"Saved to {cfg.PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    processor = DataProcessor()
    processor.run()