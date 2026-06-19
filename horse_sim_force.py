import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# --- 1. Parameters ---
N = 18              # 馬の数
DT = 0.05           # 時間刻み (s)
TOTAL_DISTANCE = 2400 # レース距離 (m)
# シミュレーションステップ数 (目安: 13m/sで走破する時間 + 500ステップの余裕)
STEPS = int((TOTAL_DISTANCE / 13.0) / DT) + 500 

# Course Design (東京競馬場 芝2400mのレイアウトを再現)
# Aコース1周距離: 2083.1m, ゴール前直線距離: 525.9m
STRAIGHT_LEN = 600.0    # 直線部分の長さ (アニメーション上のオーバーラン確保のため延長)
GOAL_OFFSET = 74.1      # ゴール板の位置を直線の終端から手前にずらす距離 (600 - 525.9 = 74.1m)

# 1周を2083.1mにするためのコーナー計算: 2*STRAIGHT_LEN + 2*np.pi*R = 2083.1
CORNER_RADIUS = (2083.1 - 2 * STRAIGHT_LEN) / (2 * np.pi) 
CORNER_LEN = np.pi * CORNER_RADIUS # コーナーの周長 (半円)

# Track
TRACK_WIDTH = 30.0      # コース幅

# Physics
TAU = 0.5               # 推進力の応答時間 (小さいほど反応が速い)
GAMMA_BASE = 15.0       # 空気抵抗係数 (ベース)
RADIUS = 0.8            # 馬の半径 (衝突判定用)
REPULSION_K = 20000.0   # 衝突反発係数
GRAVITY = 9.8           # 重力加速度

# コーナリング摩擦係数（限界速度算出に使用）
MU = 0.9

# ラストの失速係数
FATIGUE_RATE = 0.005    

# 坂道の影響
SLOPE_EFFECT_FACTOR = 100.0 

# Drafting (風よけ)
ALPHA = 0.5             # ドラフティングによる抵抗減少率
WAKE_DIST = 8.0         # ドラフティングが発生する縦方向の距離
WAKE_ANGLE = np.pi/6    # ドラフティングが発生する横方向の角度 (30度)
SAVING_COEFF = 0.001    # ドラフティングによるエネルギー貯蓄効率

# Strategy (進路取り)
RAIL_STRENGTH = 5.0     # ラチ沿いを走ろうとする力の強さ
LANE_WIDTH = 1.5        # 他の馬を避ける際の目標レーン幅

# Pacing (基本となる目標速度 - ここから個体差が生まれる)
SPEED_START = 16.1      # スタート直後の目標速度 (m/s)
SPEED_CRUISE = 17.0     # 巡航時の目標速度 (m/s)
SPEED_SPURT = 18.0      # スパート時の目標速度 (m/s)

# Sections (レース区間)
DIST_CRUISE_START = 300.0        # 巡航速度に移行する距離

# 動的スパート戦略のための定数
MIN_SPURT_ENERGY = 0.0          # スパート開始に必要な最低限の貯蓄エネルギー
SPURT_THRESHOLD_DIST = 700.0    # ゴール前700mを切るまではスパート禁止

# 逃げ馬の戦略用定数
LEAD_BUFFER_DIST = 2.5          # 後続との差がこれ以上開いたら息を入れ始める距離 (m)
SPEED_PACE_DOWN = 16.5          # 息を入れる際の目標速度 (m/s) 

# 捲り(まくり)進出の戦略用定数
MAKURI_START_DIST = 600.0       # この距離を通過して以降、捲り進出を許可する (m)
MAKURI_FORWARD_GAP = 6.0        # 前の馬とこれ以上間隔が空いていれば前が開いていると判定 (m)
MAKURI_SIDE_CLEAR = 2.0         # 外側のこの横幅以内に馬がいなければ外に出せると判定 (m)
MAKURI_SIDE_RANGE = 8.0         # 外側の馬を確認する前後方向の範囲 (m)
MAKURI_SPEED = 17.5             # 捲り進出中の目標速度 (m/s)
MAKURI_LATERAL_OFFSET = 2.5     # 捲り開始時に外側へ持ち出す固定幅 (m)


class RaceSimulation:
    def __init__(self, n_particles):
        self.n = n_particles
        self.lap_length = 2 * STRAIGHT_LEN + 2 * CORNER_LEN
        
        self.pos = np.zeros((self.n, 2))
        
        # スタート位置の計算
        goal_s_lap = 2*STRAIGHT_LEN + CORNER_LEN - GOAL_OFFSET
        start_s = (goal_s_lap - TOTAL_DISTANCE) % self.lap_length
        self.start_s_log = start_s
        
        # 初期位置 (縦軸は均等に配置)
        self.pos[:, 0] = start_s
        self.pos[:, 1] = np.linspace(1.5, min(1.5 + self.n * 1.0, TRACK_WIDTH - 2.0), self.n)
        
        self.vel = np.zeros((self.n, 2))
        self.vel[:, 0] = 1.0 
        self.saved_energy = np.zeros(self.n) # ドラフティングなどで貯まるエネルギー
        
        # スタートダッシュの個体差 (倍)
        self.start_dash = np.random.uniform(0.95, 1.05, self.n)

        # 目標速度をランダムに初期化 (離散値を使用)
        discrete_offsets = np.linspace(0, 0.1, 50) 

        self.speed_start = SPEED_START + np.random.choice(discrete_offsets, self.n)
        self.speed_cruise = SPEED_CRUISE + np.random.choice(discrete_offsets, self.n)
        self.speed_spurt = SPEED_SPURT + np.random.choice(discrete_offsets, self.n)

        # 各馬の状態管理変数
        self.is_sprinting = np.zeros(self.n, dtype=bool) # 現在スパート中か
        self.spurt_start_dist = np.zeros(self.n) # スパートを開始した距離
        self.is_makuri = np.zeros(self.n, dtype=bool) # 現在捲り進出中か
        self.makuri_target_lat = np.zeros(self.n) # 捲り進出時に固定する目標横位置
        
        # --- 結果集計用の変数を追加 ---
        self.finish_times = [None] * self.n       # ゴールタイム
        self.last3f_start_times = [None] * self.n # 残り600m通過タイム
        self.last3f_times = [None] * self.n       # 上り3Fタイム (ゴール - 残り600m)
        
        # 各コーナー通過順位の記録用変数を追加
        self.corner_orders = {1: None, 2: None, 3: None, 4: None}
        self.corner_passed = {1: False, 2: False, 3: False, 4: False}
        
        # 各コーナーの規定位置（走行距離ベース）を算出
        c1_s = 2 * STRAIGHT_LEN + CORNER_LEN
        c2_s = 2 * STRAIGHT_LEN + CORNER_LEN + CORNER_LEN / 2.0
        c3_s = STRAIGHT_LEN
        c4_s = STRAIGHT_LEN + CORNER_LEN / 2.0
        
        self.trigger_c1 = (c1_s - self.start_s_log) % self.lap_length
        self.trigger_c2 = (c2_s - self.start_s_log) % self.lap_length
        self.trigger_c3 = (c3_s - self.start_s_log) % self.lap_length
        self.trigger_c4 = (c4_s - self.start_s_log) % self.lap_length
        
        # 1周以上走るため、3角と4角の判定距離を2周目に補正
        if self.trigger_c3 < self.trigger_c2:
            self.trigger_c3 += self.lap_length
        if self.trigger_c4 < self.trigger_c3:
            self.trigger_c4 += self.lap_length
        
        # 表示用の仮の馬名リスト (1番〜N番)
        self.horse_names = [f"Horse-{i+1:02d}" for i in range(self.n)]

        # 表示用の馬体重 (450kg〜550kg、1kg刻み)
        self.horse_weights = np.random.randint(450, 500, self.n)

        # 色の設定
        colors_palette = ['white', 'black', 'red', 'blue', 'yellow', 'green', 'orange', 'pink']
        self.colors = []
        counts = [self.n // 8] * 8 
        for r in range(self.n % 8): counts[7 - r] += 1
        for i, count in enumerate(counts):
            for _ in range(count): self.colors.append(colors_palette[i])

        self.current_time = 0.0
        self.next_furlong = 1        # 次に通過するハロン標 (200m毎)
        self.last_furlong_time = 0.0 # 最後にハロン標を通過した時刻
        
        print(f"=== RACE CONFIG: Tokyo 2400m Layout (N={N}) ===")

    def get_track_slope(self, s):
        """コースの縦軸位置sにおける勾配を返す"""
        goal_s = 2 * STRAIGHT_LEN + CORNER_LEN - GOAL_OFFSET
        
        # 現在位置からゴールまでの相対距離を逆算 (0 〜 lap_length)
        s_mod = s % self.lap_length
        dist_to_goal = (goal_s - s_mod) % self.lap_length
        
        # 1. ホームストレッチの坂 (ゴール手前460mから300mにかけて高低差2.0mの上り)
        if 300.0 < dist_to_goal < 460.0:
            return 0.0125
            
        # 2. バックストレッチの起伏 (概算)
        if 1200.0 < dist_to_goal < 1400.0:
            return -0.005
            
        # 3コーナー手前(ゴールから800m〜1000m手前)で緩やかな上り
        if 800.0 < dist_to_goal < 1000.0:
            return 0.005
            
        return 0.0

    def get_track_curvature(self, s):
        """コースの縦軸位置sにおける曲率を返す"""
        pos_in_lap = s % self.lap_length
        if pos_in_lap < STRAIGHT_LEN: return 0.0 
        elif pos_in_lap < STRAIGHT_LEN + CORNER_LEN: return 1.0 / CORNER_RADIUS 
        elif pos_in_lap < 2*STRAIGHT_LEN + CORNER_LEN: return 0.0 
        else: return 1.0 / CORNER_RADIUS 

    def get_curve_speed_limit(self, curvature):
        """曲率からコーナリング限界速度 v = sqrt(mu * g * r) を返す。"""
        if curvature <= 0:
            return np.inf
        radius = 1.0 / curvature
        return np.sqrt(MU * GRAVITY * radius)

    def get_drag_coeff(self, i):
        """馬 i の空気抵抗係数を計算する (ドラフティング判定を含む)"""
        gamma = GAMMA_BASE
        is_drafting = False
        for j in range(self.n):
            if i == j: continue
            ds = self.pos[j, 0] - self.pos[i, 0] # 前後方向の距離
            dn = self.pos[j, 1] - self.pos[i, 1] # 横方向の距離
            dist = np.sqrt(ds**2 + dn**2)
            if 0 < ds < WAKE_DIST:
                # 追突を防ぐため、真後ろ (ds>0) の判定のみを行う
                angle = np.arctan2(np.abs(dn), ds)
                if angle < WAKE_ANGLE:
                    # ドラフティング効果を適用
                    gamma = min(gamma, GAMMA_BASE * (1.0 - ALPHA))
                    is_drafting = True
        return gamma, is_drafting

    def update(self):
        self.current_time += DT
        forces = np.zeros((self.n, 2))
        
        # レース全体の最大進行距離を事前に計算
        max_pos = np.max(self.pos[:, 0])
        lead_dist_now = max_pos - self.start_s_log
        
        # コーナー通過順位の判定と記録
        triggers = {1: self.trigger_c1, 2: self.trigger_c2, 3: self.trigger_c3, 4: self.trigger_c4}
        for c in range(1, 5):
            if not self.corner_passed[c] and lead_dist_now >= triggers[c]:
                self.corner_passed[c] = True
                ranks = np.zeros(self.n, dtype=int)
                sorted_indices = np.argsort(self.pos[:, 0])[::-1]
                for rank, idx in enumerate(sorted_indices):
                    ranks[idx] = rank + 1
                self.corner_orders[c] = ranks

        for i in range(self.n):
            mass_i = self.horse_weights[i] # 各馬固有の質量（馬体重）を物理演算に適用
            s_now = self.pos[i, 0]
            dist_run = s_now - self.start_s_log # スタートからの走行距離
            n_lat = self.pos[i, 1] # 横軸位置
            
            # --- 計測ロジック (上り3F & ゴール) ---
            if dist_run >= TOTAL_DISTANCE - 600.0 and self.last3f_start_times[i] is None:
                self.last3f_start_times[i] = self.current_time
            
            if dist_run >= TOTAL_DISTANCE and self.finish_times[i] is None:
                self.finish_times[i] = self.current_time
                if self.last3f_start_times[i] is not None:
                    self.last3f_times[i] = self.finish_times[i] - self.last3f_start_times[i]
            # ------------------------------------

            curvature = self.get_track_curvature(s_now)
            is_corner = (curvature > 0)
            slope = self.get_track_slope(s_now)
            
            # ドラフティング判定と空気抵抗
            gamma_i, is_drafting = self.get_drag_coeff(i)
            f_drag_s = -gamma_i * self.vel[i, 0]
            f_drag_n = -gamma_i * self.vel[i, 1]
            
            # --- スパート戦略: 動的な開始判定 ---
            if is_drafting and dist_run > DIST_CRUISE_START and not self.is_sprinting[i]:
                self.saved_energy[i] += 1.0
            
            if not self.is_sprinting[i] and dist_run > DIST_CRUISE_START:
                leaders_ds = TOTAL_DISTANCE * 2 
                for j in range(self.n):
                    if i == j: continue
                    if self.pos[j, 0] > self.pos[i, 0] and abs(self.pos[j, 1] - self.pos[i, 1]) < 1.2: 
                        leaders_ds = min(leaders_ds, self.pos[j, 0] - self.pos[i, 0])

                remaining_dist = TOTAL_DISTANCE - dist_run
                can_spurt_by_distance = (remaining_dist < SPURT_THRESHOLD_DIST)
                force_spurt = (remaining_dist < 600.0) and (leaders_ds > WAKE_DIST)

                if (self.saved_energy[i] > MIN_SPURT_ENERGY and leaders_ds > WAKE_DIST and can_spurt_by_distance) or force_spurt:
                    self.is_sprinting[i] = True
                    self.spurt_start_dist[i] = dist_run
                    self.is_makuri[i] = False # スパートに移行したら捲り状態は解除する

            # --- 捲り進出の判定 ---
            # 600m通過以降、スパート開始前の馬を対象に、外にスペースがあり前が開いていれば
            # 先頭目掛けて進出を開始する。先頭に立った後の速度はスパート開始まで抑制される。
            if not self.is_sprinting[i] and not self.is_makuri[i] and dist_run > MAKURI_START_DIST:
                # 前方の同レーン付近の馬との間隔を確認
                forward_gap = TOTAL_DISTANCE * 2
                for j in range(self.n):
                    if i == j: continue
                    ds_j = self.pos[j, 0] - self.pos[i, 0]
                    if ds_j > 0 and abs(self.pos[j, 1] - self.pos[i, 1]) < 1.2:
                        forward_gap = min(forward_gap, ds_j)
                front_is_open = (forward_gap > MAKURI_FORWARD_GAP)

                # 外側(ラチから遠い側)に出すスペースがあるか確認
                side_is_clear = True
                for j in range(self.n):
                    if i == j: continue
                    ds_j = self.pos[j, 0] - self.pos[i, 0]
                    dn_j = self.pos[j, 1] - self.pos[i, 1]
                    if abs(ds_j) < MAKURI_SIDE_RANGE and 0 < dn_j < MAKURI_SIDE_CLEAR:
                        side_is_clear = False
                        break

                # 自分が先頭でない場合のみ捲り進出の対象とする
                is_not_leader = (max_pos - s_now) > 1.0

                if front_is_open and side_is_clear and is_not_leader:
                    self.is_makuri[i] = True
                    # 目標横位置をこの時点で一度だけ固定する (毎フレーム再計算しない)
                    fixed_target = n_lat + MAKURI_LATERAL_OFFSET
                    self.makuri_target_lat[i] = min(fixed_target, TRACK_WIDTH - RADIUS)

            # --- 目標速度 v_target の決定 ---
            if dist_run < DIST_CRUISE_START:
                v_target = self.speed_start[i] * self.start_dash[i]
            elif not self.is_sprinting[i]:
                v_target = self.speed_cruise[i]

                # 捲り進出中は巡航より高い速度で前へ出ようとする
                if self.is_makuri[i]:
                    v_target = MAKURI_SPEED

                min_lead_distance = TOTAL_DISTANCE * 2
                rear_positions = self.pos[self.pos[:, 0] < s_now, 0]
                if len(rear_positions) > 0:
                    min_lead_distance = s_now - np.max(rear_positions)
                
                forward_mask = self.pos[:, 0] > s_now
                if np.any(forward_mask):
                    forward_pos = self.pos[forward_mask]
                    forward_vel = self.vel[forward_mask]
                    ds_array = forward_pos[:, 0] - s_now
                    dn_array = forward_pos[:, 1] - n_lat
                    close_leader_mask = (ds_array < WAKE_DIST) & (np.abs(dn_array) < 1.2)
                    
                    if np.any(close_leader_mask):
                        closest_leader_idx = np.argmin(ds_array[close_leader_mask])
                        closest_leader_speed = forward_vel[close_leader_mask][closest_leader_idx, 0]
                        v_target = min(v_target, closest_leader_speed + 0.5) 

                # 先頭に立っており後続を離している場合は、スパート開始前まで速度を抑える。
                # 通常の巡航馬に加え、捲りで先頭に立った馬にも同じ抑制を適用する。
                is_leader = (max_pos - s_now) < 1.0 
                if is_leader and min_lead_distance > LEAD_BUFFER_DIST:
                    v_target = SPEED_PACE_DOWN
                    if self.is_makuri[i]:
                        self.is_makuri[i] = False
                        
            else:
                bonus = min(self.saved_energy[i] * SAVING_COEFF, 3.0)
                peak_speed = self.speed_spurt[i] + bonus
                v_target = peak_speed
                
                if dist_run > TOTAL_DISTANCE - 350.0: 
                    dist_over_peak = dist_run - (TOTAL_DISTANCE - 350.0)
                    stamina_contribution = self.saved_energy[i] * 0.002
                    stamina_factor = 1.0 + stamina_contribution
                    deceleration = (dist_over_peak * FATIGUE_RATE) / stamina_factor
                    v_target = peak_speed - deceleration
                    self.saved_energy[i] = max(0.0, self.saved_energy[i] - 5.0)

            slope_correction = slope * SLOPE_EFFECT_FACTOR
            if not self.is_sprinting[i] and slope < 0:
                slope_correction *= 0.1 
            v_target -= slope_correction

            v_max_curve = self.get_curve_speed_limit(curvature)
            v_target = min(v_target, v_max_curve)

            # Forces (推進力, 抗力, 遠心力, 重力)
            v_s = self.vel[i, 0]
            f_prop_s = mass_i * (v_target - v_s) / TAU
            f_centrifugal = 0.0
            if is_corner: 
                f_centrifugal = mass_i * (v_s**2) * curvature
            f_gravity = -mass_i * GRAVITY * slope 
            
            forces[i, 0] = f_prop_s + f_drag_s + f_gravity
            forces[i, 1] = f_centrifugal + f_drag_n

            # Collision (馬同士の衝突排除)
            for j in range(self.n):
                if i == j: continue
                ds = self.pos[i, 0] - self.pos[j, 0]
                dn = self.pos[i, 1] - self.pos[j, 1]
                dist = np.sqrt(ds**2 + dn**2)
                min_dist = 2 * RADIUS
                if dist < min_dist:
                    dist = max(dist, 0.01)
                    overlap = min_dist - dist
                    force_mag = REPULSION_K * overlap * (1.0 + 1.0/dist)
                    forces[i, 0] += force_mag * (ds / dist)
                    forces[i, 1] += force_mag * (dn / dist)
            
            # --- Lane Keeping & Overtaking ---
            target_n = RADIUS 
            is_blocked = False

            for j in range(self.n):
                if i == j: continue
                ds = self.pos[j, 0] - self.pos[i, 0]
                dn = self.pos[j, 1] - self.pos[i, 1]
                if -1.0 < ds < 4.0: 
                    if dn < 0: 
                        neighbor_limit = self.pos[j, 1] + LANE_WIDTH
                        if neighbor_limit > target_n: target_n = neighbor_limit
            
            if self.is_sprinting[i]:
                for j in range(self.n):
                    if i == j: continue
                    ds = self.pos[j, 0] - self.pos[i, 0]
                    if 0 < ds < 10.0:
                        if abs(self.pos[j, 1] - self.pos[i, 1]) < 1.2:
                            is_blocked = True
                            if self.vel[i, 0] > self.vel[j, 0] - 1.0: 
                                avoid_target = self.pos[j, 1] + 1.8 
                                if avoid_target > target_n:
                                    target_n = avoid_target

            # 捲り進出中は、開始時に固定した目標横位置まで持ち出す
            if self.is_makuri[i]:
                if self.makuri_target_lat[i] > target_n:
                    target_n = self.makuri_target_lat[i]

            v_n = self.vel[i, 1]
            damping = -20.0 * v_n * mass_i 
            dn_target = target_n - n_lat
            steer_gain = RAIL_STRENGTH * 3.0
            if is_blocked:
                 steer_gain *= 1.5 

            forces[i, 1] += steer_gain * mass_i * dn_target + damping
            
            if n_lat < RADIUS: forces[i, 1] += 20000.0 * (RADIUS - n_lat)
            elif n_lat > TRACK_WIDTH - RADIUS: forces[i, 1] -= 20000.0 * (n_lat - (TRACK_WIDTH - RADIUS))

            forces[i] += np.random.normal(0, 15.0, 2)

        # 運動方程式の適用 (各馬固有の体重配列で除算)
        acc = forces / self.horse_weights[:, None]
        self.vel += acc * DT
        
        # 位置の更新
        for i in range(self.n):
            s = self.pos[i, 0]
            n_lat = self.pos[i, 1]
            curvature = self.get_track_curvature(s)
            scale_factor = CORNER_RADIUS / (CORNER_RADIUS + n_lat) if curvature > 0 else 1.0
            
            self.pos[i, 0] += self.vel[i, 0] * DT * scale_factor
            self.pos[i, 1] += self.vel[i, 1] * DT
            
            if self.pos[i, 1] < RADIUS: self.pos[i, 1] = RADIUS
            if self.pos[i, 1] > TRACK_WIDTH - RADIUS: self.pos[i, 1] = TRACK_WIDTH - RADIUS

        # 経過表示
        lead_dist = np.max(self.pos[:, 0]) - self.start_s_log
        target_dist = self.next_furlong * 200.0
        if lead_dist >= target_dist:
            lap_time = self.current_time - self.last_furlong_time
            print(f"{self.next_furlong}F ({int(target_dist)}m): {self.current_time:.1f}s  (Lap: {lap_time:.1f}s)")
            self.last_furlong_time = self.current_time
            self.next_furlong += 1

    def map_to_global(self, s, n):
        """縦軸sと横軸nのローカル座標を、アニメーション用のグローバル座標(x, y)に変換する"""
        s_mod = s % self.lap_length
        L = STRAIGHT_LEN
        R = CORNER_RADIUS
        
        if s_mod < L:
            ratio = s_mod / L
            return (L/2.0) - (ratio * L), R + n
        elif s_mod < L + CORNER_LEN:
            arc_dist = s_mod - L
            angle = (np.pi / 2.0) + (arc_dist / R)
            return -L/2.0 + (R + n) * np.cos(angle), (R + n) * np.sin(angle)
        elif s_mod < 2*L + CORNER_LEN:
            dist_on_straight = s_mod - (L + CORNER_LEN)
            return (-L/2.0) + dist_on_straight, -(R + n)
        else:
            arc_dist = s_mod - (2*L + CORNER_LEN)
            angle = (3.0 * np.pi / 2.0) + (arc_dist / R)
            return L/2.0 + (R + n) * np.cos(angle), (R + n) * np.sin(angle)

    def get_all_global_pos(self):
        """全馬のローカル座標をグローバル座標に変換して返す"""
        global_pos = np.zeros((self.n, 2))
        for i in range(self.n):
            global_pos[i] = self.map_to_global(self.pos[i, 0], self.pos[i, 1])
        return global_pos
    
    def print_results(self):
        """レース結果を表形式で出力する"""
        print("\n" + "="*85)
        print(f"   RACE RESULTS ({TOTAL_DISTANCE}m)")
        print("="*85)
        print(f"{'Rk':<4} {'No.':<4} {'Horse Name':<15} {'Weight':<8} {'Time':<10} {'Margin':<8} {'Corner':<15} {'Last 3F':<8}")
        print("-" * 85)
        
        results = []
        for i in range(self.n):
            t_finish = self.finish_times[i] if self.finish_times[i] is not None else 9999.0
            t_3f = self.last3f_times[i] if self.last3f_times[i] is not None else 0.0
            
            c1 = self.corner_orders[1][i] if self.corner_passed[1] else "-"
            c2 = self.corner_orders[2][i] if self.corner_passed[2] else "-"
            c3 = self.corner_orders[3][i] if self.corner_passed[3] else "-"
            c4 = self.corner_orders[4][i] if self.corner_passed[4] else "-"
            corner_str = f"{c1}-{c2}-{c3}-{c4}"
            
            results.append({
                "id": i,
                "name": self.horse_names[i],
                "weight": self.horse_weights[i], # 各馬の体重を結果用データに追加
                "time": t_finish,
                "corner_str": corner_str,
                "last3f": t_3f
            })
        
        results.sort(key=lambda x: x["time"])
        winner_time = results[0]["time"]
        
        for rank, r in enumerate(results):
            m = int(r["time"] // 60)
            s = r["time"] % 60
            time_str = f"{m}:{s:04.1f}"
            
            if rank == 0:
                margin_str = "-"
            else:
                diff = r["time"] - winner_time
                margin_str = f"+{diff:.1f}"
                
            last3f_str = f"{r['last3f']:.1f}"
            horse_num = r["id"] + 1
            
            # Weight列を含めて結果を出力（構文エラーを修正した箇所）
            weight_str = f"{r['weight']}kg"
            print(f"{rank+1:<4} {horse_num:<4} {r['name']:<15} {weight_str:<8} {time_str:<10} {margin_str:<8} {r['corner_str']:<15} {last3f_str:<8}")
        print("="*85 + "\n")

# --- Animation Setup ---
sim = RaceSimulation(N)

# 1. Preview (コース勾配の事前表示)
def show_preview(sim_instance):
    fig_p, ax_p = plt.subplots(figsize=(12, 6))
    ax_p.set_aspect('equal')
    
    step = 2.0
    lap_len = sim_instance.lap_length
    s_vals = np.arange(0, lap_len + step, step)
    
    points = np.array([sim_instance.map_to_global(s, 0) for s in s_vals]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    slopes = np.array([sim_instance.get_track_slope(s) for s in s_vals[:-1]])
    
    norm = plt.Normalize(-0.02, 0.02)
    lc = LineCollection(segments, cmap='coolwarm', norm=norm)
    lc.set_array(slopes)
    lc.set_linewidth(5)
    ax_p.add_collection(lc)
    
    outer = np.array([sim_instance.map_to_global(s, TRACK_WIDTH) for s in s_vals])
    ax_p.plot(outer[:,0], outer[:,1], 'gray', lw=1)
    
    s_start = sim_instance.start_s_log
    st_in = sim_instance.map_to_global(s_start, 0)
    st_out = sim_instance.map_to_global(s_start, TRACK_WIDTH)
    ax_p.plot([st_in[0], st_out[0]], [st_in[1], st_out[1]], 'b--', lw=2)
    ax_p.text(st_out[0], st_out[1]+10, "START", color='blue', ha='center')
    
    g_s = 2*STRAIGHT_LEN + CORNER_LEN - GOAL_OFFSET
    g_in = sim_instance.map_to_global(g_s, 0)
    g_out = sim_instance.map_to_global(g_s, TRACK_WIDTH)
    ax_p.plot([g_in[0], g_out[0]], [g_in[1], g_out[1]], 'k-', lw=3)
    ax_p.text(g_out[0], g_out[1]-10, "GOAL", color='black', ha='center')
    
    fig_p.colorbar(lc, ax=ax_p, label='Slope (Red=Up)')
    ax_p.set_xlim(-450, 450)
    ax_p.set_ylim(-200, 200)
    ax_p.set_title("Course Slope Preview (Red=Up, Blue=Down)")
    plt.show()

show_preview(sim)

# 2. Animation (レースシミュレーション)
fig, ax = plt.subplots(figsize=(12, 7)) 
ax.set_aspect('equal')

def plot_track_bg(ax_target):
    step = 10.0
    lap_len = sim.lap_length
    s_vals = np.arange(0, lap_len+step, step)
    inner = np.array([sim.map_to_global(s, 0) for s in s_vals])
    outer = np.array([sim.map_to_global(s, TRACK_WIDTH) for s in s_vals])
    ax_target.plot(inner[:,0], inner[:,1], color='green', alpha=0.6, lw=2)
    ax_target.plot(outer[:,0], outer[:,1], color='gray', alpha=0.6, lw=2)
    g_s = 2*STRAIGHT_LEN + CORNER_LEN - GOAL_OFFSET
    g_in = sim.map_to_global(g_s, 0)
    g_out = sim.map_to_global(g_s, TRACK_WIDTH)
    ax_target.plot([g_in[0], g_out[0]], [g_in[1], g_out[1]], 'k-', lw=3)

plot_track_bg(ax)

initial_pos = sim.get_all_global_pos()
scat = ax.scatter(initial_pos[:, 0], initial_pos[:, 1], s=30, c=sim.colors, edgecolors='black', linewidth=0.5, zorder=10)
time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes, fontsize=12)
slope_text = ax.text(0.02, 0.85, '', transform=ax.transAxes, fontsize=10)

def animate(frame, ax, sim, scat, time_text, slope_text):
    sim.update()
    gpos = sim.get_all_global_pos()
    scat.set_offsets(gpos)
    
    center_x = np.mean(gpos[:, 0])
    center_y = np.mean(gpos[:, 1])
    zoom = 60.0 
    ax.set_xlim(center_x - zoom, center_x + zoom)
    ax.set_ylim(center_y - zoom, center_y + zoom)
    
    time_text.set_text(f'Time: {frame*DT:.1f} s')
    
    lead_idx = np.argmax(sim.pos[:, 0])
    current_slope = sim.get_track_slope(sim.pos[lead_idx, 0])
    if current_slope > 0.005:
        slope_text.set_text("UPHILL!")
        slope_text.set_color('red')
    elif current_slope < -0.005:
        slope_text.set_text("DOWNHILL!")
        slope_text.set_color('blue')
    else:
        slope_text.set_text("FLAT")
        slope_text.set_color('green')
    
    if all(t is not None for t in sim.finish_times):
        print("================ GOAL (ALL HORSES) ================")
        sim.print_results() 
        ani.event_source.stop()
        
    return scat, time_text, slope_text

ani = animation.FuncAnimation(fig, animate, fargs=(ax, sim, scat, time_text, slope_text),
                              frames=STEPS, interval=20, blit=False, repeat=False)
plt.title(f"Simulation: Dynamic Energy-Based Spurt with Distance Threshold (N={N})")
plt.grid(True, alpha=0.3)
plt.show()