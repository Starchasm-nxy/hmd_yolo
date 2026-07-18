# 聚类抗风扰 + 快速收敛 策略分析

当前问题：飞机悬停不稳定 → 历史离群点拉偏 DBSCAN 聚类中心 → 坐标不准。

---

## 策略 2：迭代聚类 + EWMA 中心平滑

### 原理

把"所有点攒够了再聚一次"改为"每隔一小段时间聚一次，中心指数平滑"。

```
时间轴: ────1s──── ────1s──── ────1s──── ────1s────
         聚#1       聚#2       聚#3       聚#4
           │          │          │          │
           ▼          ▼          ▼          ▼
        C1_raw    C2_raw     C3_raw     C4_raw
           │          │          │          │
           └──────────┼──────────┼──────────┘
                      ▼          ▼
              C1 = C1_raw    C2 = α*C2_raw + (1-α)*C1
                             C3 = α*C3_raw + (1-α)*C2
                             C4 = α*C4_raw + (1-α)*C3
                             
收敛判定: |C3 - C2| < ε 且 |C4 - C3| < ε → 提前结束
```

α 取 0.6~0.7（偏信新结果但保留历史平滑）。

### 伪代码

```python
smoothed_centers = None
last_cluster_time = time.time()
CLUSTER_INTERVAL = 1.0    # 每秒聚一次
ALPHA = 0.7               # 平滑系数
CONVERGE_THRESHOLD = 0.1  # 收敛阈值 (米)

for each frame:
    points.append((ux, uy, time.time()))
    
    if time.time() - last_cluster_time >= CLUSTER_INTERVAL and len(points) >= 5:
        recent = [(x,y) for x,y,t in points if t > time.time() - 5.0]
        centers = dbscan(recent)
        
        if smoothed_centers is None:
            smoothed_centers = centers
        else:
            # 匹配本轮中心到历史中心（按最近距离配对）
            matched = match_by_proximity(centers, smoothed_centers)
            smoothed_centers = [alpha * c + (1-alpha) * s 
                               for c, s in matched]
        
        # 收敛检查
        if max(|new - old| for new, old in zip(centers, prev_centers)) < CONVERGE_THRESHOLD:
            stable_count += 1
            if stable_count >= 2:
                break  # 收敛，提前结束
        else:
            stable_count = 0
        
        last_cluster_time = time.time()
```

### 优点
- 每 1s 看到中间结果，不会等到 5s 才发现偏了
- EWMA 自动压制单次极端值
- 可提前收敛（如果 2s 就稳了，不必等 5s）

### 缺点
- 需要**跨轮匹配聚类**（本轮 3 个聚类 vs 上轮 3 个聚类，哪个对应哪个？）
- 风刚停时前两轮可能聚出 2 个或 4 个，需要处理聚类数不一致
- 状态管理比一次性聚类复杂

---

## 策略 3：点级时间衰减加权

### 原理

不改变聚类触发时机，但在算聚类中心时，**越新的点权重越高**：

```
当前: center = mean(points)            ← 所有点平等

改为: weights = exp(-(now - t) / tau)  ← 新点权重大，旧点衰减
      center = sum(w_i * p_i) / sum(w_i)
```

权重随时间指数衰减：

```
权重
1.0 ┤***
    │   ***
0.5 ┤      ****
    │          ******
0.0 ┤               ********_________
    └─────┬─────┬─────┬─────┬─────┬──→ 点的年龄(s)
          0     1     2     3     4
                τ=1.0
```

### 伪代码

```python
TAU = 1.5               # 半衰期约 1s (ln2 * tau ≈ 1s)
now = time.time()

for each cluster:
    weights = [math.exp(-(now - t) / TAU) for (x,y,t) in cluster_points]
    total_w = sum(weights)
    cx = sum(w * x for w, (x,y,t) in zip(weights, cluster_points)) / total_w
    cy = sum(w * y for w, (x,y,t) in zip(weights, cluster_points)) / total_w
```

### 优点
- 改动极小：只在算 `cluster_center` 时加权重，不动 DBSCAN
- 不需要跨轮匹配聚类
- 天然压制早期离群点，青睐最近稳定点

### 缺点
- 不改变"一次性聚类"的整体流程——还是要等触发条件
- 如果最早期点形成了错误聚类，权重低但仍参与了 DBSCAN 分组

---

## 策略 9：median 替代 mean

改动半行，但有效：

```python
# 改前
cluster_center = np.mean(cluster_points, axis=0)

# 改后
cluster_center = np.median(cluster_points, axis=0)
```

中位数对离群点天然免疫——一个远处的噪声点完全不影响中位数，但会拉偏均值。

限制：只对单峰分布好，如果簇本身是偏的，median 可能偏。

---

## 策略 A：两阶段"锁-精"（新方案）

### 思路

模仿 LockTracker 的思路——先锁住大致区域，再精细收敛：

```
阶段1 (粗锁):  收集 10 个点 → 聚类 → 得到粗中心 (R1, R2, R3)
阶段2 (精调):  后续点只接受"靠近粗中心"的 → 过滤风扰点
              用 accepted 点算加权中心 → 每 0.5s 更新一次
              中心移动 < 0.05m → 收敛，提前结束
```

```
      阶段1 (粗锁)              阶段2 (精调)
  ───────────────────── ────────────────────────────
  攒够10个点→聚类        空间过滤→只接受近点→EWMA平滑→收敛
  (1-2s)                (1-2s, 可提前结束)
```

### 伪代码

```python
COARSE_MIN = 10          # 粗锁需要的最低点数
FINE_RADIUS = 0.3        # 精调阶段的空间过滤半径 (米)
FINE_INTERVAL = 0.5      # 精调阶段更新间隔 (秒)
FINE_CONVERGE = 0.05     # 收敛阈值 (米)

phase = "coarse"

for each frame:
    ux, uy = yolo_detect(frame)
    x_cam, y_cam = pixel_to_camera(ux, uy)
    t = time.time()
    
    if phase == "coarse":
        all_points.append((x_cam, y_cam, t))
        if len(all_points) >= COARSE_MIN:
            centers = dbscan_cluster(all_points)  # 粗聚类
            coarse_centers = centers
            phase = "fine"
            last_update = t
            smoothed = centers
    
    elif phase == "fine":
        # 空间过滤：只接受靠近粗中心的点
        for each center in coarse_centers:
            if distance((x_cam, y_cam), center) < FINE_RADIUS:
                fine_points[center_idx].append((x_cam, y_cam, t))
        
        # 定期更新
        if t - last_update >= FINE_INTERVAL:
            new_centers = [weighted_median(fine_points[i]) for i in range(3)]
            smoothed = [0.7 * n + 0.3 * s for n, s in zip(new_centers, smoothed)]
            
            # 收敛检查
            if max(|n - s| for n, s in zip(new_centers, smoothed)) < FINE_CONVERGE:
                converge_count += 1
                if converge_count >= 2:
                    finalize(smoothed)  # 收敛！
                    break
            
            last_update = t
```

### 优点
- **阶段 2 的空间过滤直接屏蔽风扰点**——漂太远的点不被接受
- 可以提前收敛（1s 粗锁 + 1s 精调 = 2s 搞定）
- 比迭代聚类更简单——不需要跨轮匹配聚类

### 缺点
- 如果粗锁阶段就偏了（10 个点中有 5 个噪声），后续全偏
- 需要 3 个桶全部粗锁成功

---

## 策略对比

| | 改动量 | 抗历史噪声 | 加速收敛 | 抗实时抖动 | 复杂度 |
|---|---|---|---|---|---|
| 策略1 (时间窗) | 小 | ✅ | — | — | 低 |
| 策略2 (迭代EWMA) | 大 | ✅ | ✅✅ | ✅ | 高 |
| 策略3 (衰减加权) | 中 | ✅✅ | — | ✅ | 低 |
| 策略9 (median) | 极小 | ✅ | — | ✅ | 极低 |
| 策略A (锁-精) | 大 | ✅✅ | ✅✅✅ | ✅✅ | 中 |

---

## 推荐组合

```
策略1 (时间窗) + 策略3 (衰减加权) + 策略9 (median)
  ├─ 时间窗滤掉 >5s 的旧点        ← 简单有效
  ├─ 指数衰减压制 1-5s 内的老点    ← 平滑过渡
  └─ median 替代 mean 免疫极端值   ← 零成本增强
```

三层防御：时间窗截断 → 衰减拉偏 → median 兜底。改动全在 `DBSCANClusterer.cluster()` 和 points 数据结构里，不影响管线逻辑。

要把这个组合方案画成代码改动计划吗？
