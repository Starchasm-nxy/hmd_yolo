# 方案A增强版：m=后桶 自动推断

## 核心规则（检测到共列时）

```
┌─────────────────────────────────────────────────────┐
│  如果两桶共列 →  m = 该列后桶 (Y大)                   │
│                 l/r = 该列前桶 (Y小)                  │
│  如果三列分开 →  l/m/r = 左/中/右 (不变)              │
└─────────────────────────────────────────────────────┘
```

## 推导逻辑（程序自动判断，不需要外部输入）

```python
# Step 1: 排序 + 检测共列
camera_centers.sort(key=lambda c: c[0])  # 按 X 排
groups = group_by_x(camera_centers, threshold=0.5)

# Step 2: 组内按 Y 排（前→后）
for g in groups:
    g.sort(key=lambda c: c[1])

# Step 3: 根据分组情况决定索引映射
if len(groups) == 3:
    # 3列分开 → 正常映射
    mapping = {'l': 0, 'm': 1, 'r': 2}
elif len(groups) == 2:
    big = groups[0] if len(groups[0]) == 2 else groups[1]  # 有2个桶的列
    solo = groups[0] if len(groups[0]) == 1 else groups[1] # 有1个桶的列
    
    # 判断共列在左还是右
    if big[0][0] < solo[0][0]:
        # 共列在左: big=[左前, 左后], solo=[右]
        mapping = {'l': big[0], 'm': big[1], 'r': solo[0]}
    else:
        # 共列在右: solo=[左], big=[右前, 右后]
        mapping = {'l': solo[0], 'm': big[1], 'r': big[0]}
```

## 六种情况枚举

用 `●` = 前桶，`○` = 后桶，`◇` = 独桶。

---

### 情况1：3列分开（正常）

```
     ◇            ◇            ◇
   左(B0)       中(B1)       右(B2)
  X=-1.2       X=0.1       X=1.4

mapping: l→0, m→1, r→2
```

| 指令 | 含义 |
|------|------|
| `ml` | 中→左 |
| `mr` | 中→右 |
| `lm` | 左→中 |
| `lr` | 左→右 |
| `rm` | 右→中 |
| `rl` | 右→左 |

✅ 完全不变。

---

### 情况2：共列在左（左列有2桶）

```
    左列              右列
  ┌─────┐
  │ ●B0 │ Y=1.5 左前
  │     │
  │ ○B1 │ Y=2.5 左后 (m=这个!)
  └─────┘           ┌─────┐
  X≈-0.8            │ ◇B2 │ X=1.3
                    └─────┘

groups: [[B0,B1], [B2]]
共列在左 → mapping: l→B0(左前), m→B1(左后), r→B2(右)
```

| 指令 | 去索引 | 实际含义 |
|------|--------|----------|
| `ml` | m→l | **左后○ → 左前●** (同列后→前) |
| `mr` | m→r | 左后○ → 右◇ |
| `lm` | l→m | **左前● → 左后○** (同列前→后) |
| `lr` | l→r | 左前● → 右◇ |
| `rm` | r→m | 右◇ → 左后○ |
| `rl` | r→l | 右◇ → 左前● |

m 自动指向了左列的后桶，语义清晰。

---

### 情况3：共列在右（右列有2桶）

```
    左列              右列
  ┌─────┐           ┌─────┐
  │ ◇B0 │ X=-1.0   │ ●B1 │ Y=1.5 右前
  └─────┘           │     │
                    │ ○B2 │ Y=2.5 右后 (m=这个!)
                    └─────┘
                    X≈1.2

groups: [[B0], [B1,B2]]
共列在右 → mapping: l→B0(左), m→B2(右后), r→B1(右前)
```

| 指令 | 去索引 | 实际含义 |
|------|--------|----------|
| `ml` | m→l | 右后○ → 左◇ |
| `mr` | m→r | **右后○ → 右前●** (同列后→前) |
| `lm` | l→m | 左◇ → 右后○ |
| `lr` | l→r | 左◇ → 右前● |
| `rm` | r→m | **右前● → 右后○** (同列前→后) |
| `rl` | r→l | 右前● → 左◇ |

m 自动指向了右列的后桶。

---

### 情况4：共列在中间（左列1，中列2，右列无）

这不是真正的3列场景——实际上只有2列，但中间的"列"有2桶。

```
    左列            中间列
  ┌─────┐         ┌─────┐
  │ ◇B0 │ X=-1.2 │ ●B1 │ Y=1.5 中前
  └─────┘         │     │
                  │ ○B2 │ Y=2.5 中后 (m=这个!)
                  └─────┘
                  X≈0.0

groups: [[B0], [B1,B2]]
共列在右(中) → mapping: l→B0(左), m→B2(中后), r→B1(中前)
```

| 指令 | 去索引 | 实际含义 |
|------|--------|----------|
| `ml` | m→l | 中后○ → 左◇ |
| `mr` | m→r | **中后○ → 中前●** (同列后→前) |
| `lm` | l→m | 左◇ → 中后○ |
| `lr` | l→r | 左◇ → 中前● |
| `rm` | r→m | **中前● → 中后○** (同列前→后) |
| `rl` | r→l | 中前● → 左◇ |

---

### 情况5：全1列（极端情况）

```
    同一列
  ┌─────┐
  │ ●B0 │ Y=1.5 前
  │ ●B1 │ Y=2.0 中
  │ ○B2 │ Y=2.5 后
  └─────┘

只有1组3桶 → 无法按列拆分
退化为: l=前, m=中, r=后 (按Y排序的直接映射)
```

| 指令 | 含义 |
|------|------|
| `ml` | 中→前 |
| `mr` | 中→后 |
| `lm` | 前→中 |
| `lr` | 前→后 |
| `rm` | 后→中 |
| `rl` | 后→前 |

---

## 伪代码

```python
def resolve_targets(camera_centers, cmd_char1, cmd_char2, threshold=0.5):
    """根据实际几何布局 + 指令字符 决定取哪两个桶"""
    
    # 1. 按 X 分组
    centers = sorted(camera_centers, key=lambda c: c[0])
    groups = []
    for c in centers:
        if not groups or abs(c[0] - groups[-1][-1][0]) > threshold:
            groups.append([c])
        else:
            groups[-1].append(c)
    
    # 2. 每组内按 Y 排（前→后）
    for g in groups:
        g.sort(key=lambda c: c[1])
    
    # 3. 构造映射
    mapping = {}  # 'l'/'m'/'r' → 桶坐标
    
    if len(groups) == 3:
        # 正常: l=左列, m=中列, r=右列
        mapping = {'l': groups[0][0], 'm': groups[1][0], 'r': groups[2][0]}
    
    elif len(groups) == 2:
        big_group = groups[0] if len(groups[0]) == 2 else groups[1]  # 有2桶
        solo_group = groups[0] if len(groups[0]) == 1 else groups[1] # 有1桶
        
        if big_group[0][0] < solo_group[0][0]:
            # 共列在左 → l=左前, m=左后, r=右
            mapping = {'l': big_group[0], 'm': big_group[1], 'r': solo_group[0]}
        else:
            # 共列在右 → l=左, m=右后, r=右前
            mapping = {'l': solo_group[0], 'm': big_group[1], 'r': big_group[0]}
    
    else:  # 全1列
        mapping = {'l': groups[0][0], 'm': groups[0][1], 'r': groups[0][2]}
    
    return mapping[cmd_char1], mapping[cmd_char2]
```

## 总结

| 场景 | 外部协议 | m 语义 | l/r 语义 |
|------|---------|--------|----------|
| 3列分开 | 不改 | 中列 | 左/右列 |
| 共列在左 | 不改 | 左列后桶 | l=左列前桶, r=右列 |
| 共列在右 | 不改 | 右列后桶 | r=右列前桶, l=左列 |
| 全1列 | 不改 | 中(深度) | 前/后(深度) |

**外部 6 指令不变，程序内部自动推断哪个是"后桶"。**
