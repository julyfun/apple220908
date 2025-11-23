## 模糊边缘的数学原理

```python
mask = cairo.RadialGradient(x, y, radius * (1 - blur), x, y, radius)
mask.add_color_stop_rgba(0, 1, 1, 1, 1)  # Center: alpha = 1
mask.add_color_stop_rgba(1, 1, 1, 1, 0)  # Outside: alpha = 0
```

**径向渐变 (RadialGradient)** 定义两个同心圆：
- 内圆半径：`radius * 0.7` (假设 blur=0.3)
- 外圆半径：`radius`

在两圆之间，alpha 值从 1 线性插值到 0：

$$
\alpha(r) = \begin{cases}
1 & r \leq r_{inner} \\
\frac{r_{outer} - r}{r_{outer} - r_{inner}} & r_{inner} < r < r_{outer} \\
0 & r \geq r_{outer}
\end{cases}
$$

这创建了**软边缘**，避免硬切割，视觉上产生"模糊"或"发光"效果。

## (Additive Glow)

核心思路：每颗星星绘制两层
1. **核心层**：小半径，亮
2. **光晕层**：大半径，暗淡但可叠加

使用 Cairo 的**加法混合**，多颗星星的光晕自然累积成更亮的区域。

关键代码改动：

```python
# 在 draw_blurred_circle 中添加光晕层
def draw_blurred_circle(cr, x, y, radius, blur=BLUR_AMOUNT):
    # 光晕层（大范围、低亮度）
    cr.set_operator(cairo.OPERATOR_ADD)  # 加法混合
    halo_mask = cairo.RadialGradient(x, y, 0, x, y, radius * 3)
    halo_mask.add_color_stop_rgba(0, 1, 1, 1, 0.03)  # 中心微弱
    halo_mask.add_color_stop_rgba(1, 1, 1, 1, 0)     # 外围消失
    cr.set_source(halo_mask)
    cr.paint()
    
    # 核心层（小范围、高亮度）
    cr.set_operator(cairo.OPERATOR_OVER)  # 恢复正常混合
    core_mask = cairo.RadialGradient(x, y, radius * (1 - blur), x, y, radius)
    core_mask.add_color_stop_rgba(0, 1, 1, 1, 1)
    core_mask.add_color_stop_rgba(1, 1, 1, 1, 0)
    cr.set_source(core_mask)
    cr.paint()


def draw_blurred_circle(cr, x, y, radius, blur=BLUR_AMOUNT):
    1. Draw a large, faint glow (halo) using additive blending
    
    2. Draw a smaller, sharp core with a soft edge using normal blending
```

**数学原理**：
- 正常混合：\( C_{out} = \alpha \cdot C_{src} + (1-\alpha) \cdot C_{dst} \)
- 加法混合：\( C_{out} = C_{src} + C_{dst} \)（饱和到 1）

## 03 采样

在3D空间中有一个**边长为1的正方形**，中心在 \((1, 0, 0)\)，垂直于 \(x\) 轴（朝向相机）。

正方形的四个顶点：
- 左上：\((1, -0.5, 0.5)\)
- 右上：\((1, 0.5, 0.5)\)  
- 左下：\((1, -0.5, -0.5)\)
- 右下：\((1, 0.5, -0.5)\)

从四条边中等概率选择一条（概率各 \(\frac{1}{4}\)），然后在该边上均匀采样位置 \(t \sim \mathcal{U}(-0.5, 0.5)\)。

采样点集合：

\[
S = \{ \mathbf{p}_i \mid i = 1, 2, \ldots, 100 \}, \quad \mathbf{p}_i \in \bigcup_{j=1}^{4} \text{Edge}_j
\]

## 移动（因为看起来在空间中散布）

- [ ] 需要生成一个平面视频，说明效果不好

