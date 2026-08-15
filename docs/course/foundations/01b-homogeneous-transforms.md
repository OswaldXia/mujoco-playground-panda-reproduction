# 01B｜齐次变换与组合顺序

建议 90 分钟。前置：01A。

## 为什么增加一维

旋转后的点是 `R @ p`，再平移是 `R @ p + t`。齐次坐标把三维点 `[x,y,z]`
写成 `[x,y,z,1]`，于是旋转和平移可统一成一次矩阵乘法：

$$
T=\begin{bmatrix}R&t\\0&1\end{bmatrix},\qquad
\bar p_A=T^A_B\bar p_B.
$$

方向向量使用末位 0，因此不会收到平移。这个技巧也是“点和向量语义”在矩阵中的
显式编码。

## 组合就是沿坐标链消去中间系

若已知 `T_world_base` 和 `T_base_tip`：

```python
T_world_tip = T_world_base @ T_base_tip
```

右边最先作用。中间的 `base` 像单位一样相消：world←base←tip。若写反，矩阵
shape 可能仍合法，但物理意义错误，这是最危险的一类 bug。

逆变换不必对整个 4×4 矩阵做通用求逆：

$$
T^{-1}=\begin{bmatrix}R^T&-R^Tt\\0&1\end{bmatrix}.
$$

因为旋转矩阵的逆就是转置。

## Worked example

先将 `[1,0]` 旋转 90°，再平移 `[2,1]`：

```text
T = T_translate @ T_rotate
T @ [1,0,1] = [2,2,1]
```

若交换次序，平移向量也会被旋转，结果为 `[-1,3,1]`。所以变换乘法一般不可
交换。这里的“先”指对点的作用顺序，与代码从左到右阅读相反。

## 暂停并预测

设 `T_A_B @ T_B_C @ p_C`：

1. 最先对点作用的是哪一项？
2. 结果在哪个坐标系表达？
3. 要把结果变回 C，应乘哪个逆链？

答案：最右侧 `T_B_C`；A；`inv(T_B_C) @ inv(T_A_B)`。

## 主动实验

先运行已完成演示，再补全三维练习：

```bash
python docs/labs/01_transform_2d.py
python docs/labs/starter/01_transform_3d_exercise.py
```

第二条第一次应提示 TODO，而不是成功。按文件内 Level 1→3 提示逐步完成；只有
卡住 20 分钟后才对照 `docs/solutions/labs/01_transform_3d_solution.py`。

## 通过标准

能从坐标系下标确定乘法顺序；能手写刚体逆变换；三维 starter 的断言全部通过，
并能解释为什么最后一维为 1 的点会平移、为 0 的方向不会。
