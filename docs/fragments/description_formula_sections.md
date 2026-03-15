## 9. `fig_formula_tornado.png`

### 9.1 坐标轴
- **横坐标**：归一化敏感度 \(S_x=(x/L)\,\partial L/\partial x\)
- **纵坐标**：变量名（`N_s`, `D`, `A`, `B`, `alpha`, `beta`, `gamma`）

### 9.2 对应公式
基于公式
\[
\tilde L_S(N_s,D)=E+\left(\frac{A}{N_s^\alpha}+\frac{B}{D^\beta}\right)^\gamma
\]
定义局部归一化敏感度：
\[
S_x=\frac{x}{\tilde L_S}\cdot \frac{\partial \tilde L_S}{\partial x}
\]

### 9.3 图像描述与趋势
- 条形越长，表示该变量在基线点附近影响越大。
- 正值表示变量增大导致 \(\tilde L_S\) 增大；负值表示变量增大导致 \(\tilde L_S\) 减小。

### 9.4 可获取信息
- 公式层面最敏感参数排序。
- 局部与全局结果是否一致。

---

## 10. `fig_formula_curve_N.png`

### 10.1 坐标轴
- **横坐标**：`N_s`（log 轴）
- **纵坐标**：\(S_{N_s}\)

### 10.2 对应公式
\[
S_{N_s}=\frac{N_s}{\tilde L_S}\cdot \frac{\partial \tilde L_S}{\partial N_s}
\]

### 10.3 图像描述与趋势
- 反映参数规模变化时的边际敏感度曲线。
- \(|S_{N_s}|\) 下降常对应边际收益递减。

### 10.4 可获取信息
- 扩模最有效区间。
- 规模选择依据。

---

## 11. `fig_formula_curve_D.png`

### 11.1 坐标轴
- **横坐标**：`D`（log 轴）
- **纵坐标**：\(S_D\)

### 11.2 对应公式
\[
S_D=\frac{D}{\tilde L_S}\cdot \frac{\partial \tilde L_S}{\partial D}
\]

### 11.3 图像描述与趋势
- 反映数据规模变化时的边际敏感度曲线。
- \(|S_D|\) 下降常对应加数据边际收益递减。

### 11.4 可获取信息
- 扩数收益区间。
- 与 `N_s` 曲线联合判断扩模/扩数优先级。

---

## 12. 与原有图表的关系（补充说明）

- `fig_sobol_bar.png` 等：**全局敏感性/决策层**
- `fig_formula_tornado.png` 与 `fig_formula_curve_*.png`：**公式本体/局部解析层**
- 联合回答：
  1) 哪里最敏感（局部）  
  2) 结论是否稳健（全局）