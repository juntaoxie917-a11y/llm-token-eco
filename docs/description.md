# 敏感性实验图表讲解（含公式与变量定义）

## 0. 统一模型与变量定义

### 0.1 核心公式

1. **MVP Scaling Law**
\[
L(N,D)=A\,N^{-\alpha}+\frac{B\,D^{-\beta}}{q_{\text{eff}}}+E
\]

2. **算力约束**
\[
C \approx kND \;\Rightarrow\; D=\frac{C}{kN}
\]

3. **有效数据质量系数**
\[
q_{\text{eff}}=\text{quality\_coef}\cdot\left(1+\text{mix\_gain}\cdot(\text{token\_mix}-0.5)\right)
\]

4. **最优配比**
\[
\text{opt\_ratio}=\frac{N^*}{D^*}
\]

5. **质量-成本代理指标**
\[
\text{qpc}=\frac{1}{L^*}
\]

---

### 0.2 变量含义

- \(N\)：参数量（model size）
- \(D\)：训练 token 数（data size）
- \(C\)：训练总算力预算（FLOPs）
- \(A,B,E\)：拟合常数（含不可约误差底）
- \(\alpha\)：参数项幂指数（模型扩展收益速度）
- \(\beta\)：数据项幂指数（数据扩展收益速度）
- `quality_coef`：数据质量强度系数
- `token_mix`：语料混合比例变量（如高质量/代码/通用的抽象比例）
- `mix_gain`：`token_mix` 对有效质量的放大系数

---

## 1. `fig_sobol_bar.png`

### 1.1 坐标轴
- **横坐标**：`alpha`, `beta`, `quality_coef`, `token_mix`
- **纵坐标**：Sobol 指数（两组柱：`S1` 与 `ST`）

### 1.2 对应公式
- 一阶效应：
\[
S_i=\frac{\mathrm{Var}_{X_i}\left(\mathbb{E}[Y|X_i]\right)}{\mathrm{Var}(Y)}
\]
- 总效应：
\[
ST_i=1-\frac{\mathrm{Var}_{X_{\sim i}}\left(\mathbb{E}[Y|X_{\sim i}]\right)}{\mathrm{Var}(Y)}
\]
其中 \(Y=\text{best\_loss}\)（当前脚本主输出）。

### 1.3 图像描述与趋势
- 柱越高，变量影响越大。
- `ST` 明显大于 `S1` 说明交互作用强。

### 1.4 可获取信息
- 变量影响排序（优先优化对象）。
- 是否需要多变量联合调参（看交互强度）。

---

## 2. `fig_opt_ratio_drift.png`

### 2.1 坐标轴
- **横坐标**：`alpha`
- **纵坐标**：\(\text{opt\_ratio}=N^*/D^*\)
- **颜色**：`quality_coef`

### 2.2 对应公式
在
\[
L(N,D)=A N^{-\alpha}+B D^{-\beta}/q_{\text{eff}}+E,\quad D=C/(kN)
\]
下求
\[
(N^*,D^*)=\arg\min L(N,D),\quad \text{opt\_ratio}=N^*/D^*
\]

### 2.3 图像描述与趋势
- 散点越分散，最优配比越不稳健。
- 颜色分层明显时，质量变量对最优配比影响大。

### 2.4 可获取信息
- 预算分配（扩模 vs 扩数）是否稳健。
- 哪些假设下最优配比会翻转。

---

## 3. `fig_alpha_beta_ci.png`

### 3.1 坐标轴
- 左图：横轴 `alpha`，纵轴频数
- 右图：横轴 `beta`，纵轴频数
- 虚线：中位数；阴影：95% CI

### 3.2 对应公式
Bootstrap 统计（中位数）：
\[
\hat{\theta}^{(b)}=\mathrm{median}(X^{*(b)}),\quad
CI_{95\%}=[Q_{2.5\%},Q_{97.5\%}],\ \theta\in\{\alpha,\beta\}
\]

### 3.3 图像描述与趋势
- 分布集中且 CI 窄：参数稳定。
- 分布分散或偏态明显：不确定性较高。

### 3.4 可获取信息
- 幂指数可信区间。
- 外推风险（CI 越宽，风险越高）。

---

## 4. `fig_curve_alpha.png`

### 4.1 坐标轴
- 左图：横轴 `alpha`，纵轴 `best_loss_mean`
- 右图：横轴 `alpha`，纵轴 `opt_ratio_mean`
- 阴影：标准差带

### 4.2 对应公式
OAT（单变量）：
\[
x=\alpha\ \text{变化，其余变量固定中点},\quad
\bar{Y}(x)=\frac{1}{R}\sum_{r=1}^R Y_r(x)
\]

### 4.3 图像描述与趋势
- 曲线越陡，说明对 `alpha` 越敏感。
- 阴影越宽，说明重复实验波动越大。

### 4.4 可获取信息
- `alpha` 的高敏感区间与潜在阈值点。
- 是否需要更精细估计 `alpha`。

---

## 5. `fig_curve_beta.png`

### 5.1 坐标轴
- 左图：横轴 `beta`，纵轴 `best_loss_mean`
- 右图：横轴 `beta`，纵轴 `opt_ratio_mean`

### 5.2 对应公式
同 OAT：
\[
x=\beta,\quad \bar{Y}(x)=\frac{1}{R}\sum_{r=1}^R Y_r(x)
\]

### 5.3 图像描述与趋势
- 观察单调性、曲率、斜率变化。
- 局部斜率大处为高敏感区。

### 5.4 可获取信息
- 数据扩展收益参数不确定时的决策偏差。
- `beta` 对最优 N:D 的影响强度。

---

## 6. `fig_curve_quality_coef.png`

### 6.1 坐标轴
- 左图：横轴 `quality_coef`，纵轴 `best_loss_mean`
- 右图：横轴 `quality_coef`，纵轴 `opt_ratio_mean`

### 6.2 对应公式
\[
q_{\text{eff}}=\text{quality\_coef}\cdot(1+\text{mix\_gain}(\text{token\_mix}-0.5))
\]
代入
\[
L=A N^{-\alpha}+B D^{-\beta}/q_{\text{eff}}+E
\]

### 6.3 图像描述与趋势
- 左图常见趋势是随质量提升而 loss 下降。
- 右图若变化明显，说明数据质量会改变最优资源配比。

### 6.4 可获取信息
- 数据清洗/筛选投入的边际收益。
- 提质是否比继续扩模更划算。

---

## 7. `fig_curve_token_mix.png`

### 7.1 坐标轴
- 左图：横轴 `token_mix`，纵轴 `best_loss_mean`
- 右图：横轴 `token_mix`，纵轴 `opt_ratio_mean`

### 7.2 对应公式
\[
q_{\text{eff}}(\text{token\_mix})=\text{quality\_coef}\cdot(1+\text{mix\_gain}(\text{token\_mix}-0.5))
\]
通过 \(q_{\text{eff}}\) 间接影响 \(L\) 和 \(\text{opt\_ratio}\)。

### 7.3 图像描述与趋势
- 若左图出现最低点，表示存在最佳语料配比。
- 若右图波动大，表示 token 配方会显著影响最优 N:D。

### 7.4 可获取信息
- 语料配方优化方向（提高或降低某类语料占比）。
- 配方变化对训练策略稳定性的影响。

---

## 8. 综合可得结论类型

- **谁最重要**：看 `fig_sobol_bar.png`
- **最优 N:D 是否稳定**：看 `fig_opt_ratio_drift.png`
- **指数参数不确定性**：看 `fig_alpha_beta_ci.png`
- **单变量阈值与敏感区间**：看 `fig_curve_*.png`

以上可直接支撑后续实验决策：优先调哪些变量、在哪些区间加密采样、哪些结论可稳健外推。

<!-- formula-sections:start -->
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
<!-- formula-sections:end -->
