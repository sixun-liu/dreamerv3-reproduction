# DreamerV3 代码走读：官方 JAX 码与 PyTorch 复刻对标

> **定位**：本文是 `世界模型精读线/WM1_f1_staging.md` §WM1-1.5（论文侧深读，L1041-1343）的代码级补全。当时"池仅 paper.md"，映射表停在功能描述层并留下多个 PARTIAL 歧义；本文把每个论文组件落到真实 file:line，并对标 PyTorch 社区复刻，给出复现关键点清单与可跑命令。
>
> **走读日期**：2026-07-16。**file:line 一律以下述克隆快照的实际源码为准**（本人逐行打开核对，非记忆复述）。

---

## ① 两仓概况与锚定

| 仓 | 克隆源 | HEAD（删 .git 前记录） | 提交日期 | 快照位置 |
|---|---|---|---|---|
| 官方 JAX | `github.com/danijar/dreamerv3` | `e3f02248693a79dc8b0ebd62c93683888ddaccfe`（"Fix Atari frame maxpooling on reset (#213)"） | 2026-05-24 | `_src/dreamerv3/`（6.6M） |
| PyTorch 复刻 | `github.com/NM512/dreamerv3-torch` | `6ef8646d807cd10ce0c88e10a7e943211e7fc44c`（"Merge pull request #81"） | 2026-03-08 | `_src/dreamerv3-torch/`（3.0M） |

两仓均为 `--depth 1` 浅克隆并删除 `.git`，以上 HEAD 哈希 + 克隆日期（2026-07-16）为版本锚。下文行号均针对这两个快照。

**关键版本事实（走读第一发现）**：当前官方仓是 **Nature 2025 版的重写码**，不是 2023 arXiv v1 时代的代码。证据：

- README 引用格式即 Nature 2025（`_src/dreamerv3/README.md:12-19`）；
- 论文第 20 页 "Previous Dreamer generations" 明确列出 v3 的架构组件：**Block GRU、RMSNorm、SiLU、LaProp（RMSProp 后接 momentum）、AGC 梯度裁剪、replay 存取 latent**——重写码全部对应实现（见下文）；
- 损失权重 $\beta_{\text{dyn}}=1$（2023 v1 为 0.5）。

而 **dreamerv3-torch 对标的是 2023 v1 时代的老官方码**：标准 GRU、$\beta_{\text{dyn}}=0.5$、三个 Adam 优化器、无 replay critic 项。两代码线的差异本身就是重要信息（§⑨）。

官方仓结构：`dreamerv3/{main.py, agent.py, rssm.py, configs.yaml}`（算法核心，共 1345 行）+ `embodied/jax/{nets, heads, outs, utils, opt, agent}.py`（网络/分布/优化器基建）+ `embodied/{core, run, envs, replay}`（训练循环与环境）。torch 仓为平铺 5 文件：`dreamer.py / models.py / networks.py / tools.py / configs.yaml`。

---

## ② 走读清单逐项

以下每项给出 file:line、关键源码摘录、与论文公式的对应/出入判定。论文行号引用 `_parsed/2301.04104.txt`（Nature 对应版全文解析）。

### 1. RSSM（论文公式 (1)）

**类与默认超参**：`_src/dreamerv3/dreamerv3/rssm.py:16-32`

```python
class RSSM(nj.Module):
  deter: int = 4096
  hidden: int = 2048
  stoch: int = 32
  classes: int = 32
  ...
  unimix: float = 0.01
  ...
  blocks: int = 8
  free_nats: float = 1.0
```

实际生效值来自 `configs.yaml:91`（defaults，即 200M 档）：`deter: 8192, hidden: 1024, stoch: 32, classes: 64, act: silu, norm: rms, unimix: 0.01, blocks: 8, free_nats: 1.0`。

**确定态：Block GRU（非普通 GRU）**：`rssm.py:135-159`（`_core`）

```python
def _core(self, deter, stoch, action):
    stoch = stoch.reshape((stoch.shape[0], -1))
    action /= sg(jnp.maximum(1, jnp.abs(action)))          # L137 动作幅值裁剪
    ...
    x = self.sub('dyngru', nn.BlockLinear, 3 * self.deter, g, **self.kw)(x)  # L152
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(x) for x in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)                    # L157 update 门偏置 -1
    deter = update * cand + (1 - update) * deter
```

`nn.BlockLinear`（8 块块对角线性层，`blocks: 8`）替代稠密 GRU 权重；隐层先经 `dynhid` BlockLinear（`rssm.py:149-151`）。**对应/出入**：公式 (1) 的 sequence model $h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$ 论文示意为 GRU；Nature 版 p20 明确 "Block GRU"，码上就是 `blocks=8` 的块对角 GRU 变体，`update` 门偏置 $-1$（初始倾向保持旧状态）。与论文一致，但注意 2023 v1 论文/老码是标准 GRU——重写时不可混用。

**随机态（离散分类，straight-through）**：后验 `rssm.py:75-92`（`_observe`，deter 与 encoder tokens 拼接后过 `obs` 层出 logit，L86-87 采样）；先验 `rssm.py:161-166`（`_prior`）；分布 `rssm.py:173-176`：

```python
def _dist(self, logits):
    out = embodied.jax.outs.OneHot(logits, self.unimix)   # 1% 均匀混合在此注入
    out = embodied.jax.outs.Agg(out, 1, jnp.sum)
    return out
```

straight-through 梯度在 `embodied/jax/outs.py:265-270`（`_onehot_with_grad`：`sg(value) + (probs - sg(probs))`）。

**出入判定——"$32\times 32$ 离散"需按规模档修正**：`stoch=32` 全档固定，但 `classes` 随档变（$d/16$，见 §7）：50M 档才是 $32\times 32$；默认 200M 档是 $32\times 64$。论文/WM1_f1 口径的"$32\times 32$"对应 2023 v1 各档（classes 固定 32），Nature 版已改。

**continue flag**：头定义 `agent.py:56-58`（`binary` 空间 + `conhead`，`configs.yaml:99` 输出 `binary`）；训练目标 `agent.py:174-177`：

```python
con = f32(~obs['is_terminal'])
if self.config.contdisc:
    con *= 1 - 1 / self.config.horizon
```

**出入**：`contdisc: True`（`configs.yaml:107`）把折扣 $1-1/333$ 直接乘进 continue 目标，因此想象损失里 `disc = 1 if contdisc`（`agent.py:401`）。**重写陷阱**：若照公式 (5) 再乘一遍 $\gamma=0.997$ 就是双重折扣。

**想象 rollout**：`rssm.py:94-118`（`imagine`，policy 输入 `sg(carry)`，先验采样推进）。

### 2. 世界模型损失 + free bits（公式 (2)(3)）

**KL 两支 + free bits**：`rssm.py:120-133`

```python
def loss(self, carry, tokens, acts, reset, training):
    ...
    prior = self._prior(feat['deter'])
    post = feat['logit']
    dyn = self._dist(sg(post)).kl(self._dist(prior))     # L125  KL(sg(post) || prior)
    rep = self._dist(post).kl(self._dist(sg(prior)))     # L126  KL(post || sg(prior))
    if self.free_nats:
        dyn = jnp.maximum(dyn, self.free_nats)           # L128  1 nat 下界裁剪
        rep = jnp.maximum(rep, self.free_nats)
    losses = {'dyn': dyn, 'rep': rep}
```

**$\beta$ 权重落点**：`configs.yaml:86`

```yaml
loss_scales: {rec: 1.0, rew: 1.0, con: 1.0, dyn: 1.0, rep: 0.1, policy: 1.0, value: 1.0, repval: 0.3}
```

加权求和在 `agent.py:237-240`（`loss = sum([v.mean() * self.scales[k] ...])`）。预测损失三项（decoder 重建 + reward + continue）在 `agent.py:172-182`。

**对应**：与论文公式 (2)(3) 一致——$\beta_{\text{pred}}=1$（rec/rew/con 各 1）、$\beta_{\text{dyn}}=1$、$\beta_{\text{rep}}=0.1$（论文 parsed L283 原句给出的三个值与此逐一相同）；free bits 论文写 $\max(1,\mathrm{KL})$、码写 `maximum(KL, 1)`，同一运算，KL 低于 1 nat 时输出常数、梯度为零。**版本出入**：2023 v1 论文为 $\beta_{\text{dyn}}=0.5$（torch 复刻沿用），Nature 版改为 1。

**冒烟实证**：CPU debug 跑随机小任务时日志即出现 `train/loss/dyn 1 / train/loss/rep 1`——KL 学到 1 nat 以下被裁在下界，free bits 生效的直接可观察证据（见 §④）。

### 3. symlog/symexp（公式 (8)(9)）与 twohot 分类损失（公式 (10)(11)）

**symlog/symexp 定义**：`embodied/jax/nets.py:59-64`

```python
def symlog(x):
  return jnp.sign(x) * jnp.log1p(jnp.abs(x))
def symexp(x):
  return jnp.sign(x) * jnp.expm1(jnp.abs(x))
```

与公式 (9) 严格一致（`log1p/expm1` 为数值稳定写法）。用途落点：向量观测编码输入 `rssm.py:218-219`（`squish = nn.symlog if self.symlog else ...`）；向量观测解码目标 `rssm.py:299`（`symlog_mse`，即公式 (8) 的 symlog 平方损失，实现在 `heads.py:127-130` + `outs.py:129-141` MSE(squash=symlog)）。

**twohot 的 bin：数量与范围以码为准**：`embodied/jax/heads.py:132-144`

```python
def symexp_twohot(self, x):
    shape = (*self.space.shape, self.bins)
    logits = self.sub('logits', nets.Linear, shape, **self.kw)(x)
    if self.bins % 2 == 1:
        half = jnp.linspace(-20, 0, (self.bins - 1) // 2 + 1, dtype=f32)
        half = nets.symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]], 0)
    ...
    return outs.TwoHot(logits, bins)
```

`bins: 255`（`configs.yaml:98` reward 头、`configs.yaml:101` critic 头）。**判定**：数学上与论文公式 (10) 的 $B=\operatorname{symexp}(\text{linspace}(-20,20,255))$ 同一网格（255 点关于 0 对称、symlog 空间等距 $20/127$），码用"负半轴取点后镜像"的构造保证浮点上严格对称；范围 $\pm(\mathrm{e}^{20}-1)\approx\pm 4.85\times 10^{8}$。目标**不做 symlog 变换**、直接在原空间对指数间隔 bin 做 twohot（`TwoHot(logits, bins)` 未传 squash）。

**twohot 编码 + 交叉熵**：`embodied/jax/outs.py:311-330`

```python
def loss(self, target):
    ...
    below = (self.bins <= target[..., None]).astype(i32).sum(-1) - 1
    above = len(self.bins) - (self.bins > target[..., None]).astype(i32).sum(-1)
    ...
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    target = (jax.nn.one_hot(below, ...) * weight_below[..., None] +
              jax.nn.one_hot(above, ...) * weight_above[..., None])
    log_pred = self.logits - jax.scipy.special.logsumexp(self.logits, -1, keepdims=True)
    return -(target * log_pred).sum(-1)
```

与公式 (11)(12) 一致（相邻两 bin 线性分权、和为 1、软标签交叉熵）。

**读出的对称求和（论文点名的实现细节）**：`outs.py:285-309`，`pred()` 不用朴素 `sum(probs * bins)`，而是正负半轴各自由小到大求和再相加（L292-309），源码注释明说是为了"bins 对称 + 概率均匀时预测严格为 0"。对应论文原文（parsed L1067-1070）"the summation order matters and positive and negative bins should be summed up separately"。它与输出零初始化（§6）配合，保证初始预测恰为 0。

### 4. critic（公式 (5)）：$\lambda$-return 与 replay 权重 0.3

**$\lambda$-return**：`agent.py:482-490`

```python
def lambda_return(last, term, rew, val, boot, disc, lam):
    rets = [boot[:, -1]]
    live = (1 - f32(term))[:, 1:] * disc      # 终止 -> 停折扣
    cont = (1 - f32(last))[:, 1:] * lam       # 截断 -> 停 lambda 混合
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
```

对应公式 (5)，且把"episode 真终止（term）"与"轨迹截断（last）"分开处理。想象轨迹里 `last=0, term=1-con`（`agent.py:403-404`）；$\lambda=0.95$、horizon 333（`configs.yaml:106,108`）。

**critic 训练（想象轨迹，$\beta_{\text{val}}=1$）**：`agent.py:417-422`

```python
losses['value'] = sg(weight[:, :-1]) * (
    value.loss(sg(tar_padded)) +
    slowreg * value.loss(sg(slowvalue.pred())))[:, :-1]
```

critic 输出即 symexp_twohot 255 bins（`configs.yaml:101`），回归 $\lambda$-return 的 twohot 交叉熵 + EMA 正则（§6）。

**★ PARTIAL 补齐：replay 轨迹 critic 损失，真实权重 0.3**。三个落点：

1. 开关与调用：`agent.py:219-235`（`if self.config.repval_loss:`，`configs.yaml:115` 为 True）；
2. bootstrap 构造：`agent.py:222` `boot = imgloss_out['ret'][:, 0].reshape(B, K)`——**用想象 rollout 起点的 $\lambda$-return 作为 replay 轨迹的 on-policy 价值标注**，再对 replay 真实奖励算 $\lambda$-return（`repl_loss`，`agent.py:449-479`，其中 L466 `ret = lambda_return(last, term, rew, tarval, boot, disc, lam)`，L464 `disc = 1 - 1/horizon`）。与论文 L373-377 的机制描述逐句对应；
3. 权重：`configs.yaml:86` 的 `repval: 0.3` = 论文 Table 4 "Critic replay loss scale $\beta_{\text{repval}}$ 0.3"。

WM1_f1 记的"replay critic 权重 0.3"完全属实，且现在有了机制级答案（bootstrap 从哪来）。

**另一个易漏细节**：`slowtar: False`（`configs.yaml:108-109`）→ `tarval = val`（`agent.py:400`），即 **$\lambda$-return 用当前 critic 计算**，慢网络只出现在正则项——正合论文"allows us to compute returns using the current critic network"。

### 5. actor：百分位归一化（公式 (6)(7)）

**损失主体**：`agent.py:407-415`

```python
roffset, rscale = retnorm(ret, update)
adv = (ret - tarval[:, :-1]) / rscale
...
logpi = sum([v.logp(sg(act[k]))[:, :-1] for k, v in policy.items()])
policy_loss = sg(weight[:, :-1]) * -(
    logpi * sg(adv_normed) + actent * sum(ents.values()))
```

统一 Reinforce（离散/连续同式），熵系数 `actent=3e-4`（默认值 `agent.py:391`，配置 `configs.yaml:108`）= 论文 $\eta=3\times 10^{-4}$。注意 `advnorm`/`valnorm` 默认 `impl: none`（`configs.yaml:112-113`），所以 `adv_normed` 实际就是 `adv`——重写时不要误以为还有第二层归一化。

**百分位 + $\max(1,S)$ 落点**：`embodied/jax/utils.py:16-91`（`Normalize`）

```python
elif self.impl == 'perc':
    self._update(self.lo, self._perc(x, self.perclo))   # L52  EMA(Per(R,5))
    self._update(self.hi, self._perc(x, self.perchi))   # L53  EMA(Per(R,95))
...
elif self.impl == 'perc':
    lo, hi = self.lo.read() * corr, self.hi.read() * corr
    return sg(lo), sg(jnp.maximum(self.limit, hi - lo)) # L72  max(limit, S)
```

EMA 更新 `utils.py:90-91`（`(1-rate)*old + rate*new`）。配置 `configs.yaml:111`：`retnorm: {impl: perc, rate: 0.01, limit: 1.0, perclo: 5.0, perchi: 95.0}`——rate 0.01 即论文的 decay 0.99；`limit: 1.0` 即 $\max(1,S)$；5-95 百分位与公式 (7) 一致。**对应**：完全一致；offset 只减不加进 actor 梯度这点论文也明说（减常数不改变梯度），码中 `roffset` 仅用于 metrics（`agent.py:424`）。

连续动作分布为 `bounded_normal`（`configs.yaml:103`；`heads.py:146-155`：`tanh(mean)`、std 经 sigmoid 压到 $[0.1, 1.0]$）；policy 输出层 `outscale: 0.01`、离散 policy 亦有 `unimix: 0.01`（`configs.yaml:100`）。

### 6. 三件散落的稳定化手术（PARTIAL 全补齐）

**(a) EMA target critic——系数 0.02/每步**：`embodied/jax/utils.py:94-127`（`SlowModel`）

```python
def update(self):
    self._initonce()
    mix = jnp.where(self.count.read() % self.every == 0, self.rate, 0)
    fn = lambda src, dst: mix * src + (1 - mix) * dst
    values = jax.tree.map(fn, self.source.values, self.model.values)
```

实例化 `agent.py:66-68`（`SlowModel(MLPHead(...), source=self.val, **config.slowvalue)`），配置 `configs.yaml:110` `slowvalue: {rate: 0.02, every: 1}`——每个梯度步向当前 critic 混 2%，等价论文 Table 4 "Critic EMA decay 0.98"。更新时机：每次 train 后 `agent.py:142` `self.slowval.update()`。正则用法（非硬目标网络）：`agent.py:420-422` 的 `slowreg * value.loss(sg(slowvalue.pred()))`，`slowreg: 1.0`（`configs.yaml:108`）= Table 4 "Critic EMA regularizer 1"。

**(b) 输出层零初始化——reward 头与 critic 头**：机制在 `embodied/jax/nets.py:250-251`

```python
def _scaled_winit(self, *args, **kwargs):
    return init(self.winit)(*args, **kwargs) * self.outscale
```

`Linear` 的权重初始化乘 `outscale`（`nets.py:230-251`）；落点配置 `configs.yaml:98` `rewhead: {..., outscale: 0.0, ...}` 与 `configs.yaml:101` `value: {..., outscale: 0.0, ...}`——两处输出层权重初始化为全零（bias 本就 zeros，`nets.py:234`）。对应论文 L382-384/L1066。注意 decoder 与 continue 头 `outscale: 1.0`、policy `0.01`、RSSM logit `1.0`（`rssm.py:26` + `configs.yaml:91`）——只有 twohot 两个头是 0。

**(c) 分类分布混 1% 均匀（unimix）**：`embodied/jax/outs.py:208-217`

```python
class Categorical(Output):
  def __init__(self, logits, unimix=0.0):
    logits = f32(logits)
    if unimix:
      probs = jax.nn.softmax(logits, -1)
      uniform = jnp.ones_like(probs) / probs.shape[-1]
      probs = (1 - unimix) * probs + unimix * uniform
      logits = jnp.log(probs)
```

`OneHot` 包装它（`outs.py:243-246`）。**三处**生效：RSSM 后验与先验（`rssm.py:173-176` `_dist`，unimix=0.01）+ **离散 actor**（`configs.yaml:100` `policy: {..., unimix: 0.01}`，经 `heads.py:112-115` onehot 头）。论文 L1043-1045 说 "encoder, dynamics predictor, and actor distributions"——**WM1_f1 只记了前两处，actor 的 1% 均匀是走读新增的第三落点**（Table 4 亦列 "Actor unimix 1%"）。

### 7. 模型规模档（PARTIAL 补齐）

`configs.yaml:120-153` 定义 7 档（正则表达式 patch 语法，叠加在 defaults 上）：

```yaml
size12m:  {.*\.rssm: {deter: 2048, hidden: 256,  classes: 16}, .*\.depth: 16, .*\.units: 256}
size25m:  {deter: 3072, hidden: 384,  classes: 24}  depth/units: 24/384
size50m:  {deter: 4096, hidden: 512,  classes: 32}  depth/units: 32/512
size100m: {deter: 6144, hidden: 768,  classes: 48}  depth/units: 48/768
size200m: {deter: 8192, hidden: 1024, classes: 64}  depth/units: 64/1024
size400m: {deter: 12288, hidden: 1536, classes: 96} depth/units: 96/1536
```

另有码上独有的迷你档 `size1m`（`configs.yaml:120-123`，deter 512/units 64/classes 4）。规则（论文 p20 Table 3 及正文）：模型维 $d$ = MLP hidden units；`deter` $=8d$（8 块 Block GRU 每块 $d$）；CNN 基础通道与每 latent 类数均 $=d/16$；latent 个数（stoch=32）与层数全档固定。**defaults 即 200M 档**（`configs.yaml:91`；论文 L1188 "Dreamer uses the 200M model size by default"）。

**对应/出入**：

- 25M-400M 五档与论文 Table 3 逐格一致（已对 PDF 第 20 页核验）。
- **论文 Table 3 的 12M 行 "Recurrent units (8d)" 印作 1024，而码 `size12m` 为 `deter: 2048`**；$8d=8\times 256=2048$，规则与代码互证 2048，论文表该格疑为排版笔误（其余五档均满足 $8d$）。
- WM1_f1 期望的"XS/S/M/L/XL 档"是 2023 arXiv v1 的命名体系，Nature 版已改为参数量命名（12M-400M）；两者不能混查（v1 的 XL deter=4096 classes=32 大致对应新版 50M 档形状）。
- 域与档的绑定：`dmc_proprio` 默认叠 `size1m`（`configs.yaml:178-182`）；其余域配置块不改档即用 200M。论文 L1189-1190 称 control suites 用 12M 即可达 200M 同性能。

### 8. 训练循环与复现操作面

**入口**：`dreamerv3/main.py:19-124`。`--configs` 依序叠加 configs.yaml 里的块，其余 flag 逐项覆盖（`main.py:25-29`）。`script: train`（默认，`configs.yaml:9`）走 `embodied/run/train.py`。标准命令（README:71-75）：

```sh
python dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{timestamp} \
  --configs crafter \
  --run.train_ratio 32
```

**train_ratio 语义**：`embodied/run/train.py:24-25`

```python
batch_steps = args.batch_size * args.batch_length      # 默认 16*64=1024
should_train = elements.when.Ratio(args.train_ratio / batch_steps)
```

即 **train_ratio = 每个环境步重放训练的样本步数**（replayed steps per env step）。例：crafter `train_ratio: 512` → 每 2 个环境步做 1 次梯度更新（每次更新消费 1024 样本步）。训练触发在采集回调里（`train.py:70-81`），replay 未满一个 batch 前不训（L71-72）。

**replay buffer**：`main.py:183-209`（容量 `5e6`、uniform 采样、`online: True`、chunk 1024——`configs.yaml:39-46`）；`replay_context: 1`（`configs.yaml:15`）配合 latent 存取：policy/train 把 `enc/dyn/dec` 的 entry 写回 replay（`agent.py:132-134`、`agent.py:144-150`），训练时用存储的 latent 截断续算（`agent.py:312-340` `_apply_replay_context`）——对应论文 p20 "Replay buffer: ... storing and updating latent states"。

**日志与论文曲线口径**：episode 结束时聚合 `episode/score`（`train.py:46-51`）；`scores.jsonl` 专门滤出 `episode/score`（`main.py:160-162`）。**看论文曲线就看 `episode/score`**（横轴 env steps，Atari 类要乘 action repeat，`main.py:155` 的 multiplier）。查看：`python -m scope.viewer --basedir ~/logdir --port 8000`（README:82-85）。

**依赖（requirements.txt）**：`jax[cuda12]==0.4.33`、`numpy<2`（注释：DMLab/MineRL 需要）、`elements>=3.19.1`、`ninjax>=3.5.1`、`portal>=3.5.0`、`granular>=0.20.3`、`scope>=0.4.4`、optax/einops/chex 等；Python 3.11+（README:53）。环境包按需另装（Dockerfile:27-30：ale_py 0.9.0、procgen_mirror、crafter、dm_control）。

**Crafter 与 DMC 配置差异**（`configs.yaml:174-187`）：

| 块 | 关键覆盖 |
|---|---|
| `crafter` | `task: crafter_reward; steps 1.1e6; envs 1; train_ratio 512`（64x64 图像，`env.crafter` L32） |
| `dmc_vision` | `task: dmc_walker_walk; steps 1.1e6; train_ratio 256; env.dmc.proprio: False`（纯像素） |
| `dmc_proprio` | 叠 `size1m`；`train_ratio 1024; env.dmc.image: False`（纯向量，最轻） |
| `atari100k` | `steps 1.1e5; envs 1; train_ratio 256`（64x64、无 sticky，L33） |

**CPU 最小可跑（debug 块，`configs.yaml:204-220`）**：`jax.platform: cpu`、batch $8\times 10$、`train_ratio 8`、envs 4，网络缩到玩具级（`bins 5 / layers 1 / units 8 / stoch 2 / classes 4 / deter 8 / hidden 3 / blocks 4 / depth 2`）。**本机实测两条命令均跑通**（见 §④）。

### 9. torch 复刻对标（dreamerv3-torch）

复刻整体忠实于 **2023 v1 老官方码**；入口 `python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/...`（README:22），依赖 torch==2.4.1（requirements.txt）。保真件：symlog/symexp（`tools.py:23-28`）、twohot 编码逻辑（`tools.py:478-502`，与官方 `outs.py:311-330` 同构）、free bits（`networks.py:286-287` `torch.clip(min=free)`，等价 maximum）、unimix 0.01（`networks.py:165` + `configs.yaml:61`）、GRU update 门偏置 $-1$（`networks.py:743,766`）、slow critic rate 0.02/every 1 + 正则式（`configs.yaml:52` + `models.py:328-330,435-441`）、reward/critic 输出零初始化（`configs.yaml:52,54` `outscale: 0.0` + `tools.py:909-919` uniform_weight_init(0) → limit=0）、5-95 分位 EMA(alpha 0.01) + `clip(min=1.0)`（`models.py:11-26`）、熵 3e-4（`configs.yaml:50`）、$\lambda=0.95$/discount 0.997（`configs.yaml:76-77`）。主要差异：

1. **缺 replay critic 项（$\beta_{\text{repval}}=0.3$ 整件没有）**：`models.py` 的 `ImagBehavior._train`（L290-349）只在想象轨迹上训 critic，全仓无 repval/replay value 损失。相对 Nature 版少一件稳定化手术——论文明说该项针对"奖励难预测的环境"（L372-374），复现 Minecraft/难奖励域时是关键缺口。
2. **Reinforce 分支的优势未做回报归一化**：`models.py:404-421`——`reward_EMA` 算出的 `normed_target/normed_base` 只喂给 `dynamics` 分支（L415-416）；`reinforce` 分支用原始 `(target - value).detach()`（L417-421），**绕过了 5-95 百分位归一化**。而 crafter/atari100k/minecraft 配置全用 `imag_gradient: reinforce`（`configs.yaml:130,140,168`）。官方码对所有域统一用归一化优势乘 logp（`agent.py:408-414`）。这直接破坏"固定熵系数跨奖励尺度"的机制——**"漏一件手术就翻车"的活证据**。
3. **梯度路径按域切换 vs 统一 Reinforce**：torch 默认 DMC 用 `imag_gradient: 'dynamics'`（直通梯度回传世界模型，`configs.yaml:79`），离散域用 reinforce——这是 v1/v2 时代做法；官方 Nature 码全域统一 Reinforce（`imag_loss` 内只有 `logpi * adv` 一条路径）。
4. **优化器/数值体制不同**：torch 三个 Adam（model 1e-4 / actor 3e-5 / critic 3e-5，`configs.yaml:50-52,69`）+ 梯度范数裁剪（100/1000），precision 32（`configs.yaml:18`）；官方单一优化器 lr 4e-5 + **LaProp**（先 RMS 再 momentum，`agent.py:358-360`）+ $\epsilon=10^{-20}$ + **AGC 0.3**（`agent.py:344-346,358`；`configs.yaml:87`），全程 bfloat16（`configs.yaml:74`）。论文 L1039-1042 专门解释 LaProp 替代 Adam 的动机（允许极小 eps、避免偶发不稳）。
5. **twohot 实现路径不同**：torch `DiscDist`（`tools.py:452-506`）——目标先 symlog、在 $[-20,20]$ **线性等距** 255 桶上编码、读出 `symexp(sum(p*b))`（先加权平均再逆变换，L469-471）；官方——桶本身 symexp 间隔、目标原空间编码、读出 `sum(p*symexp_bins)` 且带对称求和（`outs.py:285-309`）。两者对同一标量给出的软标签与读出值并不逐位相等（变换与求均值的顺序不同），初始"预测严格为 0"的保证也只有官方实现显式处理。
6. **架构/规模代差**：标准 LayerNorm GRUCell（`networks.py:742-768`）vs Block GRU；无 12M-400M 档体系（各域手写 deter 512 或 4096，`configs.yaml:34,122,160`）；无 replay latent 存取（每 batch 从头 observe）；`kl 损失聚合`：torch `model_loss = sum(scaled) + kl_loss` 其中 `kl_loss = dyn_scale*dyn + rep_scale*rep` 且 **dyn_scale=0.5**（`configs.yaml:57`，v1 论文值）vs 官方 dyn 1.0。

---

## ③ 复现关键点清单（漏了会翻车的 checklist）

合并 WM1_f1 §WM1-1.5 歧义审计（4 个 PARTIAL 已全部落地）+ 本次走读新发现。重写实现时逐条对钩：

**世界模型侧**

- [ ] free bits：`max(KL, 1)` 分别作用于 dyn/rep 两支（不是先加权再裁剪）；`rssm.py:127-129`。
- [ ] KL 权重选版本：Nature 版 `dyn 1.0 / rep 0.1`；2023 v1 是 `0.5 / 0.1`。别混。
- [ ] unimix 1% 共**三处**：RSSM 后验、RSSM 先验、离散 actor（`configs.yaml:91,100`）。WM1_f1 只审计到前两处；漏 actor 那处，离散域策略可能塌成确定性。
- [ ] 向量观测两端 symlog：encoder 输入 squish（`rssm.py:218-219`）+ decoder symlog_mse 目标（`rssm.py:299`）。
- [ ] continue 目标乘 $1-1/\text{horizon}$（contdisc，`agent.py:175-176`），想象折扣处就不再乘 $\gamma$（`agent.py:401`）——二选一，别双重折扣。
- [ ] RSSM 动作输入幅值裁剪 `action /= max(1, |action|)`（`rssm.py:137`）。
- [ ] Block GRU 的 update 门偏置 $-1$（`rssm.py:157`）。

**标量预测侧**

- [ ] twohot 桶：255 个、symexp 间隔、范围 $\pm(\mathrm{e}^{20}-1)$；目标在原空间编码（官方路径）或 symlog 空间线性桶（v1/torch 路径），选定一种并前后一致。
- [ ] 读出用正负半轴对称求和（`outs.py:285-309`），否则初始预测非零、与零初始化的配合失效。
- [ ] reward 头与 critic 头输出层权重**零初始化**（`outscale: 0.0`），其余头不是零（policy 0.01、decoder/cont 1.0）。

**critic 侧（WM1_f1 PARTIAL 主坑）**

- [ ] replay critic 损失 $\beta_{\text{repval}}=0.3$：bootstrap 用想象起点的 $\lambda$-return（`agent.py:222`），对 replay 真实奖励再算 $\lambda$-return；`disc = 1-1/333` 固定、mask 用 `~is_last`（`agent.py:464-471`）。
- [ ] EMA 慢 critic：每步 2% 混合（rate 0.02/every 1）；**只作正则项（权重 1.0），$\lambda$-return 用当前 critic**（slowtar=False）。装成"硬拷贝目标网络算 return"就偏了。
- [ ] $\lambda$-return 区分 term（终止，停折扣）与 last（截断，停 $\lambda$ 混合）两种 mask（`agent.py:482-490`）。

**actor 侧**

- [ ] 归一化：$S=\operatorname{EMA}_{0.99}(\operatorname{Per}(R,95)-\operatorname{Per}(R,5))$，分母 $\max(1,S)$，**只除不减**；离散/连续统一 Reinforce；熵系数固定 $3\times 10^{-4}$。
- [ ] torch 复刻的教训：reinforce 路径若绕过归一化（`models.py:417-421`），稀疏/大奖励域的探索-利用平衡即坏——归一化必须在**所有**策略梯度路径上生效。

**训练体制侧**

- [ ] 单优化器 lr $4\times 10^{-5}$ + LaProp（RMS 在前 momentum 在后、$\epsilon=10^{-20}$）+ AGC 0.3 + warmup 1000（`agent.py:342-379`）；bfloat16。
- [ ] train_ratio 语义 = 每环境步重放样本步数，触发式 `Ratio(train_ratio/(B*T))`（`train.py:24-25`）。
- [ ] replay_context=1 + latent 存取（存 `deter/stoch` 到 buffer 并回填）——不做也能训，但与官方轨迹截断行为不同。
- [ ] 想象起点：默认拿 batch 内**全部** posterior 状态（`imag_last: 0` → K=T，`agent.py:189-191`），horizon 15。

## ④ 命令草案：CPU 冒烟与 GPU 正式训练

### CPU 冒烟（本机 2026-07-16 实测通过）

环境：Python 3.13 venv；`pip install jax==0.6.2 elements ninjax portal granular optax einops chex ruamel.yaml scope colored_traceback jaxtyping tqdm av`（+ 真环境再 `pip install crafter`）。

```sh
# 冒烟 1：dummy 任务，零外部环境依赖，约 2 分钟
python dreamerv3/main.py --configs debug --task dummy_disc \
  --run.steps 300 --logdir ~/logdir/smoke

# 冒烟 2：真环境 crafter（debug 叠加在 crafter 块之后）
python dreamerv3/main.py --configs crafter debug \
  --run.steps 200 --logdir ~/logdir/smoke_crafter
```

实测输出：两者均正常打印 `train/loss/{dyn,rep,rew,con,value,policy,repval,image}` 全项、写 `metrics.jsonl`/`scores.jsonl`、存 checkpoint；CPU 上 `fps/train` 约 55-58、`fps/policy` 约 6-9。`debug` 块已把网络缩到玩具级（deter 8/units 8/bins 5，`configs.yaml:204-220`），故仅验证管线、不学出策略。

**实测版本坑**（复现环境必读）：

- 官方口径 Python 3.11 + `jax[cuda12]==0.4.33`（requirements.txt）；
- 本机 Python 3.13 无法装 `numpy<2` 与 jaxlib 0.4.33（无 cp313 轮子）——用 **jax 0.6.2 + numpy 2.x** 可跑（numpy<2 只有 DMLab/MineRL 需要）；
- **jax >= 0.7 不可用**：`jax.jit` 移除了多位置参数签名，`embodied/jax/transform.py:56`（`jax.jit(fn, arg_shardings, params_sharding, static_argnums, None)`）直接 `TypeError`。可用上界实测为 0.6.x。

### GPU 正式训练草案（dm-051 单卡 RTX 5080 16G 口径）

论文口径是单张 A100（40G+）跑默认 200M 档；16G 卡建议降档起步（bfloat16 默认开）：

```sh
# 首选：Crafter（论文标准小域，1.1M 步，单环境）
python dreamerv3/main.py --configs crafter size50m \
  --logdir ~/logdir/crafter_50m

# DMC 本体感受（最快最省显存，自带 size1m；论文称 control suites 12M 档即达 200M 同性能）
python dreamerv3/main.py --configs dmc_proprio --logdir ~/logdir/dmcp
python dreamerv3/main.py --configs dmc_proprio size12m --logdir ~/logdir/dmcp_12m

# DMC 视觉（不叠 size 会用 200M 档，16G 大概率吃紧，显式降档）
python dreamerv3/main.py --configs dmc_vision size12m --logdir ~/logdir/dmcv_12m

# Atari100k（数据效率基准，1.1e5 步）
python dreamerv3/main.py --configs atari100k size50m --logdir ~/logdir/atari100k_50m
```

操作要点：OOM 时依次 `size25m` → `--batch_size 8`（README:107 也建议以 `--batch_size 1` 排查 OOM）；算力换数据效率用 `--run.train_ratio`（越大越省交互越费卡）；曲线看 `scores.jsonl` 的 `episode/score` 或 scope viewer。**待上机核验**：RTX 5080 为 Blackwell（sm_120），钉版 `jax[cuda12]==0.4.33`（2024-09 轮子）大概率不含该架构支持；建议直接 `pip install "jax[cuda12]==0.6.2"`（与本机 CPU 实测同版），保持 <0.7 上界。各档显存占用未实测，上表为保守起步建议而非结论。

## ⑤ 存疑待核表

| # | 事项 | 现状 | 待核动作 |
|---|---|---|---|
| 1 | 论文 Table 3 的 12M 档 "Recurrent units (8d)" 印作 **1024**，码 `size12m` 为 **2048**（`configs.yaml:126`） | 已核 PDF 第 20 页原表确为 1024；$8d$ 规则与代码互证 2048，判论文排版笔误 | 可在 danijar/dreamerv3 提 issue 求证 |
| 2 | torch 复刻缺 repval：2023 arXiv v1 论文是否本就无 $\beta_{\text{repval}}=0.3$（即 torch 是"忠实复刻旧版"还是"漏抄"） | v1 原文不在本地池，未核 | 拉 arXiv v1 版 PDF 比对 critic 节 |
| 3 | torch `reinforce` 分支绕过 RewardEMA 归一化（`models.py:417-421`）是复刻偏差还是老官方码亦如此 | 老版官方码（v1 时代 tag）未克隆 | 克隆 danijar/dreamerv3 的 2023 旧 tag 核对 actor loss |
| 4 | RTX 5080（Blackwell）与 `jax[cuda12]` 各版本兼容性、各 size 档 16G 显存占用 | 未上机；仅 CPU 实测 jax 0.6.2 可跑、0.10 不可跑 | dm-051 上冒烟 `--configs crafter debug --jax.platform cuda` 后逐档试 |
| 5 | `dmc_vision` 块不改档即用 200M 默认档，单卡 16G 可行性 | 论文用 A100；16G 未验 | 上机实测，不行则按 §④ 降档 |
| 6 | `repl_loss` 中 `slowtar: False` 时 replay $\lambda$-return 亦用当前 critic（`agent.py:460-466`），与论文 L375-377 的表述粒度不完全对齐（论文未指明 replay 侧 tarval 用哪个网络） | 码上确定：与想象侧同为当前 critic | 无需动作，以码为准记录 |

---

### 附：WM1_f1 四个 PARTIAL 的最终答案（一句话版）

1. **replay critic 权重 0.3 落点**：`configs.yaml:86` `loss_scales.repval: 0.3`，损失在 `agent.py:449-479`（`repl_loss`），bootstrap 用想象起点 $\lambda$-return（`agent.py:222`）。
2. **EMA target critic**：`utils.py:94-127` `SlowModel`，rate 0.02/every 1（`configs.yaml:110`），仅作正则（权重 1.0，`agent.py:420-422`），$\lambda$-return 用当前 critic。
3. **零初始化**：`nets.py:250-251` 权重初始化乘 `outscale`；`configs.yaml:98,101` 中 reward 头与 critic 头 `outscale: 0.0`（配套 twohot 对称求和 `outs.py:285-309`）。
4. **分类混 1% 均匀**：`outs.py:210-217`（Categorical 的 unimix 混合），三处生效——RSSM 先验/后验（`rssm.py:173-176`）+ 离散 actor（`configs.yaml:100`）。
5. **（规模档）**：`configs.yaml:120-153` 七档 size1m-size400m；论文 Table 3（p20）12M-400M 六档，规则 $d$/$8d$/$d/16$；defaults=200M；XS-XL 是 v1 旧命名。
