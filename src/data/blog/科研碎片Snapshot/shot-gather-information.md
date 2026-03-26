---
title: "How Much Information Does a Shot Gather Contain? / 一个共炮点道集蕴含了多少信息"
author: zzzzswh
pubDatetime: 2026-03-25
description: "Some animations to help you understand what a shot gather is doing from the perspective of the full wavefield./ 一些动画可以帮助你理解一个共炮点道集在震源视图下的工作原理。"
categories: ["科研碎片Snapshot"]
tags:
  - geophysics
  - seismic
  - visualization
  - deepwave
  - 中文
  - English
featured: true
date: 2026-03-25
comments: true
---

## English

A shot gather is one of the most fundamental data representations
in seismic exploration, but how much information does it actually contain?

I made a small animation to help you intuitively feel the information content of a shot gather.

In the lower left, I plotted the wavefield animation within the subsurface cross-section, with the source and receiver positions marked above.
In the upper left, the data received by each trace scrolls in real time, with its horizontal axis aligned to the x-axis of the medium.
On the right is the complete shot gather, where the red line indicates the current time step.
This simulation uses the acoustic wave equation — P-waves only.

<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model01.mp4" type="video/mp4" />
</video>

This animation helped me understand what a shot gather is really doing from the perspective of the full wavefield.
The wavefield evolving over time is shown in the lower left — the receivers are essentially sampling the full wavefield at certain spatial locations.
In other words, a shot gather is a spatially sparse, temporally continuous sampling of the wavefield.

In 2D, we can express this formally. The full wavefield is $\mathbf{u}(x, z, t)$, and the data recorded by the receivers is:

$$
d(x_r, t) = \mathbf{u}(x_r,\, z_0,\, t), \quad x_r \in \{x_1, x_2, \dots, x_N\}
$$

where $x_r$ is the horizontal position of each receiver, $z_0$ is the depth of the receivers (usually the surface), and $N$ is the number of receivers.

This also assumes that the receivers are distributed along a single horizontal line — which in reality is never exactly the case.

So in truth, we only know sampled data along a single line, and trying to reconstruct the full subsurface cross-section from that is nearly impossible.

That's why we increase the fold — vary the source positions — hoping to extract more information through the constraints of the wave equation.

**What if we make the medium more complex?**

Say, an anticline?

<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model02.mp4" type="video/mp4" />
</video>

What about adding another interface?

<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model03.mp4" type="video/mp4" />
</video>

What if we use elastic waves instead?

<video controls width="100%">
  <source src="/assets/videos/Elastic_wave_sim_model03_Vz.mp4" type="video/mp4" />
</video>

And that's the real challenge.

---

## 中文

共炮点道集是地震勘探中最基本的数据表示之一，
但它究竟蕴含了多少信息？

我做了一个小动画，可以直观的感受到共炮点道集所蕴含的信息量。

我在左边下面画出了地下截面里的波场动画，上面标出了震源和检波器的位置。  
在左边上面滚动播放的是每一道接受到的数据，同时横轴和介质的x轴对齐。  
在右边是整个共炮点道集，红线代表当前时刻。  
这里用的是声波方程的模拟，只有P波。
<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model01.mp4" type="video/mp4" />
</video>    

这个动画让我从整个波场的视角去理解共炮点道集在做什么。  
整个波场随时间的变化在左下角画出来了，检波器实际上是在对整个波场的某些空间位置上做了采样。  
这是说，共炮点道集实际上是一种空间稀疏、时间连续的波场采样。

二维情况，用公式说，完整波场为 $\mathbf{u}(x, z, t)$，
而检波器记录的数据为：

$$
d(x_r, t) = \mathbf{u}(x_r,\, z_0,\, t), \quad x_r \in \{x_1, x_2, \dots, x_N\}
$$

其中 $x_r$ 是检波器的水平位置，$z_0$ 是检波器所在的深度（通常为地表），
$N$ 是检波器的数量。  

这里还假设了检波器是分布在统一横坐标下，这在现实里也是不可能的。  

所以实际上我们只是知道了一条线的采样数据，想要还原出整个截面的信息，近乎是不可能的。  

所以我们做的是增加覆盖次数，增加震源位置的变化，渴望用波动方程的规律得到更多的信息。

**如果我们让介质更复杂一些呢？**  

比如说一个背斜？
<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model02.mp4" type="video/mp4" />
</video>   

要是再加一个界面？  
<video controls width="100%">
  <source src="/assets/videos/P_wave_sim_model03.mp4" type="video/mp4" />
</video>  

要是是弹性波呢？  
<video controls width="100%">
  <source src="/assets/videos/Elastic_wave_sim_model03_Vz.mp4" type="video/mp4" />
</video>  

这就是问题所在。  


## Code / 代码

我给deepwave写了个简单的API接口可以参考一下：
I wrote a simple API interface for deepwave, you can refer to it: 

```python
# deepwave_easy_api.py
"""
Deepwave Easy API (Refactored)
==============================
基于物理坐标 (米) 和纯面向对象架构的 2D 地震波正演封装。
核心设计理念：数据（物理模型、观测系统）与计算（仿真器）彻底分离。
"""

import numpy as np
import torch
import deepwave
from deepwave import elastic, scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# 0. 信号工具 (Signal Utilities)
# =============================================================================


def ricker(f0, nt, dt, delay=None):
    """
    生成 Ricker 子波。

    参数
    ----------
    f0 : float, 主频 (Hz)
    nt : int, 采样点数
    dt : float, 采样间隔 (s)
    delay : float, 延迟时间 (s)，默认 1.5/f0
    """
    if delay is None:
        delay = 1.5 / f0
    t = np.arange(nt) * dt - delay
    a = (np.pi * f0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


# =============================================================================
# 1. 数据对象层 (Data Objects) - 只存储信息，不处理计算逻辑
# =============================================================================

class PhysicalModel:
    """
    物理地质模型。仅持有介质参数和网格间距，不关心具体的观测系统和网格索引。
    """

    def __init__(self, vp, vs=None, rho=None, dh=10.0, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.dh = float(dh)

        # 统一转为张量并加载到指定设备
        self.vp = self._to_tensor(vp)
        self.vs = self._to_tensor(
            vs) if vs is not None else torch.zeros_like(self.vp)
        self.rho = self._to_tensor(
            rho) if rho is not None else torch.ones_like(self.vp) * 2000.0

        self.nz, self.nx = self.vp.shape
        # [left, right, bottom, top] 用于 matplotlib extent (注意 Z 轴向下)
        self.extent = [0, (self.nx - 1) * self.dh, (self.nz - 1) * self.dh, 0]

    def _to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device, dtype=torch.float32)
        return torch.tensor(data, device=self.device, dtype=torch.float32)

    @property
    def lame_parameters(self):
        """返回 Deepwave 弹性波仿真所需的 Lamé 参数"""
        return deepwave.common.vpvsrho_to_lambmubuoyancy(self.vp, self.vs, self.rho)

    def to_cpu(self):
        """将 Vp 提取到 CPU 方便绘图"""
        return self.vp.detach().cpu().numpy()


class Acquisition:
    """
    观测系统配置。仅持有物理坐标（米）、时间步长和子波。
    """

    def __init__(self, src_x, src_z, rec_x, rec_z, wavelet, dt):
        # 将标量深度自动广播匹配所有的 x 坐标
        self.src_x = np.atleast_1d(src_x)
        self.src_z = np.broadcast_to(np.atleast_1d(src_z), self.src_x.shape)

        self.rec_x = np.atleast_1d(rec_x)
        self.rec_z = np.broadcast_to(np.atleast_1d(rec_z), self.rec_x.shape)

        self.wavelet = wavelet  # 形状 (nt,)
        self.dt = float(dt)
        self.nt = len(wavelet)
        self.n_shots = len(self.src_x)
        self.n_recs = len(self.rec_x)


# =============================================================================
# 2. 仿真引擎层 (Simulator) - 处理所有转换、PML 边界和计算逻辑
# =============================================================================

class SeismicSimulator:
    """
    仿真大脑：负责将 PhysicalModel 和 Acquisition 结合，计算网格索引，调用 Deepwave 核心引擎。
    """

    def __init__(self, model: PhysicalModel, acquisition: Acquisition):
        self.model = model
        self.acq = acquisition
        self.device = model.device

    def _get_grid_locations(self, x_phys, z_phys):
        """将物理坐标(m)精确映射为 Deepwave 所需的网格索引张量"""
        iz = np.round(z_phys / self.model.dh).astype(np.int64)
        ix = np.round(x_phys / self.model.dh).astype(np.int64)

        # 边界安全裁剪，防止坐标越界
        iz = np.clip(iz, 0, self.model.nz - 1)
        ix = np.clip(ix, 0, self.model.nx - 1)

        # 构造 Deepwave 需要的形状 (n_shots, n_locations, 2)
        # 这里默认每个炮都使用相同的检波器排列（固定排列）
        locs = torch.zeros(self.acq.n_shots, len(
            ix), 2, dtype=torch.long, device=self.device)
        locs[..., 0] = torch.from_numpy(iz)
        locs[..., 1] = torch.from_numpy(ix)
        return locs

    def run_acoustic(self, pml_width=30, free_surface=True, accuracy=4, **kwargs):
        """
        运行声波仿真 (只使用 Vp)。
        **kwargs 用于接收 forward_callback 等高级参数。
        """
        src_locs = self._get_grid_locations(self.acq.src_x, self.acq.src_z)
        rec_locs = self._get_grid_locations(self.acq.rec_x, self.acq.rec_z)

        # 扩展子波维度以匹配 (n_shots, n_sources_per_shot, nt)
        wav = torch.tensor(self.acq.wavelet, device=self.device).reshape(
            1, 1, -1).expand(self.acq.n_shots, 1, -1)

        # 自由地表处理
        pml = [0, pml_width, pml_width,
               pml_width] if free_surface else [pml_width] * 4

        out = scalar(
            self.model.vp, self.model.dh, self.acq.dt,
            source_amplitudes=wav,
            source_locations=src_locs,
            receiver_locations=rec_locs,
            pml_width=pml,
            accuracy=accuracy,
            **kwargs
        )
        return out[-1]  # 返回 receiver_amplitudes，形状为 (n_shots, n_receivers, nt)

    def run_elastic(self, pml_width=30, free_surface=True, src_comp='y', rec_comp='y', accuracy=4, **kwargs):
        """
        运行弹性波仿真。
        """
        lamb, mu, buoy = self.model.lame_parameters
        src_locs = self._get_grid_locations(self.acq.src_x, self.acq.src_z)
        rec_locs = self._get_grid_locations(self.acq.rec_x, self.acq.rec_z)
        wav = torch.tensor(self.acq.wavelet, device=self.device).reshape(
            1, 1, -1).expand(self.acq.n_shots, 1, -1)

        pml = [0, pml_width, pml_width,
               pml_width] if free_surface else [pml_width] * 4

        src_args = {f'source_amplitudes_{src_comp}': wav,
                    f'source_locations_{src_comp}': src_locs}
        rec_args = {f'receiver_locations_{rec_comp}': rec_locs}

        out = elastic(
            lamb, mu, buoy, self.model.dh, self.acq.dt,
            **src_args, **rec_args,
            pml_width=pml, accuracy=accuracy,
            **kwargs
        )

        idx = -2 if rec_comp in ['y', 'z'] else -1
        return out[idx]

    def record_wavefield(self, mode='acoustic', pml_width=30, free_surface=True, freq_interval=5):
        """
        录制波场传播过程。

        参数
        ----------
        mode : str, 'acoustic' 或 'elastic'
        freq_interval : int, 采样间隔（帧数）。减小该值能提升流畅度，但会增加内存占用。

        返回
        -------
        torch.Tensor, 录制的波场快照，形状为 (n_frames, n_shots, nz, nx)
        """
        wavefield_history = []

        def callback(state):
            # 获取当前波场，为了节省显存，直接 clone 并放到 CPU 上
            wf = state.get_wavefield("wavefield_0")[0].cpu().clone()
            wavefield_history.append(wf)

        if mode == 'acoustic':
            self.run_acoustic(pml_width=pml_width, free_surface=free_surface,
                              forward_callback=callback,
                              callback_frequency=freq_interval)
        elif mode == 'elastic':
            self.run_elastic(pml_width=pml_width, free_surface=free_surface,
                             forward_callback=callback,
                             callback_frequency=freq_interval)
        else:
            raise ValueError("mode must be 'acoustic' or 'elastic'")

        return torch.stack(wavefield_history)


# =============================================================================
# 3. 绘图与可视化工具 (Visualization Utilities)
# =============================================================================

def plot_simulation(simulator: SeismicSimulator, data=None):
    """
    一键可视化模型、观测系统和生成的共炮点道集（Shot Gather）。
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes

    # --- 1. 速度模型与几何 ---
    m = simulator.model
    a = simulator.acq
    im = ax1.imshow(m.to_cpu(), extent=m.extent, cmap='terrain', aspect='auto')

    # 标出震源和检波器
    ax1.scatter(a.src_x, a.src_z, c='red', marker='*',
                s=150, label='Sources', edgecolors='black')
    ax1.scatter(a.rec_x, a.rec_z, c='blue', marker='v',
                s=20, alpha=0.5, label='Receivers')

    ax1.set_title("Velocity Model (Vp) & Survey Geometry")
    ax1.set_xlabel("Lateral Position (m)")
    ax1.set_ylabel("Depth (m)")
    ax1.legend(loc='lower left')
    plt.colorbar(im, ax=ax1, label='Velocity (m/s)')

    # --- 2. 炮集 (可选) ---
    if data is not None:
        # 取第一炮的数据，转移到 numpy
        d = data[0].detach().cpu().numpy() if isinstance(
            data, torch.Tensor) else data[0]
        # 使用 98% 分位数作为显示裁剪阈值（Clip）
        clip = np.percentile(np.abs(d), 98)

        ax2.imshow(d.T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
                   extent=[a.rec_x[0], a.rec_x[-1], a.nt * a.dt, 0])
        ax2.set_title("Synthetic Shot Gather (Shot 0)")
        ax2.set_xlabel("Receiver Position (m)")
        ax2.set_ylabel("Time (s)")

    plt.tight_layout()
    plt.show()


def save_wavefield_animation(wavefields, model: PhysicalModel, filename="wavefield.gif", shot_idx=0):
    """
    将录制的波场张量转化为 GIF 动画并保存。

    参数
    ----------
    wavefields : torch.Tensor, 形状 (n_frames, n_shots, nz, nx)
    model : PhysicalModel, 用于获取坐标范围
    shot_idx : int, 选择录制哪一炮的波场
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 提取选定炮的波场数据
    wf_data = wavefields[:, shot_idx, :, :]

    # 选取整个动画序列绝对值最大值的 10% 作为颜色条阈值
    vlimit = torch.max(torch.abs(wf_data)) * 0.1

    im = ax.imshow(wf_data[0].numpy(), cmap='RdBu', aspect='auto',
                   extent=model.extent, vmin=-vlimit, vmax=vlimit)

    ax.set_title(f"Wavefield Propagation (Shot {shot_idx})")
    ax.set_xlabel("Lateral Position (m)")
    ax.set_ylabel("Depth (m)")
    plt.colorbar(im, label="Amplitude")

    def update(frame):
        im.set_array(wf_data[frame].numpy())
        return [im]

    print(f"开始渲染动画，共 {len(wf_data)} 帧...")
    ani = animation.FuncAnimation(fig, update, frames=len(wf_data),
                                  interval=50, blit=True)

    # 使用 pillow 保存 gif
    ani.save(filename, writer='pillow')
    print(f"✅ 动画已成功保存至: {filename}")
    plt.close()


# =============================================================================
# 测试运行块 (Example Usage)
# =============================================================================
if __name__ == "__main__":
    # 1. 建立具有两层的地质模型
    vp_grid = np.ones((150, 300)) * 2000.0
    vp_grid[80:, :] = 3000.0  # 下层高速
    model = PhysicalModel(vp=vp_grid, dh=5.0)  # 网格间距 5m

    # 2. 生成 Ricker 子波并配置观测系统
    dt = 0.001
    wav = ricker(f0=25, nt=600, dt=dt)

    acq = Acquisition(
        src_x=[750], src_z=[10],                   # 地表中点激发
        rec_x=np.linspace(0, 1500, 150), rec_z=10,  # 全排列接收
        wavelet=wav, dt=dt
    )

    # 3. 初始化仿真器
    sim = SeismicSimulator(model, acq)

    # [选项 A] 生成常规地震数据，用于 AI 训练
    print("正在计算共炮点道集...")
    shot_gather = sim.run_acoustic(pml_width=30, free_surface=True)
    plot_simulation(sim, shot_gather)

    # [选项 B] 录制并生成波场动画
    print("正在录制地下波场...")
    wf_history = sim.record_wavefield(mode='acoustic', freq_interval=6)
    save_wavefield_animation(wf_history, model, filename="wavefield_demo.gif")
```

简单的使用示例：
```python
# Forward/001_gen_shots.py
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import deepwave_easy_api as dwe
import deepwave
import segyio
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

DH = 1.249  # m


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    datadir = "../data/"

    f0 = 15
    time = 4000  # ms
    dt = 0.0002

    nt = round(time / dt)

    dsrc_x = 25
    drec_x = 12.5

    data = np.load(os.path.join(datadir, "velmodel.npz"))
    vp = data["vp"]
    vs = data["vs"]
    dens = data["dens"]
    dh = data["dh"]

    print("max_vp", vp.max())
    print("min_vp", vp.min())

    nz, nx = vp.shape

    src_x = np.arange(0, dh * (nx - 1), dsrc_x)
    src_z = np.ones_like(src_x) * 10

    n_shots = len(src_x)
    print("dsrc_x:", dsrc_x)
    print(f"number of sources: {n_shots}")

    rec_x = np.arange(0, dh * (nx - 1), drec_x)
    rec_z = np.ones_like(rec_x) * 10
    n_receivers = len(rec_x)
    print("drec_x:", drec_x)
    print(f"number of receivers: {n_receivers}")

    # output src positions
    data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data"))
    os.makedirs(data_dir, exist_ok=True)
    geom_path = os.path.join(data_dir, "geometry.npz")
    np.savez(geom_path, src_x=src_x, src_z=src_z,
             rec_x=rec_x, rec_z=rec_z, dt=dt, nt=nt, f0=f0, dh=dh)
    print(f"Saved geometry to {geom_path}")

    model = dwe.VelocityModel(vp, vs, dens, dh=dh)

    wavelet = dwe.ricker(f0, nt, dt)

    shots_dir = os.path.join(data_dir, "shots")
    os.makedirs(shots_dir, exist_ok=True)

    for i in tqdm(range(n_shots), desc="Generating shots", ncols=100):
        geom = dwe.Geometry(model, src_x[i], src_z[i], rec_x, rec_z, wavelet)
        shot = dwe.elastic2d(model, geom, dt=dt,
                             pml_width=100, pml_freq=f0, accuracy=4)
        np.savez(os.path.join(shots_dir, f"shot_{i:04d}.npz"), shot=shot)
        print(f"Saved shot {i} to {shots_dir}", end="\r", flush=True,)


if __name__ == '__main__':
    main()
```  

这个为勘探地震和物理意义提供了遍历。  
This provides a comprehensive exploration of the field of seismic exploration and physical meaning.

画图代码：
```python
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import deepwave
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. 定义物理网格与时间 ---
nz, nx = 150, 300
dh = 5.0
dt = 0.001
nt = 1000
freq = 25

# --- 2. 创建复杂地下介质 (Elastic Model: vp, vs, rho) ---
# 初始化背景参数：层1 ($v_p=2000$, $v_s=1150$, $\rho=2000$)
vp = torch.ones(nz, nx, device=device) * 2000.0
vs = torch.ones(nz, nx, device=device) * 1150.0 
rho = torch.ones(nz, nx, device=device) * 2000.0

x = torch.arange(nx, device=device, dtype=torch.float32)
z = torch.arange(nz, device=device, dtype=torch.float32).unsqueeze(1)


# ----------------------------------------------------
# 界面 2: 经典的背斜构造 (Anticline)
# 使用高斯曲线在中间形成一个凸起
# 层3高速: ($v_p=3500$, $v_s=2020$, $\rho=2600$)
# ----------------------------------------------------
curve2 = 80.0 - 30.0 * torch.exp(-((x - nx/2) ** 2) / (2 * 20.0 ** 2))
mask2 = z >= curve2
vp[mask2] = 3500.0
vs[mask2] = 2020.0
rho[mask2] = 2600.0


# --- 3. 创建震源与检波器 (Sources & Receivers) ---
n_shots = 1

# 震源：放在地表中心 (z=0, x=150)
source_locations = torch.zeros(n_shots, 1, 2, dtype=torch.long, device=device)
source_locations[0, 0, 0] = 0
source_locations[0, 0, 1] = 150

# 检波器：铺满整个地表，共 300 个 (z=0, x=0~299)
receiver_locations = torch.zeros(n_shots, nx, 2, dtype=torch.long, device=device)
receiver_locations[0, :, 0] = 0                 
receiver_locations[0, :, 1] = torch.arange(nx)  

# 生成震源子波 [n_shots, n_sources_per_shot, nt]
wavelet = deepwave.wavelets.ricker(freq, nt, dt, 1.5/freq).to(device)
fz = wavelet.reshape(n_shots, 1, nt) # 垂直力

# --- 4. 准备录制 3D 波场 ---
wavefield_list = []

def record_callback(state):
    # 弹性波模型中不再是 "wavefield_0"，而是有多个分量(vy_0, vx_0, sigmayy_0等)
    # 我们动态提取包含 "vy" (垂直速度分量) 的键值，防止版本不同导致报错
    keys = list(state._wavefields.keys())
    if "vy_0" in keys:
        wf_name = "vy_0"
    elif "vy" in keys:
        wf_name = "vy"
    else:
        wf_name = keys[0] # 保底方案
        
    wf_current = state.get_wavefield(wf_name)[0].cpu().clone()
    wavefield_list.append(wf_current)

# --- 5. 运行弹性波正演模拟 ---
print("Running forward simulation (Elastic)...")

# 核心计算：转换为 Lamé 参数
lamb, mu, buoyancy = deepwave.common.vpvsrho_to_lambmubuoyancy(vp, vs, rho)

out = deepwave.elastic(
    lamb, mu, buoyancy, dh, dt,
    source_amplitudes_y=fz,           # 指定 Y 方向 (垂直) 震源振幅
    source_locations_y=source_locations, # 指定 Y 方向 (垂直) 震源位置
    receiver_locations_y=receiver_locations, # 指定 Y 方向 (垂直) 检波器位置
    pml_width=[0, 50, 50, 50], 
    pml_freq=freq,                    # 消除警告，提升 PML 吸收性能
    forward_callback=record_callback,
    callback_frequency=1
)
print("Forward simulation completed.")

# --- 6. 获取最终的张量数组 ---
# out[-2] 是 Y 方向 (垂直) 的接收数据，out[-1] 是 X 方向 (由于未设置，所以为空)
receiver_array = out[-2][0].cpu().numpy() # 取出 vz 检波器数据
wavefield_3d_array = torch.stack(wavefield_list).numpy()

print(f"检波器 vz 数据形状: {receiver_array.shape}")
print(f"地下 vz 波场数据形状: {wavefield_3d_array.shape}")


# =============================================================================
# --- 7. 绘制并保存：动态 AGC (每一帧重新归一化) + Colorbar ---
# =============================================================================
print("Preparing dynamic AGC multi-view animation...")

total_time = nt * dt
x_extent_m = nx * dh
y_extent_m = nz * dh

gather_data = receiver_array.T  # (nt, nx)

# 右侧：全局道均衡归一化 (静态)
max_amp_per_trace = np.max(np.abs(receiver_array), axis=1, keepdims=True)
max_amp_per_trace[max_amp_per_trace == 0] = 1e-10
gather_norm = (receiver_array / max_amp_per_trace).T

fig = plt.figure(figsize=(16, 9))
gs = gridspec.GridSpec(2, 2, height_ratios=[
                           1, 1.2], width_ratios=[2, 1.2], wspace=0.15)

ax_rec_slide = fig.add_subplot(gs[0, 0])
ax_wav = fig.add_subplot(gs[1, 0], sharex=ax_rec_slide)
ax_full = fig.add_subplot(gs[:, 1])

plt.setp(ax_rec_slide.get_xticklabels(), visible=False)

# ------------------------------------------
# A. 左上：检波器记录 (初始化)
# ------------------------------------------
window_steps = 40
window_time_thickness = window_steps * dt

im_rec_slide = ax_rec_slide.imshow(gather_data, aspect='auto', cmap='gray',
                                   extent=[0, x_extent_m, total_time, 0])
# 更新标题：Elastic (Vz)
ax_rec_slide.set_title(f"Sliding Window (Dynamic AGC) - Elastic (Vz)")
ax_rec_slide.set_ylabel("Time (s)")
ax_rec_slide.set_ylim(window_time_thickness, 0)

# 为滑窗道集添加 Colorbar
cbar_rec = fig.colorbar(im_rec_slide, ax=ax_rec_slide, pad=0.02)
cbar_rec.set_label('Amplitude', rotation=270, labelpad=15)

# ------------------------------------------
# B. 左下：模型背景 + 波场动态叠加 (初始化)
# ------------------------------------------
model_np = vp.cpu().numpy()
ax_wav.imshow(model_np, aspect='auto', cmap='terrain',
              extent=[0, x_extent_m, y_extent_m, 0])

# 【修复1】：将 alpha 改回 0.5，让底层的地质模型透出来
im_wav = ax_wav.imshow(wavefield_3d_array[0], aspect='auto', cmap='RdBu',
                       alpha=0.5, extent=[0, x_extent_m, y_extent_m, 0])
ax_wav.set_title("Wavefield Propagation (Dynamic AGC) - Elastic (Vz)")
ax_wav.set_ylabel("Depth (m)")
ax_wav.set_xlabel("Lateral Distance (m)")

ax_wav.scatter([150*dh], [0], c='red', marker='*', s=200,
               label='Source', edgecolors='black', zorder=10)
ax_wav.scatter(np.arange(nx)*dh, np.zeros(nx), c='blue',
               marker='v', s=10, alpha=0.5, label='Receivers', zorder=9)
ax_wav.legend(loc='lower left')

cbar_wav = fig.colorbar(im_wav, ax=ax_wav, pad=0.02)
cbar_wav.set_label('Wavefield Amp', rotation=270, labelpad=15)

# ------------------------------------------
# C. 右侧：完整道集 + 时间游标 (静态背景)
# ------------------------------------------
norm_clip = np.percentile(np.abs(gather_norm), 98)
im_full = ax_full.imshow(gather_norm, aspect='auto', cmap='gray',
                         vmin=-norm_clip, vmax=norm_clip,
                         extent=[0, x_extent_m, total_time, 0])
ax_full.set_title("Trace Normalized Full Record - Elastic (Vz)")
ax_full.set_xlabel("Lateral Distance (m)")
ax_full.set_ylabel("Time (s)")

time_line = ax_full.axhline(y=0, color='red', linewidth=2, zorder=10)

# ------------------------------------------
# 动画更新函数
# ------------------------------------------

def update(frame):
    current_time_s = frame * dt

    # --- 1. 动态归一化更新：波场 ---
    curr_wav = wavefield_3d_array[frame]
    im_wav.set_array(curr_wav)

    # 【修复2】：将 1e-5 的底线大幅降低到 1e-10，防止吃掉弹性波微小的振幅
    wav_max = max(np.percentile(np.abs(curr_wav), 99), 1e-10)
    # 取最大值的 50% 作为限幅，兼顾对比度和色彩鲜艳度
    wav_clip = wav_max * 0.5 
    im_wav.set_clim(-wav_clip, wav_clip)

    # --- 2. 动态归一化更新：滑窗道集 ---
    window_bottom = max(window_time_thickness, current_time_s)
    window_top = max(0, current_time_s - window_time_thickness)
    ax_rec_slide.set_ylim(window_bottom, window_top)

    idx_top = int(window_top / dt)
    idx_bottom = int(window_bottom / dt)

    if idx_bottom > idx_top:
        visible_gather = gather_data[idx_top:idx_bottom, :]
        # 同理，降低滑窗计算的底线限制
        rec_max = max(np.percentile(np.abs(visible_gather), 98), 1e-10)
        rec_clip = rec_max * 0.5
        im_rec_slide.set_clim(-rec_clip, rec_clip)

    # --- 3. 更新右侧红线 ---
    time_line.set_ydata([current_time_s, current_time_s])

    return [im_wav, time_line]


# ------------------------------------------
# 渲染与保存
# ------------------------------------------
ani = animation.FuncAnimation(fig, update, frames=nt, interval=30, blit=False)

mp4_filename = "Elastic_wave_sim_model03_Vz.mp4"
print(f"Saving animation to {mp4_filename}...")

try:
    ani.save(mp4_filename, writer=animation.FFMpegWriter(fps=30))
    print("✅ Animation saved successfully!")
except FileNotFoundError:
    print("\n❌ Error: ffmpeg not found.")

# plt.show()
```