---
title: "Easy API for Deepwave Forward Modeling/简化的Deepwave 正演API"
author: zzzzswh
pubDatetime: 2026-03-25
description: "Doing Forward Modeling more easily with Deepwave/更方便的用deepwave做正演" 
categories: ["科研碎片Snapshot"]
tags:
  - geophysics
  - seismic
  - deepwave
  - 中文
  - English
featured: true
date: 2026-02-25
comments: true
---

## English

[Deepwave](https://github.com/ar4/deepwave) is an incredibly powerful FWI package. It includes forward modeling modules for various wave equations and a ton of FWI-related features. The official website has plenty of tutorials, which is great for beginners. However, when I needed to do large-scale batch forward modeling (especially for generating massive synthetic datasets for AI training), I found the standard setup a bit cumbersome. 

My damn ~~perfectionism~~ struck again, so I wrote this wrapper. This is my ideal forward modeling API. In fact, I used this exact same object-oriented design philosophy—strictly separating the physical data from the computation engine—when writing my own Julia wave simulator, Fomo.jl. Hope you find it useful!

## 中文

[Deepwave](https://github.com/ar4/deepwave)是一个非常强大的FWI包，里面有多种方程的正演模块，还有很多FWI相关的模块支持非常多的功能。

官网上有非常多的教程，非常适合初学者，但是我在批量正演的时候还是有点麻烦，我那该死的~~完美主义~~又犯罪了，写了个API，这是我心目中完美的正演API，我自己写的Fomo.jl也是这样写的，希望你们喜欢：

```python
# Deepwave_easy_api.py
"""
Deepwave Easy API
Separates data (physical model, survey geometry) from computation (simulator).
"""

import numpy as np
import torch
import deepwave
from deepwave import elastic, scalar
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def ricker(f0, nt, dt, delay=None):
    """Generate a Ricker wavelet."""
    delay = delay or 1.5 / f0
    t = np.arange(nt) * dt - delay
    a = (np.pi * f0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


class PhysicalModel:
    """Geological model: holds medium parameters and grid spacing."""
    def __init__(self, vp, vs=None, rho=None, dh=10.0, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dh = float(dh)
        
        self.vp = self._to_tensor(vp)
        self.vs = self._to_tensor(vs) if vs is not None else torch.zeros_like(self.vp)
        self.rho = self._to_tensor(rho) if rho is not None else torch.ones_like(self.vp) * 2000.0
        
        self.nz, self.nx = self.vp.shape
        self.extent = [0, (self.nx - 1) * self.dh, (self.nz - 1) * self.dh, 0]

    def _to_tensor(self, data):
        return data.to(self.device, dtype=torch.float32) if isinstance(data, torch.Tensor) else torch.tensor(data, device=self.device, dtype=torch.float32)

    @property
    def lame_parameters(self):
        return deepwave.common.vpvsrho_to_lambmubuoyancy(self.vp, self.vs, self.rho)

    def to_cpu(self):
        return self.vp.detach().cpu().numpy()


class Acquisition:
    """Survey geometry: holds physical coordinates (m) and wavelet."""
    def __init__(self, src_x, src_z, rec_x, rec_z, wavelet, dt):
        self.src_x = np.atleast_1d(src_x)
        self.src_z = np.broadcast_to(np.atleast_1d(src_z), self.src_x.shape)
        
        self.rec_x = np.atleast_1d(rec_x)
        self.rec_z = np.broadcast_to(np.atleast_1d(rec_z), self.rec_x.shape)
        
        self.wavelet = wavelet
        self.dt = float(dt)
        self.nt = len(wavelet)
        self.n_shots = len(self.src_x)
        self.n_recs = len(self.rec_x)


class SeismicSimulator:
    """Engine combining Model and Acquisition to run Deepwave."""
    def __init__(self, model: PhysicalModel, acquisition: Acquisition):
        self.model = model
        self.acq = acquisition
        self.device = model.device

    def _get_grid_locations(self, x_phys, z_phys):
        """Map physical coords (m) to safe grid indices."""
        iz = np.clip(np.round(z_phys / self.model.dh).astype(np.int64), 0, self.model.nz - 1)
        ix = np.clip(np.round(x_phys / self.model.dh).astype(np.int64), 0, self.model.nx - 1)
        
        locs = torch.zeros(self.acq.n_shots, len(ix), 2, dtype=torch.long, device=self.device)
        locs[..., 0], locs[..., 1] = torch.from_numpy(iz), torch.from_numpy(ix)
        return locs

    def run_acoustic(self, pml_width=30, free_surface=True, accuracy=4, **kwargs):
        src_locs = self._get_grid_locations(self.acq.src_x, self.acq.src_z)
        rec_locs = self._get_grid_locations(self.acq.rec_x, self.acq.rec_z)
        wav = torch.tensor(self.acq.wavelet, device=self.device).reshape(1, 1, -1).expand(self.acq.n_shots, 1, -1)
        pml = [0, pml_width, pml_width, pml_width] if free_surface else [pml_width] * 4

        out = scalar(self.model.vp, self.model.dh, self.acq.dt,
                     source_amplitudes=wav, source_locations=src_locs,
                     receiver_locations=rec_locs, pml_width=pml, accuracy=accuracy, **kwargs)
        return out[-1] 

    def run_elastic(self, pml_width=30, free_surface=True, src_comp='y', rec_comp='y', accuracy=4, **kwargs):
        lamb, mu, buoy = self.model.lame_parameters
        src_locs = self._get_grid_locations(self.acq.src_x, self.acq.src_z)
        rec_locs = self._get_grid_locations(self.acq.rec_x, self.acq.rec_z)
        wav = torch.tensor(self.acq.wavelet, device=self.device).reshape(1, 1, -1).expand(self.acq.n_shots, 1, -1)
        pml = [0, pml_width, pml_width, pml_width] if free_surface else [pml_width] * 4
        
        src_args = {f'source_amplitudes_{src_comp}': wav, f'source_locations_{src_comp}': src_locs}
        rec_args = {f'receiver_locations_{rec_comp}': rec_locs}

        out = elastic(lamb, mu, buoy, self.model.dh, self.acq.dt,
                      **src_args, **rec_args, pml_width=pml, accuracy=accuracy, **kwargs)
        return out[-2 if rec_comp in ['y', 'z'] else -1]

    def record_wavefield(self, mode='acoustic', pml_width=30, free_surface=True, freq_interval=5):
        """Record wavefield propagation history."""
        wavefield_history = []
        
        def callback(state):
            wavefield_history.append(state.get_wavefield("wavefield_0")[0].cpu().clone())

        run_func = self.run_acoustic if mode == 'acoustic' else self.run_elastic
        run_func(pml_width=pml_width, free_surface=free_surface,
                 forward_callback=callback, callback_frequency=freq_interval)
        
        return torch.stack(wavefield_history)


def plot_simulation(simulator: SeismicSimulator, data=None):
    """Visualize velocity model, geometry, and synthetic shot gather."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes
    m, a = simulator.model, simulator.acq

    im = ax1.imshow(m.to_cpu(), extent=m.extent, cmap='terrain', aspect='auto')
    ax1.scatter(a.src_x, a.src_z, c='red', marker='*', s=150, label='Sources', edgecolors='black')
    ax1.scatter(a.rec_x, a.rec_z, c='blue', marker='v', s=20, alpha=0.5, label='Receivers')
    ax1.set(title="Velocity Model (Vp) & Geometry", xlabel="Lateral (m)", ylabel="Depth (m)")
    ax1.legend(loc='lower left')
    plt.colorbar(im, ax=ax1, label='Velocity (m/s)')

    if data is not None:
        d = data[0].detach().cpu().numpy() if isinstance(data, torch.Tensor) else data[0]
        clip = np.percentile(np.abs(d), 98)
        ax2.imshow(d.T, aspect='auto', cmap='gray', vmin=-clip, vmax=clip,
                   extent=[a.rec_x[0], a.rec_x[-1], a.nt * a.dt, 0])
        ax2.set(title="Synthetic Shot Gather (Shot 0)", xlabel="Receiver (m)", ylabel="Time (s)")

    plt.tight_layout()
    plt.show()


def save_wavefield_animation(wavefields, model: PhysicalModel, filename="wavefield.gif", shot_idx=0):
    """Save wavefield tensor as a GIF."""
    fig, ax = plt.subplots(figsize=(10, 6))
    wf_data = wavefields[:, shot_idx, :, :]
    vlimit = torch.max(torch.abs(wf_data)) * 0.1

    im = ax.imshow(wf_data[0].numpy(), cmap='RdBu', aspect='auto',
                   extent=model.extent, vmin=-vlimit, vmax=vlimit)
    ax.set(title=f"Wavefield Propagation (Shot {shot_idx})", xlabel="Lateral (m)", ylabel="Depth (m)")
    plt.colorbar(im, label="Amplitude")

    def update(frame):
        im.set_array(wf_data[frame].numpy())
        return [im]

    print(f"Rendering {len(wf_data)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=len(wf_data), interval=50, blit=True)
    ani.save(filename, writer='pillow')
    print(f"✅ Saved to: {filename}")
    plt.close()


if __name__ == "__main__":
    # 1. Build 2-layer model
    vp_grid = np.ones((150, 300)) * 2000.0
    vp_grid[80:, :] = 3000.0
    model = PhysicalModel(vp=vp_grid, dh=5.0)

    # 2. Setup acquisition
    dt = 0.001
    wav = ricker(f0=25, nt=600, dt=dt)
    acq = Acquisition(
        src_x=[750], src_z=[10], 
        rec_x=np.linspace(0, 1500, 150), rec_z=10, 
        wavelet=wav, dt=dt
    )

    # 3. Initialize & Run
    sim = SeismicSimulator(model, acq)

    # Option A: Generate shot gather
    print("Computing shot gather...")
    shot_gather = sim.run_acoustic(pml_width=30, free_surface=True)
    plot_simulation(sim, shot_gather)

    # Option B: Record animation
    print("Recording wavefield...")
    wf_history = sim.record_wavefield(mode='acoustic', freq_interval=6)
    save_wavefield_animation(wf_history, model, filename="wavefield_demo.gif")