---
title: "Fomo.jl Benchmark"
author: zzzzswh
pubDatetime: 2026-03-25
description: "Benchmark of Fomo.jl"
categories: ["科研碎片Snapshot"]
tags:
  - Fomo.jl
  - Deepwave
  - Forward Modeling
  - 中文
  - English
featured: true
date: 2026-03-25
comments: true
---

## English

**Deepwave's** forward modeling module is highly efficient and offers excellent accuracy. It supports scalar acoustic, acoustic, and elastic wave equations (using a staggered grid + PML).  
I conducted a benchmark comparing my Julia package, **Fomo.jl**, against both Deepwave and Specfem2D. The results are shown below:  
[Fomo.jl](https://www.github.com/zzzzswh/Fomo.jl)  

**Elastic Wave Module Comparison** Shot Gather Comparison
![Benchmark Results Shots](../../../assets/images/Benchmark_Fomo01.jpg)

Waveform/Trace Comparison
![Benchmark Results Traces](../../../assets/images/Benchmark_Fomo02.jpg)

Computational Time Comparison
|                     | Deepwave | Fomo.jl | Specfem2D |
|---------------------|----------|---------|-----------|
| Runtime per shot    | 0.205s   | 0.381s  | 10.3s     |


## 中文

**Deepwave**的正演模块效率很高，准确度也不错，支持标量声波方程、声波方程还有弹性波方程（staggered grid + PML）。  
我把我的julia包**Fomo.jl**和他还有specfem2D做过一个**Benchmark**，结果如下：  
[Fomo.jl](https://www.github.com/zzzzswh/Fomo.jl)  

**弹性波模块对比** 炮集对比
![Benchmark Results Shots](../../../assets/images/Benchmark_Fomo01.jpg)

波形对比
![Benchmark Results Traces](../../../assets/images/Benchmark_Fomo02.jpg)

时间对比
|           |Deepwave | Fomo.jl | Specfem2D |
|-----------|---------|---------|-----------|
|单炮运行时间| 0.205s  | 0.381s  | 10.3s     |