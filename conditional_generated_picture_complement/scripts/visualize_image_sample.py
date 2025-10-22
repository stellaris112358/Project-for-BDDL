#!/usr/bin/env python3
import os
import sys
# 将项目根目录（diffusionRL）添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from models.unet_cond import UNetCond

def make_condition_mask(N, K=10):
    """生成中心KxK为1的掩码"""
    mask = np.zeros((N, N), dtype=np.float32)
    s = (N - K) // 2
    mask[s:s+K, s:s+K] = 1.0
    return mask

def make_linear_beta(beta_start, beta_end, T):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

def load_global_max(data_dir):
    path = os.path.join(data_dir, "global_max.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")
    return np.load(path).item()

def visualize_image_sample(sample_idx=None):
    # 加载配置
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model = UNetCond(
        ch=cfg["model_channels"],
        in_ch=4,  # 3 RGB + 1 mask
        out_ch=3,
        T=cfg["T"]
    ).to(device)
    model_path = os.path.join(cfg["save_dir"], "ddpm_cond_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"模型加载成功: {model_path}")

    # 加载数据
    data_files = sorted([f for f in os.listdir(cfg["data_dir"]) if f.startswith("train_") and f.endswith(".npy")])
    if len(data_files) == 0:
        raise FileNotFoundError("数据为空")
    sample_idx = sample_idx or np.random.randint(len(data_files))
    sample_idx = max(0, min(sample_idx, len(data_files)-1))
    sample_path = os.path.join(cfg["data_dir"], data_files[sample_idx])
    gt_norm = np.load(sample_path).astype(np.float32)  # [H,W,3]
    gt_norm = np.transpose(gt_norm, (2,0,1))  # [3,H,W]
    print(f"加载样本 {sample_idx}: {sample_path}, 形状={gt_norm.shape}")

    # 反归一化
    global_max = load_global_max(cfg["data_dir"])
    gt_real = gt_norm * global_max  # 转回0-255范围
    gt_real = np.clip(gt_real, 0, 255).astype(np.uint8)  # 转为整数像素值

    # 条件掩码
    N = cfg["image_size"]
    K = cfg["condition_size"]
    mask = make_condition_mask(N, K)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=device)\
                .unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)  # [1,3,H,W]

    # 初始化反向扩散
    x = torch.randn(1, 3, N, N, device=device)  # 随机噪声
    cond = torch.tensor(gt_norm, dtype=torch.float32, device=device).unsqueeze(0) * mask_t  # 观察区条件

    # 扩散参数
    betas = torch.tensor(make_linear_beta(cfg["beta_start"], cfg["beta_end"], cfg["T"]), device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # 反向扩散采样
    with torch.no_grad():
        for t in reversed(range(cfg["T"])):
            t_tensor = torch.tensor([t], dtype=torch.long, device=device)
            # 输入：3通道带条件的图像 + 1通道掩码
            inp = torch.cat([x * (1-mask_t) + cond, mask_t[:, :1, :, :]], dim=1)  # [1,4,H,W]
            eps_pred = model(inp, t_tensor)  # 预测噪声

            # 计算均值和方差
            a_t = alphas[t]
            a_bar_t = alpha_bar[t]
            beta_t = betas[t]
            coef = beta_t / torch.sqrt(1.0 - a_bar_t)
            mean = (x - coef.view(1,1,1,1) * eps_pred) / torch.sqrt(a_t)

            # 加噪声（最后一步不加）
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    # 替换观察区，反归一化
    gen_norm = x.squeeze(0).cpu().numpy()  # [3,H,W]
    for i in range(3):
        gen_norm[i][mask==1] = gt_norm[i][mask==1]  # 观察区用真实值
    gen_real = gen_norm * global_max
    gen_real = np.clip(gen_real, 0, 255).astype(np.uint8)  # 转为图像格式

    # 调整通道顺序（[3,H,W] -> [H,W,3]）
    gt_real = np.transpose(gt_real, (1,2,0))
    gen_real = np.transpose(gen_real, (1,2,0))
    diff_real = np.abs(gen_real - gt_real).astype(np.uint8)  # 差异图

    # 可视化
    output_dir = cfg.get("output_dir", "outputs_image")
    os.makedirs(output_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(gt_real)
    axs[0].set_title("原始图像 (GT)")
    axs[1].imshow(gen_real)
    axs[1].set_title("生成图像 (Generated)")
    axs[2].imshow(diff_real, cmap="gray")
    axs[2].set_title("像素差异 (绝对值)")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"image_compare_{sample_idx:04d}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 图像保存至 {save_path}")

if __name__ == "__main__":
    visualize_image_sample()