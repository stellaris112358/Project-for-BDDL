#!/usr/bin/env python3
import os
import sys
# 将项目根目录（diffusionRL）添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import glob
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.unet_cond import UNetCond

# -------------------------
# 数据集：加载RGB图像（[H,W,3] -> [3,H,W]）
# -------------------------
class ImageNPYDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "train_*.npy")))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No train_*.npy files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)  # [H,W,3]（RGB）
        arr = np.transpose(arr, (2, 0, 1))                 # 转为[3,H,W]（通道优先）
        return arr

# -------------------------
# 工具函数
# -------------------------
def make_center_mask(N, K):
    """生成中心KxK为1（观察区），其余为0（待补全区）的掩码"""
    mask = np.zeros((N, N), dtype=np.float32)
    s = (N - K) // 2
    mask[s:s+K, s:s+K] = 1.0
    return mask

def linear_beta_schedule(beta_start, beta_end, T):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

# -------------------------
# 训练函数
# -------------------------
def train(cfg):
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据集和数据加载器
    ds = ImageNPYDataset(cfg["data_dir"])
    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        pin_memory=True
    )

    # 模型：输入3通道图像+1通道掩码=4通道；输出3通道图像
    model = UNetCond(
        ch=cfg["model_channels"],
        in_ch=cfg["in_channels"],  # 4 = 3 RGB + 1 mask
        out_ch=cfg["out_channels"],  # 3 RGB
        T=cfg["T"]
    ).to(device)
    print(f"模型初始化完成：输入通道={cfg['in_channels']}, 输出通道={cfg['out_channels']}")

    # 优化器和调度器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["scheduler_T0"],
        T_mult=cfg["scheduler_Tmult"]
    )

    # 扩散参数
    T = cfg["T"]
    betas = linear_beta_schedule(cfg["beta_start"], cfg["beta_end"], T)
    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)
    alphas = 1.0 - betas_t
    alpha_bar = torch.cumprod(alphas, dim=0)  # 累积乘积

    # 掩码模板（适配3通道图像）
    N = cfg["image_size"]
    K = cfg["condition_size"]
    mask_np = make_center_mask(N, K)  # [H,W]
    # 扩展到3通道掩码（用于图像）和1通道掩码（用于输入拼接）
    mask_channels = torch.tensor(mask_np, dtype=torch.float32, device=device)\
                        .unsqueeze(0).unsqueeze(0).repeat(1, cfg["out_channels"], 1, 1)  # [1,3,H,W]
    mask_single = torch.tensor(mask_np, dtype=torch.float32, device=device)\
                        .unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    best_loss = float("inf")
    model.train()

    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
            # 加载批次数据 [B,3,H,W]
            batch = batch.to(device)
            B = batch.shape[0]

            # 随机采样时间步
            t = torch.randint(0, T, (B,), device=device)  # [B]
            a_bar_t = alpha_bar[t].view(B, 1, 1, 1)  # [B,1,1,1]

            # 前向扩散：x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * noise
            noise = torch.randn_like(batch)  # [B,3,H,W]
            sqrt_a_bar = torch.sqrt(a_bar_t)
            sqrt_1_a_bar = torch.sqrt(1.0 - a_bar_t)
            x_t = sqrt_a_bar * batch + sqrt_1_a_bar * noise  # [B,3,H,W]

            # 构建条件输入：观察区用真实值，非观察区用x_t
            mask_exp_channels = mask_channels.repeat(B, 1, 1, 1)  # [B,3,H,W]
            mask_exp_single = mask_single.repeat(B, 1, 1, 1)      # [B,1,H,W]
            cond = batch * mask_exp_channels  # 中心观察区用真实数据
            in_img = x_t * (1.0 - mask_exp_channels) + cond  # 拼接输入图像
            inp = torch.cat([in_img, mask_exp_single], dim=1)  # [B,4,H,W]（3+1）

            # 模型预测噪声
            pred_noise = model(inp, t)  # [B,3,H,W]

            # 仅计算非观察区的损失（掩码为0的区域）
            mask_unobs = (1.0 - mask_exp_channels)  # [B,3,H,W]
            num_unobs = mask_unobs.sum() + 1e-8  # 避免除零
            mse_map = (pred_noise - noise) **2
            loss = (mse_map * mask_unobs).sum() / num_unobs  # 非观察区平均MSE

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + pbar.n / len(loader))

            epoch_loss += loss.item() * B
            pbar.set_postfix({"batch_loss": f"{loss.item():.6f}"})

        # 计算 epoch 平均损失
        avg_loss = epoch_loss / len(ds)
        print(f"Epoch {epoch+1}/{cfg['epochs']} | 平均损失: {avg_loss:.6f}")

        # 保存最佳模型
        save_dir = cfg.get("save_dir", "models_image")
        os.makedirs(save_dir, exist_ok=True)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "ddpm_cond_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"已保存最佳模型 -> {best_path} (损失={best_loss:.6f})")

        # 定期保存快照
        if (epoch + 1) % 10 == 0:
            snap_path = os.path.join(save_dir, f"ddpm_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), snap_path)
            print(f"已保存快照 -> {snap_path}")

    # 保存最终模型
    final_path = os.path.join(save_dir, "ddpm_cond_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"训练完成。最终模型已保存 -> {final_path}")


if __name__ == "__main__":
    cfg_path = "configs/config.yaml"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path, "r"))
    train(cfg)