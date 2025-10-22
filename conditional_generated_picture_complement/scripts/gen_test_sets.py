#!/usr/bin/env python3
import os
import numpy as np
import pickle
import torch
import yaml
from models.unet_cond import UNetCond

# --------------------------
# 工具函数
# --------------------------
def load_cifar_test(file_path, num_samples=100):
    """加载CIFAR-10测试集前N个样本"""
    with open(file_path, 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
    data = test_batch['data'][:num_samples]  # [100, 3072]
    labels = test_batch['labels'][:num_samples]  # [100,]
    return data, labels

def make_center_mask(N, K):
    """生成中心KxK为1（观察区）的掩码"""
    mask = np.zeros((N, N), dtype=np.float32)
    s = (N - K) // 2
    mask[s:s+K, s:s+K] = 1.0
    return mask

def linear_beta_schedule(beta_start, beta_end, T):
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

def save_cifar_format(save_path, data, labels):
    batch = {
        'data': data,
        'labels': labels,
        'batch_label': 'test_batch_100',
        'filenames': [f'test_{i}.png' for i in range(len(data))]
    }
    with open(save_path, 'wb') as f:
        pickle.dump(batch, f)
    print(f"✅ 已保存 CIFAR 格式文件（100样本）至: {save_path}")

# --------------------------
# 生成带mask的原测试集
# --------------------------
def generate_masked_testset(cifar_test_path, save_path, mask_size=10, image_size=32):
    print("\n=== 生成带mask的测试集 ===")
    data, labels = load_cifar_test(cifar_test_path, num_samples=100)
    N, K = image_size, mask_size
    mask = make_center_mask(N, K)

    masked_data = []
    for img_flat in data:
        img = img_flat.reshape(3, N, N)
        img_masked = img * mask  # 仅中心区域保留
        masked_data.append(img_masked.flatten())

    masked_data = np.array(masked_data, dtype=np.uint8)
    save_cifar_format(save_path, masked_data, labels)
    return masked_data, labels

# --------------------------
# 生成补全后的测试集
# --------------------------
def generate_completed_testset(cifar_test_path, model, cfg, save_path, mask_size=10, image_size=32):
    print("\n=== 生成补全后的测试集 ===")
    data, labels = load_cifar_test(cifar_test_path, num_samples=100)
    global_max = np.load(os.path.join(cfg["data_dir"], "global_max.npy")).item()  # 255.0
    device = next(model.parameters()).device
    N, K = image_size, mask_size
    mask = make_center_mask(N, K)
    mask_rgb = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1)
    mask_single = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # 扩散参数
    T = cfg["timesteps"]
    betas = torch.tensor(linear_beta_schedule(cfg["beta_start"], cfg["beta_end"], T), device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    completed_data = []
    model.eval()
    with torch.no_grad():
        for i, img_flat in enumerate(data):
            img = img_flat.reshape(3, N, N).astype(np.float32) / global_max
            img_torch = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)

            # 条件
            cond = img_torch * mask_rgb
            mean_channel = cond.mean(dim=1, keepdim=True)

            # 初始化噪声
            x = torch.randn_like(img_torch)

            # 反向扩散生成
            for t in reversed(range(T)):
                t_tensor = torch.tensor([t], dtype=torch.long, device=device)
                inp = torch.cat([x*(1-mask_rgb) + cond, mean_channel, mask_single], dim=1)  # [1,5,H,W]
                eps_pred = model(inp, t_tensor)

                a_t = alphas[t]
                a_bar_t = alpha_bar[t]
                beta_t = betas[t]
                coef = beta_t / torch.sqrt(1.0 - a_bar_t)
                mean = (x - coef.view(1,1,1,1)*eps_pred)/torch.sqrt(a_t)

                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(beta_t) * noise
                else:
                    x = mean

            # 合成最终图像：未知区域是生成，中心区域保留真实值
            gen = x.squeeze(0).cpu().numpy()
            mask_np = mask_rgb.squeeze(0).cpu().numpy()
            gen[:, mask_np[0]==1] = img[:, mask_np[0]==1]
            gen_real = (gen * global_max).clip(0,255).astype(np.uint8)
            completed_data.append(gen_real.flatten())

            if (i+1)%10==0:
                print(f"已处理 {i+1}/100 张图像")

    completed_data = np.array(completed_data, dtype=np.uint8)
    save_cifar_format(save_path, completed_data, labels)
    return completed_data, labels

# --------------------------
# 主函数
# --------------------------
def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    cifar_test_path = "cifar-10-batches-py/test_batch"
    masked_save_path = "data/test_sets_100/masked_test_batch_100"
    completed_save_path = "data/test_sets_100/completed_test_batch_100"
    os.makedirs(os.path.dirname(masked_save_path), exist_ok=True)

    # 1. 带mask的测试集
    generate_masked_testset(
        cifar_test_path=cifar_test_path,
        save_path=masked_save_path,
        mask_size=cfg["condition_size"],
        image_size=cfg["image_size"]
    )

    # 2. 加载 5 通道模型并生成补全
    model = UNetCond(
        ch=cfg["model_channels"],
        in_ch=5,  # RGB + mean + mask
        out_ch=cfg["out_channels"],
        T=cfg["timesteps"]
    ).to(device)
    model_path = os.path.join(cfg["save_dir"], "ddpm_cond_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ 已加载模型: {model_path}")

    generate_completed_testset(
        cifar_test_path=cifar_test_path,
        model=model,
        cfg=cfg,
        save_path=completed_save_path,
        mask_size=cfg["condition_size"],
        image_size=cfg["image_size"]
    )

    print("\n=== 100样本测试集生成完成 ===")
    print(f"带mask的原测试集: {masked_save_path}")
    print(f"补全后的测试集: {completed_save_path}")

if __name__=="__main__":
    main()
