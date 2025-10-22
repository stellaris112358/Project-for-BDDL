# models/unet_cond.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBlock(nn.Module):
    """基础卷积块（含残差连接，采用 GroupNorm，增强小 batch 下稳定性）"""
    def __init__(self, in_ch, out_ch, groups=8, use_residual=True):
        super(SimpleBlock, self).__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        if self.use_residual:
            x = x + residual
        return F.relu(x)


class TimeEmbedding(nn.Module):
    """时间步嵌入（sin/cos + MLP），输入 t: [B] 或 [B,]"""
    def __init__(self, T, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, t):
        # t: tensor shape [B] (long or float)
        # produce [B, dim]
        half_dim = self.dim // 2
        device = t.device
        emb_scale = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) *
                              -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        t_float = t.float().unsqueeze(1)  # [B,1]
        emb = t_float * emb_scale.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)  # [B, dim]
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb  # [B, dim]


class UNetCond(nn.Module):
    """
    条件 U-Net（输入：图像 + 掩码 concat，输出：每通道噪声预测）
    - in_ch: 输入通道数（例如 6 = 5 traffic + 1 mask）
    - out_ch: 输出通道数（例如 5）
    - ch: 基础通道宽度
    - T: 最大时间步（用于 time embedding）
    """
    def __init__(self, ch=128, in_ch=6, out_ch=5, T=1000, groups_gn=8):
        super(UNetCond, self).__init__()
        self.T = T
        self.ch = ch
        self.out_ch = out_ch

        # 时间嵌入
        self.time_emb = TimeEmbedding(T, ch)
        self.time_to_enc1 = nn.Linear(ch, ch)
        self.time_to_enc2 = nn.Linear(ch, ch * 2)
        self.time_to_enc3 = nn.Linear(ch, ch * 4)
        self.time_to_enc4 = nn.Linear(ch, ch * 8)
        self.time_to_bottleneck = nn.Linear(ch, ch * 8)

        # 编码器
        self.enc1 = SimpleBlock(in_ch, ch, groups=groups_gn)
        self.enc2 = SimpleBlock(ch, ch * 2, groups=groups_gn)
        self.enc3 = SimpleBlock(ch * 2, ch * 4, groups=groups_gn)
        self.enc4 = SimpleBlock(ch * 4, ch * 8, groups=groups_gn)

        # 瓶颈
        self.bottleneck = SimpleBlock(ch * 8, ch * 8, groups=groups_gn)

        # 解码器
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = SimpleBlock(ch * 8, ch * 4, groups=groups_gn)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = SimpleBlock(ch * 4, ch * 2, groups=groups_gn)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = SimpleBlock(ch * 2, ch, groups=groups_gn)

        # 输出层：预测 out_ch 个通道的噪声
        self.outc = nn.Conv2d(ch, out_ch, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, t):
        """
        x: [B, in_ch, H, W]
        t: [B] (long) or [B,] scalar times
        return: [B, out_ch, H, W]
        """
        B, _, H, W = x.shape

        # 时间嵌入
        t_feat = self.time_emb(t)  # [B, ch]
        enc1_t = self.time_to_enc1(t_feat)[:, :, None, None]  # [B, ch,1,1]
        enc2_t = self.time_to_enc2(t_feat)[:, :, None, None]  # [B, ch*2,1,1]
        enc3_t = self.time_to_enc3(t_feat)[:, :, None, None]  # [B, ch*4,1,1]
        enc4_t = self.time_to_enc4(t_feat)[:, :, None, None]  # [B, ch*8,1,1]
        bottleneck_t = self.time_to_bottleneck(t_feat)[:, :, None, None]  # [B, ch*8,1,1]

        # 编码器（融合时间特征），按当前 H W 自动 repeat
        e1 = self.enc1(x) + enc1_t.repeat(1, 1, H, W)
        e2 = self.enc2(self.pool(e1)) + enc2_t.repeat(1, 1, H // 2, W // 2)
        e3 = self.enc3(self.pool(e2)) + enc3_t.repeat(1, 1, H // 4, W // 4)
        e4 = self.enc4(self.pool(e3)) + enc4_t.repeat(1, 1, H // 8, W // 8)

        # 瓶颈
        b = self.bottleneck(e4 + bottleneck_t.repeat(1, 1, H // 8, W // 8))

        # 解码器
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.outc(d1)
        return out
