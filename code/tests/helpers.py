"""测试共用的合成图像：椭圆（可旋转）、带噪声的背景。"""
import numpy as np

RNG = np.random.default_rng(0)


def ellipse(shape: tuple[int, int], center: tuple[float, float], radii: tuple[float, float],
            angle: float = 0.0) -> np.ndarray:
    """radii = (沿行方向, 沿列方向)，angle 是整个椭圆逆时针转多少度。"""
    yy, xx = np.mgrid[:shape[0], :shape[1]] + 0.5
    dy, dx = yy - center[0], xx - center[1]
    t = np.radians(angle)
    u, v = dy * np.cos(t) + dx * np.sin(t), -dy * np.sin(t) + dx * np.cos(t)
    return (u / radii[0]) ** 2 + (v / radii[1]) ** 2 <= 1


def image(regions: list[tuple[np.ndarray, float]], shape: tuple[int, int], bands: int = 4,
          gradient: float = 0.0) -> np.ndarray:
    """背景 0.03，每块区域一个亮度；gradient 是整幅图（背景和蛋一起）从左到右增加的亮度。"""
    img = 0.03 + RNG.normal(0, 0.005, (*shape, bands))
    for region, level in regions:
        img[region] = level + RNG.normal(0, 0.01, (int(region.sum()), bands))
    return img + gradient * np.linspace(0, 1, shape[1])[None, :, None]
