import taichi as ti

ti.init(arch=ti.cpu)
import numpy as np

rng = np.random.default_rng(12345)
noise = rng.integers(0, high=1000, size=(4096, 4096), dtype=np.uint16)
signal = rng.integers(0, high=5000, size=(4096, 4096), dtype=np.uint16)
image = noise | signal


@ti.kernel
def remove_noise_taichi_4(arr: ti.types.ndarray(), noise_level: int):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] < noise_level:
                arr[i, j] = 0


remove_noise_taichi_4(image.copy(), 1000)


@ti.kernel
def remove_noise_taichi_5(arr: ti.types.ndarray(), noise_level: int):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            res = arr[i, j] - noise_level
            mask_to_zero_if_wrapped = -(res <= arr[i, j])
            res = (res & mask_to_zero_if_wrapped) + (
                    noise_level & mask_to_zero_if_wrapped)
            arr[i, j] = res


remove_noise_taichi_5(image.copy(), 1000)


@ti.kernel
def remove_noise_taichi_6(arr: ti.types.ndarray(), noise_level: int):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i, j] = (
                arr[i, j] if arr[i, j] >= noise_level
                else 0
            )


remove_noise_taichi_6(image.copy(), 1000)
