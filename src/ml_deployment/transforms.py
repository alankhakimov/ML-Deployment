from torchvision import transforms as T

IMG_SIZE = 224 # Initial size for RadImageNet pretrained models. Later, we will resize.

# RadImageNet's own preprocessing: (x * 255 - 127.5) * 2 / 255, which on a
# [0, 1] tensor is the same as Normalize(mean=0.5, std=0.5), giving range [-1, 1].
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]


def build_transform(size: int = IMG_SIZE):
    return T.Compose([
        T.Grayscale(num_output_channels=3),   # files are mode 'L'; the pretrained first layer expects 3 channels
        T.Resize((size, size)),               # fixed size so images can be stacked into batches
        T.ToTensor(),                         # PIL -> float tensor, 0-255 -> 0-1, layout (C, H, W)
        T.Normalize(MEAN, STD),               # 0-1 -> -1 to 1
    ])