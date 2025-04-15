import torch
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def show_images(images, title="", titles=[]):
    plt.figure(figsize=(8, 8))
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i+1)
        img = images[i].permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def rescale(value, in_range, out_range, clamp=False):
    in_min, in_max = in_range
    out_min, out_max = out_range

    in_span = in_max - in_min
    out_span = out_max - out_min

    scaled_value = (value - in_min) / (in_span + 1e-8)
    rescaled_value = out_min + (scaled_value * out_span)

    if clamp:
        rescaled_value = torch.clamp(
            rescaled_value,
            out_min, out_max
        )

    return rescaled_value


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")


def plot_loss(losses, losses_save_path):
    plt.figure(figsize=(16, 8))

    plt.plot(losses, label='Loss', color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    min_loss = min(losses)
    min_loss_epoch = losses.index(min_loss)
    plt.scatter(min_loss_epoch, min_loss, color='green', zorder=5)
    plt.text(min_loss_epoch,
             min_loss+0.025,
             f'{min_loss:.4f}',
             fontsize=12,
             verticalalignment='center',
             horizontalalignment='center',
             color='green')

    plt.legend()
    plt.grid()
    plt.savefig(losses_save_path)
