import numpy as np
import imageio
import matplotlib.pyplot as plt
import einops
from .arrays import to_np


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def plot2img(fig, remove_margins=True):

    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if remove_margins:
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img_as_string, (width, height) = canvas.print_to_buffer()
    return np.fromstring(img_as_string, dtype="uint8").reshape((height, width, 4))


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def zipsafe(*args):
    length = len(args[0])
    assert all([len(a) == length for a in args])
    return zip(*args)


def zipkw(*args, **kwargs):
    nargs = len(args)
    keys = kwargs.keys()
    vals = [kwargs[k] for k in keys]
    zipped = zipsafe(*args, *vals)
    for items in zipped:
        zipped_args = items[:nargs]
        zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
        yield zipped_args, zipped_kwargs


MAZE_BOUNDS = {
    "maze2d-large-v1": (0, 9, 0, 12),
}


class MazeRenderer:

    def __init__(self, env):
        self._config = env._config
        self._background = self._config != " "
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, title=None):
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        plt.imshow(
            self._background * 0.5,
            extent=self._extent,
            cmap=plt.cm.binary,
            vmin=0,
            vmax=1,
        )

        path_length = len(observations)
        colors = plt.cm.jet(np.linspace(0, 1, path_length))
        plt.plot(observations[:, 1], observations[:, 0], c="black", zorder=10)
        plt.scatter(observations[:, 1], observations[:, 0], c=colors, zorder=20)

        theta = np.linspace(0, 2 * np.pi, 100)
        x = 0.8 / 12.0 * np.cos(theta) + 6.0 / 12.0
        y = 0.8 / 9.0 * np.sin(theta) + 5.2 / 9.0
        plt.plot(x, y, c="red", zorder=10)

        x = (
            0.8 / 12.0 * np.sqrt(np.abs(np.cos(theta))) * np.sign(np.cos(theta))
            + 5.5 / 12.0
        )
        y = (
            0.8 / 9.0 * np.sqrt(np.abs(np.sin(theta))) * np.sign(np.sin(theta))
            + 2.2 / 9.0
        )
        plt.plot(x, y, c="red", zorder=10)

        plt.axis("off")
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    def composite(self, savepath, paths, ncol=5, **kwargs):

        assert (
            len(paths) % ncol == 0
        ), "Number of paths must be divisible by number of columns"

        images = []
        for path, kw in zipkw(paths, **kwargs):
            img = self.renders(*path, **kw)
            images.append(img)
        images = np.stack(images, axis=0)

        nrow = len(images) // ncol
        images = einops.rearrange(
            images, "(nrow ncol) H W C -> (nrow H) (ncol W) C", nrow=nrow, ncol=ncol
        )
        imageio.imsave(savepath, images)
        print(f"Saved {len(paths)} samples to: {savepath}")


class Maze2dRenderer(MazeRenderer):

    def __init__(self, env_name, env, observation_dim=None):
        self.env_name = env_name
        self.env = env
        self.observation_dim = np.prod(self.env.observation_space.shape)
        self.action_dim = np.prod(self.env.action_space.shape)
        self.goal = None
        self._background = self.env.maze_arr == 10
        self._remove_margins = False
        self._extent = (0, 1, 1, 0)

    def renders(self, observations, conditions=None, **kwargs):
        bounds = MAZE_BOUNDS[self.env_name]

        observations = observations + 0.7  # note offset
        if len(bounds) == 2:
            _, scale = bounds
            observations /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[:, 0] /= iscale
            observations[:, 1] /= jscale
        else:
            raise RuntimeError(f"Unrecognized bounds for {self.env_name}: {bounds}")

        if conditions is not None:
            conditions /= scale
        return super().renders(observations, conditions, **kwargs)

    def _normalize_observations(self, observations):
        observations = np.array(observations, copy=True)
        observations[..., 0] += 0.7
        observations[..., 1] += 0.7

        bounds = MAZE_BOUNDS[self.env_name]
        if len(bounds) == 2:
            _, scale = bounds
            observations[..., 0] /= scale
            observations[..., 1] /= scale
        elif len(bounds) == 4:
            _, iscale, _, jscale = bounds
            observations[..., 0] /= iscale
            observations[..., 1] /= jscale
        else:
            raise RuntimeError(f"Unrecognized bounds for {self.env_name}: {bounds}")

        return observations
