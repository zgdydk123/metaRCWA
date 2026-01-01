import torch


def _to_tensor(value, dtype, device):
    return torch.as_tensor(value, dtype=dtype, device=device)


class Structure:
    """
    Utility for constructing 2D parametric masks with soft edges.
    """

    def __init__(self,
                 Lx: float = 1.,
                 Ly: float = 1.,
                 nx: int = 100,
                 ny: int = 100,
                 edge_sharpness: float = 1000.,
                 *,
                 dtype=torch.float32,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.edge_sharpness = edge_sharpness
        self.dtype = dtype
        self.device = device

        self._x_grid = None
        self._y_grid = None

    # Grid -------------------------------------------------------------------
    def generate_grid(self):
        """Create evenly spaced sampling grids on x-y domain."""
        x = (self.Lx / self.nx) * (torch.arange(self.nx, dtype=self.dtype, device=self.device) + 0.5)
        y = (self.Ly / self.ny) * (torch.arange(self.ny, dtype=self.dtype, device=self.device) + 0.5)
        self._x_grid, self._y_grid = torch.meshgrid(x, y, indexing='ij')
        return self._x_grid, self._y_grid

    # Shapes -----------------------------------------------------------------
    def circle_mask(self, radius, cx, cy):
        """Soft-edged circle."""
        xg, yg = self._ensure_grid()
        level = 1. - torch.sqrt(((xg - cx) / radius) ** 2 + ((yg - cy) / radius) ** 2)
        return torch.sigmoid(self.edge_sharpness * level)

    def ellipse_mask(self, rx, ry, cx, cy, theta=0.):
        """Soft-edged ellipse with rotation."""
        theta = _to_tensor(theta, self.dtype, self.device)
        xg, yg = self._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - torch.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)
        return torch.sigmoid(self.edge_sharpness * level)

    def square_mask(self, width, cx, cy, theta=0.):
        """Soft-edged square rotated about center."""
        return self.rectangle_mask(width, width, cx, cy, theta)

    def rectangle_mask(self, wx, wy, cx, cy, theta=0.):
        """Soft-edged rectangle rotated about center."""
        theta = _to_tensor(theta, self.dtype, self.device)
        xg, yg = self._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - torch.maximum(torch.abs(x_rot / (wx / 2.)), torch.abs(y_rot / (wy / 2.)))
        return torch.sigmoid(self.edge_sharpness * level)

    def rhombus_mask(self, wx, wy, cx, cy, theta=0.):
        """Soft-edged rhombus rotated about center."""
        theta = _to_tensor(theta, self.dtype, self.device)
        xg, yg = self._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - (torch.abs(x_rot / (wx / 2.)) + torch.abs(y_rot / (wy / 2.)))
        return torch.sigmoid(self.edge_sharpness * level)

    def super_ellipse_mask(self, wx, wy, cx, cy, theta=0., power=2.):
        """Soft-edged super-ellipse parameterized by `power`."""
        theta = _to_tensor(theta, self.dtype, self.device)
        xg, yg = self._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - (torch.abs(x_rot / (wx / 2.)) ** power + torch.abs(y_rot / (wy / 2.)) ** power) ** (1 / power)
        return torch.sigmoid(self.edge_sharpness * level)

    # Boolean ops ------------------------------------------------------------
    @staticmethod
    def union_mask(mask_a, mask_b):
        return torch.maximum(mask_a, mask_b)

    @staticmethod
    def intersection_mask(mask_a, mask_b):
        return torch.minimum(mask_a, mask_b)

    @staticmethod
    def difference_mask(mask_a, mask_b):
        return torch.minimum(mask_a, 1. - mask_b)

    # Utilities --------------------------------------------------------------
    def _ensure_grid(self):
        if self._x_grid is None or self._y_grid is None:
            self.generate_grid()
        return self._x_grid, self._y_grid

class StructureLibrary:
    """
    Class-level variant of Structure for quick, shared usage.
    """

    edge_sharpness = 100.
    Lx = 1.
    Ly = 1.
    nx = 100
    ny = 100
    dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _x_grid = None
    _y_grid = None

    def __init__(self):
        pass

    @classmethod
    def generate_grid(cls):
        x = (cls.Lx / cls.nx) * (torch.arange(cls.nx, dtype=cls.dtype, device=cls.device) + 0.5)
        y = (cls.Ly / cls.ny) * (torch.arange(cls.ny, dtype=cls.dtype, device=cls.device) + 0.5)
        cls._x_grid, cls._y_grid = torch.meshgrid(x, y, indexing='ij')
        return cls._x_grid, cls._y_grid

    @classmethod
    def circle_mask(cls, radius, cx, cy):
        xg, yg = cls._ensure_grid()
        level = 1. - torch.sqrt(((xg - cx) / radius) ** 2 + ((yg - cy) / radius) ** 2)
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def ellipse_mask(cls, rx, ry, cx, cy, theta=0.):
        theta = _to_tensor(theta, cls.dtype, cls.device)
        xg, yg = cls._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - torch.sqrt((x_rot / rx) ** 2 + (y_rot / ry) ** 2)
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def square_mask(cls, width, cx, cy, theta=0.):
        return cls.rectangle_mask(width, width, cx, cy, theta)

    @classmethod
    def rectangle_mask(cls, wx, wy, cx, cy, theta=0.):
        theta = _to_tensor(theta, cls.dtype, cls.device)
        xg, yg = cls._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - torch.maximum(torch.abs(x_rot / (wx / 2.)), torch.abs(y_rot / (wy / 2.)))
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def rhombus_mask(cls, wx, wy, cx, cy, theta=0.):
        theta = _to_tensor(theta, cls.dtype, cls.device)
        xg, yg = cls._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - (torch.abs(x_rot / (wx / 2.)) + torch.abs(y_rot / (wy / 2.)))
        return torch.sigmoid(cls.edge_sharpness * level)

    @classmethod
    def super_ellipse_mask(cls, wx, wy, cx, cy, theta=0., power=2.):
        theta = _to_tensor(theta, cls.dtype, cls.device)
        xg, yg = cls._ensure_grid()
        x_rot = (xg - cx) * torch.cos(theta) + (yg - cy) * torch.sin(theta)
        y_rot = -(xg - cx) * torch.sin(theta) + (yg - cy) * torch.cos(theta)
        level = 1. - (torch.abs(x_rot / (wx / 2.)) ** power + torch.abs(y_rot / (wy / 2.)) ** power) ** (1 / power)
        return torch.sigmoid(cls.edge_sharpness * level)

    @staticmethod
    def union_mask(mask_a, mask_b):
        return torch.maximum(mask_a, mask_b)

    @staticmethod
    def intersection_mask(mask_a, mask_b):
        return torch.minimum(mask_a, mask_b)

    @staticmethod
    def difference_mask(mask_a, mask_b):
        return torch.minimum(mask_a, 1. - mask_b)

    @classmethod
    def _ensure_grid(cls):
        if cls._x_grid is None or cls._y_grid is None:
            cls.generate_grid()
        return cls._x_grid, cls._y_grid
