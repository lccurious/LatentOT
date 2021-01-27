import torch
import torch.nn.functional as F
from tqdm import trange
from utils import kmeans, uniform_distribution


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


class LatentOT(object):
    def __init__(self, x: torch.Tensor, y: torch.Tensor,
                 n_anchors_x: int, n_anchors_y: int,
                 eps_x: float, eps_y: float, eps_z: float,
                 m_x=None, m_y=None, m_z=None,
                 device=torch.device('cpu')):
        """
        Latent Optimal Transport

        :param x: NxD source samples
        :param y: NxD target samples
        :param n_anchors_x: number of source anchors
        :param n_anchors_y: number of target anchors
        :param eps_x:
        :param eps_y:
        :param eps_z:
        :param m_x: source mahalanobis distance weight matrix
        :param m_y: target mahalanobis distance weight matrix
        """
        self.device = device
        assert x.shape[1] == y.shape[1], "dimension of inputs not match"
        self.x, self.y = x.to(device), y.to(device)
        self.tol = 1e-5
        self.n_max_iter = 1e5
        self.n_anchors_x = n_anchors_x
        self.n_anchors_y = n_anchors_y
        self.zx = None
        self.zy = None
        self.n_dim = y.shape[1]
        self.eps_x = eps_x
        self.eps_y = eps_y
        self.eps_z = eps_z
        self.n_source = x.shape[0]
        self.n_target = y.shape[0]

        # measures of X and Y
        # TODO: use simplex uniform distribution now, better solution needed
        self.mu = uniform_distribution(self.n_source).to(device)
        self.nu = uniform_distribution(self.n_target).to(device)

        # Mahalanobis weight matrix
        if m_x is None:
            self.m_x = torch.eye(self.n_dim, device=device)
        else:
            self.m_x = m_x.to(device)

        if m_y is None:
            self.m_y = torch.eye(self.n_dim, device=device)
        else:
            self.m_y = m_y.to(device)

        if m_z is None:
            self.m_z = torch.eye(self.n_dim, device=device)
        else:
            self.m_z = m_z.to(device)

        # Intermediate parameters
        self.plan_x = None
        self.plan_y = None
        self.plan_z = None
        self.mu_z = None
        self.nu_z = None

    def initialize(self, x, y):
        """
        Initialize the algorithm 1 details

        :param x: NxD source data samples
        :param y: MxD target data samples
        :return:
        """
        x_labels, x_anchors = kmeans(x, num_clusters=self.n_anchors_x, device=self.device)
        y_labels, y_anchors = kmeans(y, num_clusters=self.n_anchors_y, device=self.device)
        self.plan_x = F.one_hot(x_labels).float().to(self.device)
        self.plan_y = torch.transpose(F.one_hot(y_labels).float(), 0, 1).to(self.device)
        self.plan_z = torch.eye(self.n_anchors_x, self.n_anchors_y, device=self.device)
        self.mu_z = x_labels.float().histc(bins=self.n_anchors_x).to(self.device)
        self.nu_z = y_labels.float().histc(bins=self.n_anchors_y).to(self.device)

    def first_order_condition(self, x, y):
        """
        Calc the first-order stationary condition of OT

        :param x: NxD source samples
        :param y: MxD target samples
        :return vec(zx), vec(zy):
        """
        tl_m = kronecker_product(torch.diag(self.mu_z), self.m_x+self.m_z)
        tr_m = kronecker_product(self.plan_z, self.m_z)
        bl_m = kronecker_product(-torch.transpose(self.plan_z, 0, 1), self.m_z)
        br_m = kronecker_product(torch.diag(self.nu_z), self.m_y+self.m_z)
        a = torch.cat([torch.cat([tl_m, tr_m], dim=1), torch.cat([bl_m, br_m], dim=1)])
        vec_x, vec_y = x.flatten(), y.flatten()

        upper_m = kronecker_product(torch.transpose(self.plan_x, 0, 1), self.m_x).matmul(vec_x)
        lower_m = kronecker_product(self.plan_y, self.m_y).matmul(vec_y)
        ans = torch.matmul(a.inverse(), torch.cat([upper_m, lower_m]))
        return ans.split([self.n_anchors_x*self.n_dim, self.n_anchors_y*self.n_dim])

    def gibbs_kernel(self, x, y, zx, zy):
        """
        Generate the gibbs sampling result

        :param x: NxD source data samples
        :param y: NxD target data samples
        :param zx: K1xD source anchor points
        :param zy: K2xD target anchor points
        :return:
        """
        kernel_x = torch.exp(-self._distance(x, zx) / self.eps_x)
        kernel_y = torch.exp(-self._distance(zy, y) / self.eps_y)
        kernel_z = torch.exp(-self._distance(zx, zy) / self.eps_z)
        return kernel_x, kernel_y, kernel_z

    def fit(self, max_iter=None):
        if max_iter is not None:
            self.n_max_iter = max_iter
        self.initialize(self.x, self.y)

        iterations = 0
        zx, zy = 0, 0
        for i in trange(self.n_max_iter):
            # print("Update loop: {}".format(iterations))
            vec_zx, vec_zy = self.first_order_condition(self.x, self.y)
            _zx = vec_zx.reshape(self.n_anchors_x, -1)
            _zy = vec_zy.reshape(self.n_anchors_y, -1)
            kernel_x, kernel_y, kernel_z = self.gibbs_kernel(self.x, self.y, _zx, _zy)
            self.plan_x, self.plan_y, self.plan_z = self._update(kernel_x, kernel_y, kernel_z)

            iterations += 1

            self.zx, self.zy = zx, zy
            # converge condition
            diff_zx = (zx - _zx).norm()
            diff_zy = (zy - _zy).norm()

            zx, zy = _zx, _zy
            if diff_zx < self.tol and diff_zy < self.tol:
                return self.plan_x, self.plan_y, self.plan_z, zx, zy
        return self.plan_x, self.plan_y, self.plan_z, zx, zy

    def _update(self, kernel_x: torch.Tensor, kernel_y: torch.Tensor, kernel_z: torch.Tensor):
        """
        The update plan

        :param kernel_x: NxK1 kernel matrix
        :param kernel_y: MxK2 kernel matrix
        :param kernel_z: K1xK2 kernel matrix
        :return:
        """
        alpha_x = torch.ones(self.n_source, device=self.device)
        beta_x = torch.ones(self.n_anchors_x, device=self.device)
        alpha_y = torch.ones(self.n_anchors_y, device=self.device)
        beta_y = torch.ones(self.n_target, device=self.device)
        alpha_z = torch.ones(self.n_anchors_x, device=self.device)
        beta_z = torch.ones(self.n_anchors_y, device=self.device)
        iterations = 0
        plan_x, plan_y, plan_z = 0, 0, 0
        while True:
            # print("Update iteration: {}".format(iterations))
            alpha_x = self.mu / kernel_x.matmul(beta_x)
            beta_y = self.nu / torch.transpose(kernel_y, 0, 1).matmul(alpha_y)
            self.mu_z = torch.sqrt(alpha_z * kernel_z.matmul(beta_z) *
                                   beta_x * torch.transpose(kernel_x, 0, 1).matmul(alpha_x))
            beta_x = self.mu_z / torch.transpose(kernel_x, 0, 1).matmul(alpha_x)
            alpha_z = self.mu_z / kernel_z.matmul(beta_z)
            self.nu_z = torch.sqrt((alpha_y * kernel_y.matmul(beta_y)) *
                                   (beta_z * torch.transpose(kernel_z, 0, 1).matmul(alpha_z)))
            beta_z = self.nu_z / torch.transpose(kernel_z, 0, 1).matmul(alpha_z)
            alpha_y = self.nu_z / kernel_y.matmul(beta_y)

            iterations += 1
            # TODO: converging state?
            _plan_x = torch.matmul(torch.diag(alpha_x).matmul(kernel_x), torch.diag(beta_x))
            _plan_y = torch.matmul(torch.diag(alpha_y).matmul(kernel_y), torch.diag(beta_y))
            _plan_z = torch.matmul(torch.diag(alpha_z).matmul(kernel_z), torch.diag(beta_z))
            diff_x = (plan_x - _plan_x).norm()
            diff_y = (plan_y - _plan_y).norm()
            diff_z = (plan_z - _plan_z).norm()
            plan_x, plan_y, plan_z = _plan_x, _plan_y, _plan_z
            # print(f"diff_x: {diff_x}, diff_y: {diff_y}, diff_z:{diff_z}")
            if diff_x < self.tol and diff_y < self.tol and diff_z < self.tol or iterations > self.n_max_iter:
                if iterations > self.n_max_iter:
                    print("Maximum iteration reached, not Converge!")
                return plan_x, plan_y, plan_z

    def _distance(self, x, y=None, M=None, squared=True):
        """
        Calc the pairwise distances between samples

        :param x: NxD dataset matrix
        :param y: NxD dataset matrix
        :param M: DxD mahalanobis matrix default is identity matrix
        :return cost: The distance matrix
        """
        n_dim = x.shape[1]

        if M is None:
            M = torch.eye(n_dim, device=self.device)

        if y is not None:
            assert x.shape[1] == y.shape[1], "Dimension between two x, y not equal!"
        else:
            y = x.unsqueeze(0)
        x = x.unsqueeze(1)

        pairwise_distances_squared = ((x - y).matmul(M) * (x - y)).sum(-1)
        # Deal with numerical inaccuracies. Set small negative to zero
        pairwise_distances_squared = torch.clamp_min(pairwise_distances_squared, 0.0)
        # Get mask where the zero distance are at
        error_mask = pairwise_distances_squared.le(0.0)

        # Ensure diagonal is zero if x=y
        if y is None:
            pairwise_distances_squared = pairwise_distances_squared - torch.diag(pairwise_distances_squared)

        if squared:
            pairwise_distances = pairwise_distances_squared
        else:
            pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

        return pairwise_distances
