"""
Optical flow by per-window optimization instead of a trained network.

These "models" have no weights. They plug into eval_flow.py like the networks, so that they see the
same sequences and are scored by the same metrics, but they estimate the flow of every window by
optimizing a contrast objective on the most recent events (contrast maximization, Gallego et al.,
CVPR 2018).

Their memory is a buffer with the last `num_events` events, which plays the role that the recurrent
state plays in the networks: the past is kept outside of the input representation. A velocity field
(px/s) is estimated from the buffer and converted to the displacement of the current window, which
is the quantity the metrics of this repository expect.

CMaxOptimizer   own, simple implementation: multi-scale grid of flow vectors optimized with Adam.
ExternalCMax    wrapper around the public code of Shiba et al., "Secrets of Event-Based Optical
                Flow" (ECCV 2022), https://github.com/tub-rip/event_based_optical_flow. That code is
                GPL-3.0 and is NOT part of this repository: it is imported at run time from the
                folder given by the environment variable EVFLOW_SHIBA_REPO.
"""

import contextlib
import copy
import io
import os
import sys

import torch
import torch.nn.functional as F

from .base import BaseModel
from loss.flow import EventWarping
from utils.iwe import get_interpolation, interpolate


class EventOptimizer(BaseModel):
    """
    Common part: event buffer, bookkeeping of time, conversion of the estimated velocity to the units
    of the networks (displacement of the current window divided by flow_scaling).
    """

    needs_events = True

    def __init__(self, kwargs):
        super().__init__()
        self.mask = kwargs.get("mask_output", True)
        self.num_events = int(kwargs.get("num_events", 30000))
        self.flow_scaling = 128
        self.res = None
        self.windows = 0
        self.reset_states()

    def configure(self, config):
        """
        Called once by eval_flow.py: the flow has to be expressed with the scaling of the metrics.
        """

        assert config["loader"]["batch_size"] == 1, "Optimization-based estimators need batch_size = 1"
        self.flow_scaling = config["metrics"]["flow_scaling"]
        self.res = list(config["loader"]["resolution"])

    def reset_states(self):
        self.buffer = None  # [N x 5] (t [s], y, x, pos, neg)
        self.clock = 0.0
        self.velocity = None  # [1 x 2 x H x W] (x, y) in px/s, last estimate
        self.buffer_full = False  # the buffer holds num_events events (not the case at sequence start)
        self.reset_solver()

    def reset_solver(self):
        pass

    def forward(self, event_voxel, event_cnt, log=False, events=None, optimize=True):
        """
        :param events: dataloader dictionary of the current window (event_list, event_list_pol_mask,
        dt_input)
        :param optimize: if False, the events are only added to the buffer and the previous estimate
        is returned (used for windows that are not scored)
        :return dictionary with the flow of the current window, [[1 x 2 x H x W]] (x, y)
        """

        assert events is not None, "eval_flow.py has to pass the events to optimization-based estimators"
        device = event_cnt.device
        res = self.res if self.res is not None else list(event_cnt.shape[-2:])
        event_list = events["event_list"].to(device)  # [1 x N x 4] (ts in [0, 1], y, x, p)
        pol_mask = events["event_list_pol_mask"].to(device)  # [1 x N x 2] (pos, neg)
        dt_input = float(events["dt_input"])

        # add the window to the buffer, in absolute time
        if event_list.shape[1] > 0:
            new = torch.cat([event_list[0, :, 0:3], pol_mask[0]], dim=1).clone()
            new[:, 0] = self.clock + new[:, 0] * dt_input
            self.buffer = new if self.buffer is None else torch.cat([self.buffer, new], dim=0)
            self.buffer = self.buffer[-self.num_events :]
        self.clock += dt_input

        duration = 0.0 if self.buffer is None else float(self.buffer[-1, 0] - self.buffer[0, 0])
        if optimize and duration > 0.0 and self.buffer.shape[0] >= 16:
            self.buffer_full = self.buffer.shape[0] >= self.num_events
            with torch.enable_grad():
                self.velocity = self.solve(self.buffer, duration, res, device).detach()
            self.windows += 1

        if self.velocity is None:
            flow = torch.zeros((1, 2, res[0], res[1]), device=device)
        else:
            flow = self.velocity * dt_input / self.flow_scaling  # displacement of the current window
        return {"flow": [flow], "activity": None}

    def solve(self, buffer, duration, res, device):
        """
        :param buffer: [N x 5] (t [s], y, x, pos, neg) most recent events, sorted in time
        :param duration: time [s] spanned by the buffer
        :return [1 x 2 x H x W] (x, y) velocity in px/s
        """

        raise NotImplementedError


class CMaxOptimizer(EventOptimizer):
    """
    Multi-scale grid of flow vectors (1x1 up to the finest scale), summed after bilinear upsampling
    and optimized with Adam from the previous estimate.

    At the start of a sequence the buffer holds few events and the estimates are poor; warm-starting
    from them with a few steps recovers slowly. Until the buffer has been full once, every
    optimization therefore starts from zero with `iters_first` steps.

    objective = "timestamp": the loss the networks of this repository are trained with (image of
        averaged timestamps, forward and backward, plus smoothness), optimized directly.
    objective = "variance": variance of the image of warped events as in Gallego et al. 2018,
        relative to the variance without warping, plus smoothness.
    """

    def __init__(self, kwargs):
        self.scales = [int(s) for s in kwargs.get("scales", [1, 2, 4, 8, 16])]
        self.objective = kwargs.get("objective", "timestamp")
        assert self.objective in ["timestamp", "variance"]
        self.iters = int(kwargs.get("iters", 40))
        self.iters_first = int(kwargs.get("iters_first", 200))
        self.lr = float(kwargs.get("lr", 0.2))  # px, in units of displacement over the buffer
        self.regul_weight = float(kwargs.get("regul_weight", 0.001))
        super().__init__(kwargs)

    def reset_solver(self):
        self.params = None  # displacement [px] over the buffer, one tensor per scale
        self.prev_duration = None
        self.warm = False  # a previous estimate from a full buffer exists

    def dense(self, params, res):
        flow = 0
        for p in params:
            flow = flow + F.interpolate(p, size=(res[0], res[1]), mode="bilinear", align_corners=False)
        return flow

    def solve(self, buffer, duration, res, device):
        ts = (buffer[:, 0:1] - buffer[0, 0]) / duration
        event_list = torch.cat([ts, buffer[:, 1:3], torch.zeros_like(ts)], dim=1).unsqueeze(0)
        pol_mask = buffer[:, 3:5].unsqueeze(0)
        idx = (event_list[0, :, 1] * res[1] + event_list[0, :, 2]).long()
        event_mask = torch.zeros(res[0] * res[1], device=device)
        event_mask[idx] = 1.0
        event_mask = event_mask.view(1, 1, res[0], res[1])

        # warm start: the velocity is kept, so the displacement scales with the duration of the buffer
        first = self.params is None or not self.warm
        self.warm = self.buffer_full
        if first:
            self.params = [torch.zeros((1, 2, s, s), device=device) for s in self.scales]
        else:
            self.params = [p * (duration / self.prev_duration) for p in self.params]
        self.params = [p.detach().requires_grad_(True) for p in self.params]
        self.prev_duration = duration

        iters = self.iters_first if first else self.iters
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=self.lr * 0.05)
        if self.objective == "timestamp":
            config = {
                "loader": {"resolution": res},
                "loss": {"flow_regul_weight": self.regul_weight, "overwrite_intermediate": False},
                "model": {"mask_output": True},
            }
            criterion = EventWarping(config, device, flow_scaling=1.0)

        for _ in range(iters):
            optimizer.zero_grad()
            flow = self.dense(self.params, res)  # displacement [px] over the buffer
            if self.objective == "timestamp":
                criterion.event_flow_association([flow], event_list.clone(), pol_mask, event_mask)
                loss = criterion()
                criterion.reset()
            else:
                loss = self.variance_loss(flow, event_list, event_mask, res)
            loss.backward()
            optimizer.step()
            scheduler.step()

        self.params = [p.detach() for p in self.params]
        return self.dense(self.params, res) / duration

    def variance_loss(self, flow, event_list, event_mask, res):
        flat = flow.view(1, 2, -1)
        idx = (event_list[:, :, 1] * res[1] + event_list[:, :, 2]).long()
        event_flow = torch.stack([torch.gather(flat[:, 1], 1, idx), torch.gather(flat[:, 0], 1, idx)], dim=2)  # (y, x)

        contrast = 0
        for tref in [0.0, 1.0]:
            warped_idx, weights = get_interpolation(event_list, event_flow, tref, res, 1.0)
            iwe = interpolate(warped_idx.long(), weights, res)
            contrast = contrast + torch.var(iwe.view(1, -1), dim=1).sum()
        zero_idx, zero_weights = get_interpolation(event_list, torch.zeros_like(event_flow), 0.0, res, 1.0)
        reference = torch.var(interpolate(zero_idx.long(), zero_weights, res).view(1, -1), dim=1).sum()

        dx = (flow[:, :, :, :-1] - flow[:, :, :, 1:]) * (event_mask[:, :, :, :-1] * event_mask[:, :, :, 1:])
        dy = (flow[:, :, :-1, :] - flow[:, :, 1:, :]) * (event_mask[:, :, :-1, :] * event_mask[:, :, 1:, :])
        smoothness = torch.sqrt(dx ** 2 + 1e-6).sum() + torch.sqrt(dy ** 2 + 1e-6).sum()
        return -contrast / (2.0 * reference + 1e-9) + self.regul_weight * smoothness / 2.0


class ExternalCMax(EventOptimizer):
    """
    The solver of Shiba et al. (ECCV 2022), run on the event buffer of this repository.

    kwargs:
        config: yaml file of the external repository, relative to it
            (default: configs/mvsec_indoor_no_timeaware.yaml)
        num_events: events per optimization (30000 in the released config, which is for
            indoor_flying1; the released code contains no config for the outdoor sequence)
    The image size and the tile crop of the external config are replaced by the resolution used here.
    As in the external evaluation script, every optimization starts from the previous estimate, except
    while the buffer is still filling at the start of a sequence (the external script always optimizes
    on a full set of events and has no such phase).
    """

    def __init__(self, kwargs):
        self.external_config = kwargs.get("config", "configs/mvsec_indoor_no_timeaware.yaml")
        self.solver = None
        super().__init__(kwargs)

    def reset_solver(self):
        self.solver = None  # rebuilt for every sequence: no estimate is carried over

    def build(self, res):
        import yaml

        repo = os.environ.get("EVFLOW_SHIBA_REPO", "")
        if not os.path.isdir(os.path.join(repo, "src", "solver")):
            raise FileNotFoundError(
                "EVFLOW_SHIBA_REPO must point to a clone of https://github.com/tub-rip/event_based_optical_flow "
                "(got '{}'). See cluster/setup_shiba.sh.".format(repo)
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src import solver as external_solver  # noqa: E402

        with open(os.path.join(repo, self.external_config)) as fid:
            config = yaml.safe_load(fid)
        config["solver"]["patch"]["crop_height"] = res[0]
        config["solver"]["patch"]["crop_width"] = res[1]
        self.time_aware = bool(config["solver"].get("time_aware", False))
        assert not self.time_aware, "Only the configuration without time-aware flow is supported"
        return external_solver.collections[config["solver"]["method"]](
            (res[0], res[1]),
            calibration_parameter={},
            solver_config=copy.deepcopy(config["solver"]),
            optimizer_config=copy.deepcopy(config["optimizer"]),
            output_config={"output_dir": "."},
            visualize_module=None,
        )

    def solve(self, buffer, duration, res, device):
        if self.solver is None:
            self.solver = self.build(res)

        # external convention: [N x 4] (row, column, t [s] starting at 0, polarity in {-1, 1})
        events = torch.stack(
            [buffer[:, 1], buffer[:, 2], buffer[:, 0] - buffer[0, 0], buffer[:, 3] - buffer[:, 4]], dim=1
        )
        events = events.double().cpu().numpy()

        cwd = os.getcwd()
        scratch = os.environ.get("EVFLOW_SHIBA_SCRATCH", os.environ.get("TMPDIR", "/tmp"))
        os.chdir(scratch)  # the external solver writes a profile file into the working directory
        try:
            # the external solver prints a convergence report for every scale of every window
            with contextlib.redirect_stdout(io.StringIO()):
                motion = self.solver.optimize(events)
            if self.buffer_full:
                self.solver.set_previous_frame_best_estimation(motion)
            else:
                self.solver.previous_frame_best_estimation = None
            dense = self.solver.motion_to_dense_flow(motion)  # [2 x H x W] (y, x) in px/s
        finally:
            os.chdir(cwd)

        dense = torch.as_tensor(dense, dtype=torch.float32, device=device)
        return torch.stack([dense[1], dense[0]], dim=0).unsqueeze(0)  # (x, y)
