import argparse

import torch
import datetime
import time

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_info(*args):
    dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{dtstr}]", *args)

def get_time_ttl_and_eta(time_start, elapsed_iter, total_iter):
    """
    Get estimated total time and ETA time.
    :param time_start:
    :param elapsed_iter:
    :param total_iter:
    :return: string of elapsed time, string of ETA
    """

    def sec_to_str(sec):
        val = int(sec)  # seconds in int type
        s = val % 60
        val = val // 60  # minutes
        m = val % 60
        val = val // 60  # hours
        h = val % 24
        d = val // 24  # days
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"

    elapsed_time = time.time() - time_start  # seconds elapsed
    elp = sec_to_str(elapsed_time)
    if elapsed_iter == 0:
        eta = 'NA'
    else:
        # seconds
        eta = elapsed_time * (total_iter - elapsed_iter) / elapsed_iter
        eta = sec_to_str(eta)
    return elp, eta

def save_list(lst, name, f_path: str, msg=None, fmt="{:.11f}"):
    def num2str(num_arr):
        flt_arr = [float(n) for n in num_arr]
        str_arr = [fmt.format(f) for f in flt_arr]
        return " ".join(str_arr)

    cnt = len(lst) if lst is not None else 0
    with open(f_path, 'w') as f_ptr:
        if type(msg) is list:
            [f_ptr.write(f"# {m}\n") for m in msg]
        elif msg:
            f_ptr.write(f"# {msg}\n")
        if name:
            for i in range(0, cnt, 10):
                r = min(i + 10, cnt)  # right bound
                f_ptr.write(f"{name}[{i:04d}~]: {num2str(lst[i:r])}\n")
        else:
            for i in range(0, cnt):
                f_ptr.write(f"{fmt.format(lst[i])}\n")
    # with

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_geometric_series(start: float, end: float, ratio: float, count: int):
    """
    Create geometric series.
    :param start: start point, float. included in the result.
    :param end:   end point, float. included in the result.
    :param ratio: ratio
    :param count:
    :return:
    """
    res = [start]
    sum = end - start
    item_cnt = count - 1  # item count. Because we include both start and end points.
    if ratio == 1.:
        [res.append(start + sum * i / item_cnt) for i in range(1, item_cnt)]
    else:
        base = sum / (ratio ** item_cnt - 1)
        [res.append(start + base * (ratio**i - 1)) for i in range(1, item_cnt)]

    res.append(end)  # include the end point by default
    return res

def linear_interpolate(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as key points.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp,
     we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size,
            C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand

_onetime_log_map = {}

def onetime_log_append(key, msg):
    if key not in _onetime_log_map:
        _onetime_log_map[key] = []
    arr = _onetime_log_map[key]
    arr.append(msg)

def onetime_log_flush(log_fn=log_info):
    for key in _onetime_log_map:
        log_fn(key)
        [log_fn(msg) for msg in _onetime_log_map[key]]
    # for
    _onetime_log_map.clear()
