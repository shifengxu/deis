import torch
import os
import time
import datetime
import torchvision.utils as tvu
import numpy as np
import torch_fidelity
import th_deis as deis
from models.diffusion import Model
from models.ema import EMAHelper
# import jax_deis as deis
import utils

log_fn = utils.log_info

def model_load_from_local(model, args):
    def apply_ema():
        log_fn(f"  ema_helper: EMAHelper()")
        ema_helper = EMAHelper()
        ema_helper.register(model)
        k = "ema_helper" if isinstance(states, dict) else -1
        log_fn(f"  ema_helper: load from states[{k}]")
        ema_helper.load_state_dict(states[k])
        log_fn(f"  ema_helper: apply to model {type(model).__name__}")
        ema_helper.ema(model)

    ckpt_path = args.sample_ckpt_path
    log_fn(f"load ckpt: {ckpt_path}")
    states = torch.load(ckpt_path, map_location=args.device)
    if 'model' not in states:
        log_fn(f"  !!! Not found 'model' in states. Will take it as pure model")
        model.load_state_dict(states)
    else:
        key = 'model' if isinstance(states, dict) else 0
        model.load_state_dict(states[key], strict=True)
        ckpt_tt = states.get('ts_type', 'discrete')
        model_tt = model.ts_type
        if ckpt_tt != model_tt:
            raise ValueError(f"ts_type not match. ckpt_tt={ckpt_tt}, model_tt={model_tt}")
        if not hasattr(args, 'ema_flag'):
            log_fn(f"  !!! Not found ema_flag in args. Assume it is true.")
            apply_ema()
        elif args.ema_flag:
            log_fn(f"  Found args.ema_flag: {args.ema_flag}.")
            apply_ema()
    # endif
    model = model.to(args.device)
    if len(args.gpu_ids) > 1:
        log_fn(f"  torch.nn.DataParallel(model, device_ids={args.gpu_ids})")
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)

    return model

_ts_list = None
def get_eps_fn(args, config):
    eps_model = Model(config)
    eps_model = model_load_from_local(eps_model, args)
    eps_model = eps_model.to(args.device)
    eps_model.eval()

    def eps_fn(x_t, scalar_t):
        global _ts_list
        if _ts_list is not None:
            _ts_list.append(scalar_t)
        vec_t = (torch.ones(x_t.shape[0])).float().to(x_t) * scalar_t
        vec_t = vec_t.to(args.device)
        # vec_t *= 1000.  # from [0,1] to [0, 1000]
        # vec_t -= 1      # change upper limitation from 1000 to 999
        with torch.no_grad():
            return eps_model(x_t, vec_t)
    return eps_fn

def get_sde(is_discrete=True):
    # mappings between t and alpha in VPSDE
    # we provide popular linear and cos mappings

    if is_discrete:
        # linear scheduled alpha_bar
        li_ab = [0.999900, 0.999780, 0.999640, 0.999481, 0.999301, 0.999102, 0.998882, 0.998643, 0.998384, 0.998105,
                 0.997807, 0.997488, 0.997150, 0.996792, 0.996414, 0.996017, 0.995600, 0.995163, 0.994707, 0.994231,
                 0.993735, 0.993220, 0.992686, 0.992131, 0.991558, 0.990965, 0.990353, 0.989721, 0.989070, 0.988400,
                 0.987710, 0.987002, 0.986274, 0.985527, 0.984761, 0.983976, 0.983172, 0.982349, 0.981507, 0.980646,
                 0.979767, 0.978869, 0.977952, 0.977016, 0.976062, 0.975090, 0.974099, 0.973089, 0.972062, 0.971016,
                 0.969951, 0.968869, 0.967768, 0.966650, 0.965514, 0.964359, 0.963187, 0.961997, 0.960789, 0.959564,
                 0.958321, 0.957061, 0.955783, 0.954488, 0.953176, 0.951846, 0.950500, 0.949136, 0.947756, 0.946358,
                 0.944944, 0.943513, 0.942065, 0.940601, 0.939121, 0.937624, 0.936110, 0.934581, 0.933035, 0.931474,
                 0.929896, 0.928303, 0.926694, 0.925069, 0.923428, 0.921773, 0.920101, 0.918415, 0.916713, 0.914996,
                 0.913264, 0.911517, 0.909756, 0.907979, 0.906188, 0.904383, 0.902563, 0.900729, 0.898880, 0.897018,
                 0.895141, 0.893251, 0.891347, 0.889429, 0.887497, 0.885552, 0.883594, 0.881622, 0.879637, 0.877639,
                 0.875629, 0.873605, 0.871569, 0.869520, 0.867458, 0.865384, 0.863298, 0.861200, 0.859089, 0.856967,
                 0.854832, 0.852687, 0.850529, 0.848360, 0.846180, 0.843988, 0.841785, 0.839572, 0.837347, 0.835112,
                 0.832865, 0.830609, 0.828342, 0.826064, 0.823777, 0.821479, 0.819171, 0.816854, 0.814527, 0.812190,
                 0.809844, 0.807488, 0.805123, 0.802750, 0.800367, 0.797975, 0.795574, 0.793165, 0.790747, 0.788321,
                 0.785887, 0.783444, 0.780994, 0.778536, 0.776069, 0.773596, 0.771114, 0.768626, 0.766130, 0.763626,
                 0.761116, 0.758599, 0.756075, 0.753545, 0.751008, 0.748464, 0.745914, 0.743358, 0.740797, 0.738229,
                 0.735655, 0.733075, 0.730490, 0.727900, 0.725304, 0.722703, 0.720097, 0.717486, 0.714871, 0.712250,
                 0.709625, 0.706995, 0.704362, 0.701724, 0.699081, 0.696435, 0.693785, 0.691131, 0.688474, 0.685813,
                 0.683149, 0.680481, 0.677811, 0.675137, 0.672461, 0.669782, 0.667099, 0.664415, 0.661728, 0.659039,
                 0.656347, 0.653654, 0.650958, 0.648261, 0.645561, 0.642861, 0.640158, 0.637455, 0.634750, 0.632044,
                 0.629336, 0.626628, 0.623919, 0.621210, 0.618500, 0.615789, 0.613078, 0.610366, 0.607655, 0.604943,
                 0.602232, 0.599520, 0.596809, 0.594098, 0.591388, 0.588678, 0.585969, 0.583261, 0.580554, 0.577847,
                 0.575142, 0.572438, 0.569735, 0.567034, 0.564334, 0.561636, 0.558940, 0.556245, 0.553552, 0.550861,
                 0.548173, 0.545486, 0.542802, 0.540120, 0.537441, 0.534764, 0.532090, 0.529419, 0.526751, 0.524085,
                 0.521423, 0.518764, 0.516108, 0.513455, 0.510806, 0.508160, 0.505518, 0.502880, 0.500245, 0.497614,
                 0.494987, 0.492364, 0.489745, 0.487130, 0.484520, 0.481914, 0.479312, 0.476715, 0.474122, 0.471534,
                 0.468951, 0.466372, 0.463799, 0.461230, 0.458667, 0.456108, 0.453555, 0.451007, 0.448464, 0.445927,
                 0.443395, 0.440869, 0.438348, 0.435833, 0.433324, 0.430821, 0.428324, 0.425832, 0.423346, 0.420867,
                 0.418394, 0.415926, 0.413466, 0.411011, 0.408563, 0.406121, 0.403686, 0.401257, 0.398835, 0.396420,
                 0.394011, 0.391609, 0.389214, 0.386826, 0.384445, 0.382071, 0.379704, 0.377344, 0.374991, 0.372645,
                 0.370307, 0.367976, 0.365652, 0.363335, 0.361027, 0.358725, 0.356431, 0.354145, 0.351866, 0.349595,
                 0.347331, 0.345076, 0.342828, 0.340588, 0.338356, 0.336131, 0.333915, 0.331706, 0.329506, 0.327314,
                 0.325129, 0.322953, 0.320785, 0.318625, 0.316473, 0.314329, 0.312194, 0.310067, 0.307949, 0.305838,
                 0.303736, 0.301643, 0.299558, 0.297481, 0.295413, 0.293353, 0.291302, 0.289259, 0.287225, 0.285200,
                 0.283183, 0.281174, 0.279175, 0.277184, 0.275201, 0.273228, 0.271263, 0.269307, 0.267359, 0.265420,
                 0.263490, 0.261569, 0.259657, 0.257754, 0.255859, 0.253973, 0.252096, 0.250228, 0.248368, 0.246518,
                 0.244676, 0.242844, 0.241020, 0.239205, 0.237399, 0.235602, 0.233814, 0.232034, 0.230264, 0.228503,
                 0.226750, 0.225007, 0.223272, 0.221546, 0.219829, 0.218121, 0.216422, 0.214732, 0.213051, 0.211379,
                 0.209716, 0.208061, 0.206416, 0.204779, 0.203152, 0.201533, 0.199923, 0.198322, 0.196730, 0.195146,
                 0.193572, 0.192006, 0.190450, 0.188902, 0.187363, 0.185832, 0.184311, 0.182798, 0.181294, 0.179799,
                 0.178313, 0.176835, 0.175366, 0.173906, 0.172454, 0.171011, 0.169577, 0.168152, 0.166735, 0.165326,
                 0.163927, 0.162535, 0.161153, 0.159779, 0.158413, 0.157056, 0.155708, 0.154368, 0.153036, 0.151713,
                 0.150399, 0.149092, 0.147794, 0.146505, 0.145224, 0.143951, 0.142686, 0.141430, 0.140182, 0.138942,
                 0.137710, 0.136487, 0.135271, 0.134064, 0.132865, 0.131674, 0.130491, 0.129316, 0.128149, 0.126990,
                 0.125839, 0.124696, 0.123561, 0.122433, 0.121314, 0.120202, 0.119098, 0.118002, 0.116914, 0.115833,
                 0.114760, 0.113695, 0.112637, 0.111587, 0.110544, 0.109509, 0.108482, 0.107462, 0.106449, 0.105444,
                 0.104447, 0.103456, 0.102473, 0.101497, 0.100529, 0.099568, 0.098614, 0.097667, 0.096727, 0.095794,
                 0.094869, 0.093950, 0.093039, 0.092134, 0.091237, 0.090346, 0.089463, 0.088586, 0.087716, 0.086853,
                 0.085996, 0.085146, 0.084303, 0.083467, 0.082637, 0.081814, 0.080998, 0.080188, 0.079384, 0.078587,
                 0.077797, 0.077013, 0.076235, 0.075463, 0.074698, 0.073939, 0.073186, 0.072440, 0.071700, 0.070966,
                 0.070238, 0.069516, 0.068800, 0.068090, 0.067386, 0.066688, 0.065996, 0.065309, 0.064629, 0.063954,
                 0.063285, 0.062622, 0.061965, 0.061313, 0.060667, 0.060026, 0.059392, 0.058762, 0.058138, 0.057520,
                 0.056907, 0.056299, 0.055697, 0.055100, 0.054508, 0.053922, 0.053341, 0.052765, 0.052194, 0.051629,
                 0.051068, 0.050513, 0.049962, 0.049417, 0.048876, 0.048341, 0.047810, 0.047284, 0.046764, 0.046247,
                 0.045736, 0.045230, 0.044728, 0.044231, 0.043738, 0.043250, 0.042767, 0.042288, 0.041814, 0.041344,
                 0.040879, 0.040418, 0.039961, 0.039509, 0.039061, 0.038618, 0.038178, 0.037743, 0.037313, 0.036886,
                 0.036463, 0.036045, 0.035631, 0.035220, 0.034814, 0.034412, 0.034014, 0.033619, 0.033229, 0.032842,
                 0.032460, 0.032081, 0.031706, 0.031334, 0.030966, 0.030603, 0.030242, 0.029886, 0.029533, 0.029183,
                 0.028837, 0.028495, 0.028156, 0.027821, 0.027489, 0.027160, 0.026835, 0.026513, 0.026195, 0.025879,
                 0.025567, 0.025259, 0.024953, 0.024651, 0.024352, 0.024056, 0.023763, 0.023474, 0.023187, 0.022903,
                 0.022623, 0.022345, 0.022071, 0.021799, 0.021530, 0.021264, 0.021001, 0.020741, 0.020484, 0.020229,
                 0.019977, 0.019728, 0.019482, 0.019238, 0.018997, 0.018758, 0.018523, 0.018289, 0.018059, 0.017831,
                 0.017605, 0.017382, 0.017161, 0.016943, 0.016728, 0.016514, 0.016304, 0.016095, 0.015889, 0.015685,
                 0.015484, 0.015284, 0.015087, 0.014893, 0.014700, 0.014510, 0.014321, 0.014135, 0.013952, 0.013770,
                 0.013590, 0.013413, 0.013237, 0.013064, 0.012892, 0.012723, 0.012555, 0.012389, 0.012226, 0.012064,
                 0.011904, 0.011746, 0.011590, 0.011436, 0.011284, 0.011133, 0.010984, 0.010837, 0.010692, 0.010548,
                 0.010407, 0.010266, 0.010128, 0.009991, 0.009856, 0.009722, 0.009591, 0.009460, 0.009332, 0.009204,
                 0.009079, 0.008955, 0.008832, 0.008711, 0.008592, 0.008474, 0.008357, 0.008242, 0.008128, 0.008016,
                 0.007905, 0.007795, 0.007687, 0.007580, 0.007474, 0.007370, 0.007267, 0.007166, 0.007065, 0.006966,
                 0.006868, 0.006772, 0.006676, 0.006582, 0.006489, 0.006397, 0.006307, 0.006217, 0.006129, 0.006042,
                 0.005956, 0.005871, 0.005787, 0.005704, 0.005623, 0.005542, 0.005462, 0.005384, 0.005306, 0.005230,
                 0.005154, 0.005080, 0.005006, 0.004933, 0.004862, 0.004791, 0.004721, 0.004652, 0.004585, 0.004518,
                 0.004451, 0.004386, 0.004322, 0.004258, 0.004195, 0.004134, 0.004073, 0.004012, 0.003953, 0.003894,
                 0.003837, 0.003780, 0.003723, 0.003668, 0.003613, 0.003559, 0.003506, 0.003453, 0.003402, 0.003351,
                 0.003300, 0.003250, 0.003201, 0.003153, 0.003105, 0.003058, 0.003012, 0.002966, 0.002921, 0.002877,
                 0.002833, 0.002790, 0.002747, 0.002705, 0.002664, 0.002623, 0.002582, 0.002543, 0.002504, 0.002465,
                 0.002427, 0.002389, 0.002352, 0.002316, 0.002280, 0.002245, 0.002210, 0.002175, 0.002141, 0.002108,
                 0.002075, 0.002042, 0.002010, 0.001979, 0.001948, 0.001917, 0.001887, 0.001857, 0.001828, 0.001799,
                 0.001770, 0.001742, 0.001715, 0.001687, 0.001661, 0.001634, 0.001608, 0.001582, 0.001557, 0.001532,
                 0.001508, 0.001483, 0.001459, 0.001436, 0.001413, 0.001390, 0.001368, 0.001345, 0.001324, 0.001302,
                 0.001281, 0.001260, 0.001240, 0.001220, 0.001200, 0.001180, 0.001161, 0.001142, 0.001123, 0.001105,
                 0.001086, 0.001069, 0.001051, 0.001034, 0.001017, 0.001000, 0.000983, 0.000967, 0.000951, 0.000935,
                 0.000919, 0.000904, 0.000889, 0.000874, 0.000860, 0.000845, 0.000831, 0.000817, 0.000803, 0.000790,
                 0.000777, 0.000764, 0.000751, 0.000738, 0.000726, 0.000713, 0.000701, 0.000689, 0.000678, 0.000666,
                 0.000655, 0.000643, 0.000633, 0.000622, 0.000611, 0.000601, 0.000590, 0.000580, 0.000570, 0.000560,
                 0.000551, 0.000541, 0.000532, 0.000523, 0.000514, 0.000505, 0.000496, 0.000487, 0.000479, 0.000471,
                 0.000462, 0.000454, 0.000446, 0.000439, 0.000431, 0.000423, 0.000416, 0.000409, 0.000401, 0.000394,
                 0.000387, 0.000381, 0.000374, 0.000367, 0.000361, 0.000354, 0.000348, 0.000342, 0.000336, 0.000330,
                 0.000324, 0.000318, 0.000312, 0.000307, 0.000301, 0.000296, 0.000291, 0.000285, 0.000280, 0.000275,
                 0.000270, 0.000265, 0.000261, 0.000256, 0.000251, 0.000247, 0.000242, 0.000238, 0.000233, 0.000229,
                 0.000225, 0.000221, 0.000217, 0.000213, 0.000209, 0.000205, 0.000201, 0.000198, 0.000194, 0.000191,
                 0.000187, 0.000184, 0.000180, 0.000177, 0.000174, 0.000170, 0.000167, 0.000164, 0.000161, 0.000158,
                 0.000155, 0.000152, 0.000149, 0.000147, 0.000144, 0.000141, 0.000139, 0.000136, 0.000133, 0.000131,
                 0.000128, 0.000126, 0.000124, 0.000121, 0.000119, 0.000117, 0.000114, 0.000112, 0.000110, 0.000108,
                 0.000106, 0.000104, 0.000102, 0.000100, 0.000098, 0.000096, 0.000094, 0.000093, 0.000091, 0.000089,
                 0.000087, 0.000086, 0.000084, 0.000082, 0.000081, 0.000079, 0.000078, 0.000076, 0.000075, 0.000073,
                 0.000072, 0.000071, 0.000069, 0.000068, 0.000066, 0.000065, 0.000064, 0.000063, 0.000061, 0.000060,
                 0.000059, 0.000058, 0.000057, 0.000056, 0.000055, 0.000053, 0.000052, 0.000051, 0.000050, 0.000049,
                 0.000048, 0.000047, 0.000046, 0.000046, 0.000045, 0.000044, 0.000043, 0.000042, 0.000041, 0.000040]
        li_ab_tensor = torch.tensor(li_ab)
        dis_vpsde = deis.DiscreteVPSDE(li_ab_tensor)
        return dis_vpsde
    t2alpha_fn, alpha2t_fn = deis.get_linear_alpha_fns(beta_0=0.01, beta_1=20)
    vpsde = deis.VPSDE(
        t2alpha_fn,
        alpha2t_fn,
        sampling_eps=0.,   # sampling end time t_0
        sampling_T=1.      # sampling starting time t_T
    )
    return vpsde

class DeisRunner:
    """"""
    def __init__(self, args, config, eps_fn, ts_phase_arr=None, ts_order_arr=None,
                 num_step_arr=None, method_arr=None, ab_order_arr=None, rk_method_arr=None):
        self.args = args
        self.config = config
        self.eps_fn = eps_fn
        self.sde = get_sde()
        self.ts_phase_arr = ts_phase_arr or ["t"]
        self.ts_order_arr = ts_order_arr or ["t"]
        self.num_step_arr = num_step_arr or [10]
        self.method_arr = method_arr or ["t_ab"]
        self.ab_order_arr = ab_order_arr or [1]
        self.rk_method_arr = rk_method_arr or ["3kutta"]
        self.ts_phase = ts_phase_arr[0]
        self.ts_order = ts_order_arr[0]
        self.num_step = num_step_arr[0]
        self.method = method_arr[0]
        self.ab_order = ab_order_arr[0]
        self.rk_method = rk_method_arr[0]
        log_fn(f"DeisRunner()...")
        log_fn(f"  ts_phase_arr : {ts_phase_arr}")
        log_fn(f"  ts_order_arr : {ts_order_arr}")
        log_fn(f"  num_step_arr : {num_step_arr}")
        log_fn(f"  method_arr   : {method_arr}")
        log_fn(f"  ab_order_arr : {ab_order_arr}")
        log_fn(f"  rk_method_arr: {rk_method_arr}")

    @staticmethod
    def load_predefined_aap(f_path: str, meta_dict=None):
        if not os.path.exists(f_path):
            raise Exception(f"File not found: {f_path}")
        if not os.path.isfile(f_path):
            raise Exception(f"Not file: {f_path}")
        if meta_dict is None:
            meta_dict = {}
        log_fn(f"Load file: {f_path}")
        with open(f_path, 'r') as f_ptr:
            lines = f_ptr.readlines()
        cnt_empty = 0
        cnt_comment = 0
        ab_arr = []  # alpha_bar array
        ts_arr = []  # timestep array
        for line in lines:
            line = line.strip()
            if line == '':
                cnt_empty += 1
                continue
            if line.startswith('#'):  # line is like "# order     : 2"
                cnt_comment += 1
                arr = line[1:].strip().split(':')
                key = arr[0].strip()
                if key in meta_dict: meta_dict[key] = arr[1].strip()
                continue
            arr = line.split(':')
            ab, ts = float(arr[0]), float(arr[1])
            ab_arr.append(ab)
            ts_arr.append(ts)
        ab2s = lambda ff: ' '.join([f"{f:8.6f}" for f in ff])
        ts2s = lambda ff: ' '.join([f"{f:10.5f}" for f in ff])
        log_fn(f"  cnt_empty  : {cnt_empty}")
        log_fn(f"  cnt_comment: {cnt_comment}")
        log_fn(f"  cnt_valid  : {len(ab_arr)}")
        log_fn(f"  ab[:5]     : [{ab2s(ab_arr[:5])}]")
        log_fn(f"  ab[-5:]    : [{ab2s(ab_arr[-5:])}]")
        log_fn(f"  ts[:5]     : [{ts2s(ts_arr[:5])}]")
        log_fn(f"  ts[-5:]    : [{ts2s(ts_arr[-5:])}]")
        for k, v in meta_dict.items():
            log_fn(f"  {k:11s}: {v}")
        return ab_arr, ts_arr

    def gen_sampler_fn(self, aap_file=None):
        if aap_file:
            meta_dict = {"ts_phase": None, "ts_order": None, "num_step": None,
                         "method": None, "ab_order": None, "rk_method": None}
            ab_arr, _ = self.load_predefined_aap(aap_file, meta_dict)
            ab_arr.insert(0, 0.9999)
            ab_arr.reverse()
            ts_arr = [self.sde.alpha2t_fn(ab) for ab in ab_arr]
            deis.sampler._rev_ts = ts_arr
            if meta_dict['ts_phase']:  self.ts_phase = meta_dict['ts_phase']
            if meta_dict['ts_order']:  self.ts_order = float(meta_dict['ts_order'])
            if meta_dict['num_step']:  self.num_step = int(meta_dict['num_step'])
            if meta_dict['method']:    self.method = meta_dict['method']
            if meta_dict['ab_order']:  self.ab_order = int(meta_dict['ab_order'])
            if meta_dict['rk_method']: self.rk_method = meta_dict['rk_method']
        sampler_fn = deis.get_sampler(
            self.sde,
            self.eps_fn,
            ts_phase=self.ts_phase,   # support "rho", "t", "log"
            ts_order=self.ts_order,
            num_step=self.num_step,
            method=self.method,       # deis sampling algorithms: support "rho_rk", "rho_ab", "t_ab", "ipndm"
            ab_order=self.ab_order,   # gt 0, used for "rho_ab", "t_ab", other algorithms will ignore it
            rk_method=self.rk_method  # used for "rho_rk" algorithms, other algorithms will ignore the arg
        )
        return sampler_fn

    def sample_baseline(self):
        def save_result(_msg_arr, _fid_arr):
            with open('./sample_all_result.txt', 'w') as f_ptr:
                [f_ptr.write(f"# {m}\n") for m in _msg_arr]
                [f_ptr.write(f"[{dt}] {a:8.4f} {s:.4f}: {k}\n") for dt, a, s, k in _fid_arr]
            # with
        msg_arr = [f"  ts_phase_arr : {self.ts_phase_arr}",
                   f"  ts_order_arr : {self.ts_order_arr}",
                   f"  num_step_arr : {self.num_step_arr}",
                   f"  method_arr   : {self.method_arr}",
                   f"  ab_order_arr : {self.ab_order_arr}",
                   f"  rk_method_arr: {self.rk_method_arr}"]
        fid_arr = []
        for ts_phase in self.ts_phase_arr:
            self.ts_phase = ts_phase
            for ts_order in self.ts_order_arr:
                self.ts_order = ts_order
                for num_step in self.num_step_arr:
                    self.num_step = num_step
                    for method in self.method_arr:
                        self.method = method
                        for ab_order in self.ab_order_arr:
                            self.ab_order = ab_order
                            for rk_method in self.rk_method_arr:
                                self.rk_method = rk_method
                                key, avg, std = self.sample_times()
                                dtstr = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                                fid_arr.append([dtstr, avg, std, key])
                                save_result(msg_arr, fid_arr)
                            # for
                        # for
                    # for
                # for
            # for
        # for

    def alpha_bar_all(self):
        def save_ab_file(file_path, ts_list):
            if len(ts_list) != self.num_step:
                raise Exception(f"alpha_bar count {len(ts_list)} not match steps {self.num_step}")
            ts_list.sort()  # timestep increase, alpha_bar decrease
            with open(file_path, 'w') as f_ptr:
                f_ptr.write(f"# ts_phase : {self.ts_phase}\n")
                f_ptr.write(f"# ts_order : {self.ts_order}\n")
                f_ptr.write(f"# num_step : {self.num_step}\n")
                f_ptr.write(f"# method   : {self.method}\n")
                f_ptr.write(f"# ab_order : {self.ab_order}\n")
                f_ptr.write(f"# rk_method: {self.rk_method}\n")
                f_ptr.write(f"\n")
                f_ptr.write(f"# alpha_bar : index\n")
                for idx in ts_list:
                    idx = idx.cpu().numpy()
                    ab = self.sde.t2alpha_fn(idx)
                    f_ptr.write(f"{ab:.8f}  : {idx:10.5f}\n")
            # with
        # def
        ab_dir = self.args.ab_original_dir or '.'
        if not os.path.exists(ab_dir):
            log_fn(f"os.makedirs({ab_dir})")
            os.makedirs(ab_dir)
        global _ts_list
        for ts_phase in self.ts_phase_arr:
            self.ts_phase = ts_phase
            for ts_order in self.ts_order_arr:
                self.ts_order = ts_order
                for num_step in self.num_step_arr:
                    self.num_step = num_step
                    for method in self.method_arr:
                        self.method = method
                        for ab_order in self.ab_order_arr:
                            self.ab_order = ab_order
                            for rk_method in self.rk_method_arr:
                                self.rk_method = rk_method
                                _ts_list = []
                                self.sample(sample_count=1)
                                key = self.config_key_str()
                                f_path = os.path.join(ab_dir, f"{key}.txt")
                                save_ab_file(f_path, _ts_list)
                                log_fn(f"File saved: {f_path}")
                                _ts_list = None
                            # for
                        # for
                    # for
                # for
            # for
        # for

    def config_key_str(self):
        ks = f"ts_{self.ts_phase}-{self.ts_order}_s{self.num_step:02d}" \
             f"_{self.method}_{self.ab_order}_{self.rk_method}"
        return ks

    def sample_times(self, times=None, aap_file=None):
        args = self.args
        times = times or args.repeat_times
        fid_arr = []
        ss = self.config_key_str()
        input1, input2 = args.fid_input1 or 'cifar10-train', args.sample_output_dir
        sampler_fn = self.gen_sampler_fn(aap_file)
        for i in range(times):
            self.sample(sampler_fn)
            log_fn(f"{ss}-{i}/{times} => FID calculating...")
            log_fn(f"  input1: {input1}")
            log_fn(f"  input2: {input2}")
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=input1,
                input2=input2,
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                samples_find_deep=True,
            )
            fid = metrics_dict['frechet_inception_distance']
            log_fn(f"{ss}-{i}/{times} => FID: {fid:.6f}")
            fid_arr.append(fid)
        # for
        avg, std = np.mean(fid_arr), np.std(fid_arr)
        return ss, avg, std

    def sample(self, sampler_fn=None, sample_count=None, aap_file=None):
        args, config = self.args, self.config
        if sampler_fn is None:
            sampler_fn = self.gen_sampler_fn(aap_file)
        if sample_count is None:
            sample_count = args.sample_count
        b_sz  = args.sample_batch_size
        b_cnt = (sample_count - 1) // b_sz + 1
        log_fn(f"deis_runner::sample()...")
        log_fn(f"  sample_ckpt_path : {args.sample_ckpt_path}")
        log_fn(f"  sample_output_dir: {args.sample_output_dir}")
        log_fn(f"  sample_count     : {sample_count}")
        log_fn(f"  sample_batch_size: {args.sample_batch_size}")
        log_fn(f"  batch_count      : {b_cnt}")
        time_start = time.time()
        d = config.data
        for b_idx in range(b_cnt):
            n = b_sz if b_idx + 1 < b_cnt else sample_count - b_idx * b_sz
            noise = torch.randn(n, d.channels, d.image_size, d.image_size, device=args.device)
            sample = sampler_fn(noise)
            self.save_images(sample, args.sample_output_dir, b_idx, b_sz, b_cnt, time_start)

    @staticmethod
    def save_images(x, img_dir, b_idx, b_sz, b_cnt, time_start):
        x = (x + 1.0) / 2.0           # from [-1, 1] to [0, 1]
        x = torch.clamp(x, 0.0, 1.0)  # make sure range is [0, 1]
        if not os.path.exists(img_dir):
            log_fn(f"os.makedirs({img_dir})")
            os.makedirs(img_dir)
        img_cnt = len(x)
        img_path = None
        for i in range(img_cnt):
            img_id = b_idx * b_sz + i
            img_path = os.path.join(img_dir, f"{img_id:05d}.png")
            tvu.save_image(x[i], img_path)
        elp, eta = utils.get_time_ttl_and_eta(time_start, b_idx + 1, b_cnt)
        log_fn(f"saved {img_cnt} images: {img_path}. elp:{elp}, eta:{eta}")
