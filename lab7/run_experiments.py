"""Run the prescribed Lab 7 experiments; all numerical work uses float64 PyTorch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import torch
import lab7 as lab

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'


def load_lab3():
    spec = importlib.util.spec_from_file_location('previous_lab3', HERE.parent / 'lab3/lab3.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rmse(a, b):
    return float((a-b).square().mean().sqrt())


def lag_one(values):
    a = torch.tensor(values, dtype=torch.float64)
    return float(torch.corrcoef(torch.stack((a[:-1], a[1:])))[0, 1])


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_dtype(torch.float64)
    start = time.monotonic()
    images, histories = {}, {}
    summary = {'base_seed': 7, 'seeds': {'gmrf_each': 7, 'qggmrf_each': 7,
               'measurement_noise': 7, 'posterior_moments': 8,
               'posterior_start_y': 9, 'posterior_start_zero': 10},
               'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                               'device': 'cpu', 'dtype': 'float64', 'threads': 1},
               'gmrf': {}, 'qggmrf': {}}
    with torch.no_grad():
        for sigma in (.05, .01, .2):
            torch.manual_seed(7)
            x = torch.zeros((256, 256))
            means, stds, energies = [], [], []
            for sweep in range(500 if sigma == .05 else 200):
                x = lab.gibbs_sweep_gmrf(x, sigma)
                means.append(float(x.mean()))
                stds.append(float((x-x.mean()).std(correction=0)))
                energies.append(float(lab.gmrf_energy(x, sigma)))
                if sweep == 199:
                    images[f'gmrf_{sigma}'] = x.clone()
            shown = images[f'gmrf_{sigma}']
            centered = shown-shown.mean()
            summary['gmrf'][str(sigma)] = {'raw_mean': float(shown.mean()),
                'display_range': [float(centered.min()), float(centered.max())],
                'std': float(centered.std(correction=0))}
            if sigma == .05:
                histories['prior_mean'], histories['prior_std'] = means, stds
                histories['prior_energy'] = energies
            print(f'GMRF sigma={sigma} done', flush=True)

        for width in (.002, .02, .2):
            torch.manual_seed(7)
            x = torch.zeros((256, 256))
            acceptance, energy, jumps = [], [], []
            for sweep in range(300):
                previous = x
                x, rate = lab.metropolis_sweep_qggmrf(x, .05, 1.2, 2., .1, width)
                acceptance.append(rate)
                energy.append(float(lab.qggmrf_energy(x, .05, 1.2, 2., .1)))
                jumps.append(float((x-previous).square().mean()))
            key = str(width)
            images[f'qggmrf_{width}'] = x
            histories[f'qggmrf_{width}'] = {'acceptance': acceptance, 'energy': energy,
                                           'mean_squared_jump': jumps}
            centered = x-x.mean()
            summary['qggmrf'][key] = {
                'acceptance_all': sum(acceptance)/300,
                'acceptance_last100': sum(acceptance[-100:])/100,
                'energy_final': energy[-1], 'energy_last100_mean': sum(energy[-100:])/100,
                'energy_lag1_last100': lag_one(energy[-100:]),
                'mean_squared_jump_last100': sum(jumps[-100:])/100,
                'display_range': [float(centered.min()), float(centered.max())],
                'horizontal_difference_std': float((x[:, 1:]-x[:, :-1]).std(correction=0))}
            print(f'QGGMRF width={width} {summary["qggmrf"][key]}', flush=True)
        best = max(summary['qggmrf'], key=lambda k: summary['qggmrf'][k]['mean_squared_jump_last100'])
        summary['best_proposal'] = float(best)

        old = load_lab3()
        source = HERE.parent / 'lab3/kodim23.pgm'
        full = old.load_pgm(source)
        clean = full[128:384, 256:512].clone()
        summary['data'] = {'source': 'lab3/kodim23.pgm', 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'original_shape': list(full.shape), 'crop': 'rows [128:384), columns [256:512), zero based'}
        torch.manual_seed(7)
        noisy = clean + .05 * torch.randn_like(clean)
        torch.manual_seed(8)
        mean, std, posterior_energies = lab.posterior_statistics(noisy, .05, .05, 100, 400)
        images.update(clean=clean, noisy=noisy, posterior_mean=mean, posterior_std=std)
        histories['posterior_statistics_energy'] = posterior_energies
        kernel = torch.ones((1, 1))
        map_image, map_history = old.gradient_descent(noisy, kernel, .05, .05, 200)
        images['map'] = map_image
        histories['map_cost'] = map_history
        summary['map_gradient_max'] = float(old.cost_grad(map_image, noisy, kernel, .05, .05).abs().max())

        for label, seed in (('y', 9), ('zero', 10)):
            torch.manual_seed(seed)
            x = noisy.clone() if label == 'y' else torch.zeros_like(noisy)
            energies, means = [], []
            for sweep in range(500):
                x = lab.gibbs_sweep_posterior(x, noisy, .05, .05)
                energies.append(float(lab.posterior_energy(x, noisy, .05, .05)))
                means.append(float(x.mean()))
            images[f'posterior_final_{label}'] = x
            histories[f'start_{label}'] = {'energy': energies, 'mean': means}
            print(f'Posterior start={label} done', flush=True)
        images['single_sample'] = images['posterior_final_y']
        edge = torch.zeros_like(std, dtype=torch.bool)
        edge[0, :], edge[-1, :], edge[:, 0], edge[:, -1] = True, True, True, True
        summary['posterior'] = {'rmse': {key: rmse(images[key], clean) for key in ('noisy', 'single_sample', 'posterior_mean', 'map')},
            'mean_map_rmse': rmse(mean, map_image), 'std_range': [float(std.min()), float(std.max())],
            'std_boundary_mean': float(std[edge].mean()), 'std_interior_mean': float(std[~edge].mean()),
            'mean_std_squared': float(std.square().mean()),
            'roughness': {key: float((images[key][:, 1:]-images[key][:, :-1]).square().mean().sqrt())
                          for key in ('single_sample', 'posterior_mean', 'map')},
            'prior_mean_range': [min(histories['prior_mean']), max(histories['prior_mean'])],
            'posterior_mean_range_after100': [min(histories['start_y']['mean'][100:]), max(histories['start_y']['mean'][100:])],
            'data_mean': float(noisy.mean())}
        for label in ('y', 'zero'):
            values = torch.tensor(histories[f'start_{label}']['energy'])
            summary['posterior'][f'energy_{label}_last400_mean'] = float(values[100:].mean())
            summary['posterior'][f'energy_{label}_last400_std'] = float(values[100:].std(correction=0))
        summary['elapsed_seconds'] = time.monotonic()-start
        torch.save(images, RESULTS / 'images.pt')
        (RESULTS / 'histories.json').write_text(json.dumps(histories, indent=2)+'\n')
        (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary['posterior'], indent=2), flush=True)
        print(f'Completed in {summary["elapsed_seconds"]:.1f} s', flush=True)


if __name__ == '__main__':
    main()
