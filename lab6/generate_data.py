"""Generate the lab's data once, preserving interleaved RNG draw order."""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TRUE_WEIGHTS = np.array([0.4, 0.4, 0.2])
TRUE_MEANS = np.array([[2., 2.], [-2., -2.], [5.5, 2.]])
TRUE_COVARIANCES = np.array([[[1., .1], [.1, 1.]],
                             [[1., -.1], [-.1, 1.]],
                             [[1., .2], [.2, .5]]])

def main():
    path = HERE / 'results/data.npz'
    if path.exists():
        print(f'Reusing {path}')
        return
    rng = np.random.default_rng(0)
    x = np.empty(500, dtype=np.int64)
    y = np.empty((500, 2), dtype=np.float64)
    for n in range(500):
        x[n] = rng.choice(3, p=TRUE_WEIGHTS)
        y[n] = rng.multivariate_normal(TRUE_MEANS[x[n]], TRUE_COVARIANCES[x[n]])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, y=y, x=x)
    print('Saved', path, 'class counts:', np.bincount(x).tolist())

if __name__ == '__main__':
    main()
