import numpy as np
import scipy.stats


def _corr_vectorized(a, b):
    """
    Vectorized correlation of a with each column of b.
    Faster than looping over scipy.stats.pearsonr.
    """
    a = np.asarray(a)
    b = np.asarray(b)

    a_mean = a.mean()
    b_mean = b.mean(axis=0)

    num = np.sum((a[:, None] - a_mean) * (b - b_mean), axis=0)
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2, axis=0))

    r = num / den
    r = np.clip(r, -1.0, 1.0)  # numerical stability

    # Compute p-values for Pearson correlation
    n = len(a)
    df = n - 2
    t = r * np.sqrt(df / (1 - r**2))
    p = 2 * scipy.stats.t.sf(np.abs(t), df)

    return r, p


def AFQ_MultiCompCorrection(data=None, y=None, alpha=0.05, cThresh=None, nperm=1000):
    """
    Optimized permutation-based multiple comparison correction.
    """
    if cThresh is None:
        cThresh = alpha

    n, m = data.shape

    if y is None or len(y) == 0:
        y = np.random.randn(n)
        print('No behavioral data provided so randn will be used')
        stattest = 'corr'
    else:
        y = np.asarray(y)
        if np.array_equal(np.unique(y), [0, 1]):
            stattest = 'ttest'
        else:
            stattest = 'corr'

    p = np.zeros((nperm, m))
    stat = np.zeros((nperm, m))
    clusMax = np.zeros(nperm)
    stats = {}

    rng = np.random.default_rng()

    if stattest == 'corr':
        for ii in range(nperm):
            rows = rng.permutation(n)
            stat[ii, :], p[ii, :] = _corr_vectorized(y, data[rows, :])

    else:  # independent t-test
        for ii in range(nperm):
            perm = rng.permutation(y)
            mask = perm > 0
            ttest_res = scipy.stats.ttest_ind(data[mask, :], data[~mask, :], axis=0, equal_var=False)
            p[ii, :] = ttest_res.pvalue
            stat[ii, :] = ttest_res.statistic

    # Sort results
    stats["pMin"] = np.sort(p.min(axis=1))
    stats["statMax"] = np.sort(stat.max(axis=1))[::-1]
    alphaFWE = stats["pMin"][int(round(alpha * nperm))]
    statFWE = stats["statMax"][int(round(alpha * nperm))]

    # Cluster correction
    pThresh = p < cThresh
    for ii in range(nperm):
        arr = np.r_[0, pThresh[ii, :].astype(int), 0]
        clusSiz = np.diff(np.flatnonzero(arr == 0))
        clusMax[ii] = clusSiz.max() if len(clusSiz) > 0 else 0

    stats["clusMax"] = np.sort(clusMax)[::-1]
    clusterFWE = stats["clusMax"][int(round(alpha * nperm))]

    return alphaFWE, statFWE, clusterFWE, stats
