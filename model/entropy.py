import torch

def entropy_local(mean, var, thr, device, dtype):
    # Entropy of Bernoulli distribution with prob p = posterior(x) cdf @ thr
    entropy = torch.zeros(mean.shape[0]).to(device, dtype)

    for i in range(entropy.shape[0]):
        try:
            normal = torch.distributions.Normal(loc=mean[i], scale=var[i] ** 0.5)
            p0 = 1 - normal.cdf(thr)
            p1 = normal.cdf(thr) - 0
            entropy[i] = -p0 * torch.log(p0) - p1 * torch.log(p1)

        except:
            entropy[i] = 0

    # NaN arises from very small p, i.e. x is sure to be above/under thr
    entropy = torch.nan_to_num(entropy, nan=0)
    return entropy