import torch

def entropy_local(mean, var, thr, device, dtype):
    # Entropy of Bernoulli distribution with prob p = posterior(x) cdf @ thr
    entropy = torch.zeros(mean.shape[0]).to(device, dtype)

    for i in range(entropy.shape[0]):
        try:
            normal = torch.distributions.Normal(loc=mean[i].to(device, dtype), scale=(var[i] ** 0.5).to(device, dtype))
            p0 = (1 - normal.cdf(thr)).to(device, dtype)
            p1 = normal.cdf(thr).to(device, dtype)
            # Adding debug print statements
            print(f"Index {i}: mean={mean[i].to(device, dtype)}, var={var[i].to(device, dtype)}, p0={p0}, p1={p1}")

            # Ensure p0 and p1 are not zero to avoid log(0)
            if p0 > 0 and p1 > 0:
                entropy[i] = (-p0 * torch.log(p0) - p1 * torch.log(p1)).to(device, dtype)
                print(f"Index {i}: entropy={entropy[i]}")
            else:
                entropy[i] = torch.tensor(0.0).to(device, dtype)
                print(f"Index {i}: entropy set to 0 due to p0 or p1 being zero or negative")

        except Exception as e:
            print(f"Error at index {i}: {e}")
            entropy[i] = torch.tensor(0.0).to(device, dtype)

    # NaN arises from very small p, i.e. x is sure to be above/under thr
    entropy = torch.nan_to_num(entropy, nan=0).to(device, dtype)
    return entropy

