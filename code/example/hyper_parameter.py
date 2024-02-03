import os
import bilby 
import numpy as np
import pandas as pd

def hyper_prior(dataset, mu, sigma):
    return (np.exp(- (dataset['zeta'] - mu)**2 / (2 * sigma**2)) /
        (2 * np.pi * sigma**2)**0.5)

outdir = 'outdir'
label = 'hyper'

bilby.core.utils.setup_logger(outdir=outdir, label=label)




directory = '/home/zjcao/zhaozc/CPT_O3b/runs/'
events_name_list = sorted(os.listdir(directory))

events_name_list.remove('GW191204_171526')
events_name_list.remove('GW191204_171526_More_Point_stricter')



result_file_list = [directory + name + '/run1/outdir/result/sola_0_result.json' for name in events_name_list]
flag = [os.path.exists(file) for file in result_file_list]


if False not in flag:
    results = [bilby.result.read_in_result(file) for file in result_file_list]

    samples = [pd.DataFrame(result.posterior['zeta']) for result in results]
    evidences = [result.log_evidence for result in results]
    for sample in samples:
        sample["log_prior"] = np.log(1 / 0.2)

    hp_likelihood = bilby.hyper.likelihood.HyperparameterLikelihood(
    posteriors=samples, hyper_prior=hyper_prior,
    log_evidences=evidences)


    hp_priors = dict(mu=bilby.core.prior.Uniform(-1.0, 1.0, 'mu', '$\mu$'),
                 sigma=bilby.core.prior.Uniform(0.0, 1.0, 'sigma', '$\sigma$'))

    # And run sampler

    if __name__ == '__main__':
        result = bilby.run_sampler(
            likelihood=hp_likelihood, priors=hp_priors, sampler='dynesty', nlive=1000,
            use_ratio=False, outdir=outdir, label=label,npool = 8,
            verbose=True, clean=True)
        
        result.plot_corner()
        
        
