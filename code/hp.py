samples = [result.posterior for result in results]

def hyper_prior(dataset, mu, sigma):
    return (
        numpy.exp(-((dataset["x"] - mu) ** 2) / (2 * sigma**2))
        / (2 * numpy.pi * sigma**2) ** 0.5
    )

for sample in samples:
    #sample["prior"] = 1 / (x_prior_max-x_prior_min)
    sample["log_prior"] = numpy.log(1 / (x_prior_max-x_prior_min))

evidences = [result.log_evidence for result in results]

hp_likelihood = bilby.hyper.likelihood.HyperparameterLikelihood(
    posteriors=samples,
    hyper_prior=hyper_prior,
    log_evidences=evidences,
    max_samples=500,
)

hp_priors = dict(mu=bilby.core.prior.Uniform(-1.0, 1.0, 'mu', r'$\mu$'),
                 sigma=bilby.core.prior.Uniform(0.0, 1.0, 'sigma', r'$\sigma$'))

result = bilby.run_sampler(
    likelihood=hp_likelihood,
    priors=hp_priors,
    sampler='dynesty',
    nlive=1000,
    use_ratio=False,
    outdir=outdir,
    label=label,
    verbose=True,
    clean=True,
    plot=True,
)
