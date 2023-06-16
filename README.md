
# psychofit
[![Coverage Status](https://coveralls.io/repos/github/cortex-lab/psychofit/badge.svg?branch=main)](https://coveralls.io/github/cortex-lab/psychofit?branch=main)
![CI workflow](https://github.com/cortex-lab/psychofit/actions/workflows/main.yaml/badge.svg?branch=main)

A module for fitting 2AFC psychometric data

The psychofit module contains tools to fit two-alternative psychometric
data. The fitting is done using maximal likelihood estimation: one
assumes that the responses of the subject are given by a binomial
distribution whose mean is given by the psychometric function.

The data can be expressed in fraction correct (from 50 to 100%) or in
fraction of one specific choice (from 0 to 100%). To fit them you can use
these functions:

 - `weibull50`          - Weibull function from 0.5 to 1, with lapse rate 
 - `weibull`            - Weibull function from 0 to 1, with lapse rate  
 - `erf_psycho`         - erf function from 0 to 1, with lapse rate
 - `erf_psycho_2gammas` - erf function from 0 to 1, with two lapse rates

Functions in the toolbox are:

 - `mle_fit_psycho`     - Maximumum likelihood fit of psychometric function
 - `neg_likelihood`     - Negative likelihood of a psychometric function

For more info, see:
  Examples           - Examples of use of psychofit toolbox

Matteo Carandini (2000-2017) initial Matlab code<br>
Miles Wells (2017-2018) ported to Python
