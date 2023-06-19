import unittest
import numpy as np
import psychofit as psy


class PsychofitTest(unittest.TestCase):
    def setUp(self) -> None:
        """Data are 3 x n arrays"""
        data = {'weibull50': np.vstack([
            10 ** np.linspace(-4, -1, 8),
            np.ones(8) * 80,
            np.array([0.5125, 0.35, 0.5625, 0.5375, 0.8875, 0.8875, 0.9125, 0.8625])
        ]), 'weibull': np.vstack([
            10 ** np.linspace(-4, -1, 8),
            np.ones(8) * 80,
            np.array([0.125, 0.1125, 0.1375, 0.4, 0.8125, 0.9, 0.925, 0.875])
        ]), 'erf_psycho_2gammas': np.vstack([
            np.arange(-50, 50, 10),
            np.ones(10) * 40,
            np.array([0.175, 0.225, 0.35, 0.275, 0.725, 0.9, 0.925, 0.975, 1., 1.])
        ]), 'erf_psycho': np.vstack([
            np.arange(-50, 50, 10),
            np.ones(10) * 40,
            np.array([0.1, 0.125, 0.25, 0.15, 0.6, 0.65, 0.75, 0.9, 0.9, 0.85])
        ])}
        self.test_data = data
        np.random.seed(0)

    def test_weibull50(self):
        xx = self.test_data['weibull50'][0, :]

        # test parameters
        alpha = 10 ** -2.5
        beta = 2.
        gamma = 0.1

        # fake experimental data given those parameters
        actual = psy.weibull50((alpha, beta, gamma), xx)
        expected = np.array(
            [0.5003998, 0.50286841, 0.5201905, 0.62446761, 0.87264857, 0.9, 0.9, 0.9]
        )
        self.assertTrue(np.allclose(expected, actual))

        self.assertRaises(ValueError, psy.weibull50, (alpha, beta), xx)
        self.assertRaises(TypeError, psy.weibull50, None, xx)

    def test_weibull(self):
        xx = self.test_data['weibull'][0, :]

        # test parameters
        alpha = 10 ** -2.5
        beta = 2.
        gamma = 0.1

        # fake experimental data given those parameters
        actual = psy.weibull((alpha, beta, gamma), xx)
        expected = np.array(
            [0.1007996, 0.10573682, 0.14038101, 0.34893523, 0.84529714, 0.9, 0.9, 0.9]
        )
        self.assertTrue(np.allclose(expected, actual))

        self.assertRaises(ValueError, psy.weibull, (alpha, beta), xx)
        self.assertRaises(TypeError, psy.weibull, None, xx)

    def test_erf_psycho(self):
        xx = self.test_data['erf_psycho'][0, :]

        # test parameters
        bias = -10.
        threshold = 20.
        lapse = .1

        # fake experimental data given those parameters
        actual = psy.erf_psycho((bias, threshold, lapse), xx)
        expected = np.array(
            [0.10187109, 0.11355794, 0.16291968, 0.29180005, 0.5,
             0.70819995, 0.83708032, 0.88644206, 0.89812891, 0.89983722]
        )
        self.assertTrue(np.allclose(expected, actual))

        self.assertRaises(TypeError, psy.erf_psycho, None, xx)
        with self.assertRaises(ValueError):
            psy.erf_psycho((bias, threshold, lapse, lapse), xx)

    def test_erf_psycho_2gammas(self):
        xx = self.test_data['erf_psycho_2gammas'][0, :]

        # test parameters
        bias = -10.
        threshold = 20.
        gamma1 = .2
        gamma2 = 0.

        # fake experimental data given those parameters
        actual = psy.erf_psycho_2gammas((bias, threshold, gamma1, gamma2), xx)
        expected = np.array(
            [0.20187109, 0.21355794, 0.26291968, 0.39180005, 0.6,
             0.80819995, 0.93708032, 0.98644206, 0.99812891, 0.99983722]
        )
        self.assertTrue(np.allclose(expected, actual))

        self.assertRaises(TypeError, psy.erf_psycho_2gammas, None, xx)
        with self.assertRaises(ValueError):
            psy.erf_psycho_2gammas((bias, threshold, gamma1), xx)

    def test_neg_likelihood(self):
        data = self.test_data['erf_psycho']
        self.assertRaises(ValueError, psy.neg_likelihood, (10, 20, .05), data[1:, :])
        self.assertRaises(TypeError, psy.neg_likelihood, '(10, 20, .05)', data)
        self.assertRaises(TypeError, psy.neg_likelihood, '(10, 20, .05)', None)
        self.assertRaises(ValueError, psy.neg_likelihood, (.5, 10, .05), data, P_model='foo')

        ll = psy.neg_likelihood((-20, 30, 2), data.tolist(), P_model='erf_psycho',
                                parmin=np.array((-10, 20, 0)), parmax=np.array((10, 10, .05)))
        self.assertTrue(ll > 10000)

    def test_mle_fit_psycho(self):
        expected = {
            'weibull50': (np.array([0.0034045, 3.9029162, .1119576]), -334.1149693046583),
            'weibull': (np.array([0.00316341, 1.72552866, 0.1032307]), -261.235178611311),
            'erf_psycho': (np.array([-9.78747259, 10., 0.15967605]), -193.0509031440323),
            'erf_psycho_2gammas': (np.array([-11.45463779, 9.9999999, 0.24117732, 0.0270835]),
                                   -147.02380025592902)
        }
        for model in self.test_data.keys():
            pars, L = psy.mle_fit_psycho(self.test_data[model], P_model=model, nfits=10)
            expected_pars, expected_L = expected[model]
            self.assertTrue(np.allclose(expected_pars, pars, atol=1e-3),
                            f'unexpected pars for {model}')
            self.assertTrue(np.isclose(expected_L, L, atol=1e-3),
                            f'unexpected likelihood for {model}')

        # Test one of the models with function pars
        params = {
            'parmin': np.array([-5., 10., 0.]),
            'parmax': np.array([5., 15., .1]),
            'parstart': np.array([0., 11., 0.1]),
            'nfits': 5
        }
        model = 'erf_psycho'
        pars, L = psy.mle_fit_psycho(self.test_data[model].tolist(), P_model=model, **params)
        expected = [-5, 15, 0.1]
        self.assertTrue(np.allclose(expected, pars, rtol=.01), f'unexpected pars for {model}')
        self.assertTrue(np.isclose(-195.55603, L, atol=1e-5), f'unexpected likelihood for {model}')

        # Test input validation
        self.assertRaises(ValueError, psy.mle_fit_psycho, np.zeros((4, 4)))  # wrong shape
        self.assertRaises(TypeError, psy.mle_fit_psycho, None)  # wrong type

    def tearDown(self):
        np.random.seed()


if __name__ == '__main__':
    unittest.main()
