from reggae.psdmodel import PSDModel
from reggae.dipolestar import DipoleStar

class MagneticDipoleStar(DipoleStar):

    _psdmodel = MagneticPsd
    _extra_ndim = 3

    @partial(jax.jit, static_argnums=(0,))
    def ptform(self, u):
        """ Turns coordinates on the unit cube into θ_reg object

        Parameters
        ----------
        u: list
            Point location in the unit hypercube.

        Returns
        -------
        x: jnp.array
            Point location in model parameter space.
        """

        θ_reg = ThetaReg.prior_transform(u[:ThetaReg.dims], bounds=self.bounds)

        norm = jnp.array([self.bounds[-1][0] + u[-1] * (self.bounds[-1][1] - self.bounds[-1][0])])
        dnu_mag = ...
        a = ...
        return jnp.concatenate((θ_reg, norm, dnu_mag, a))

    @partial(jax.jit, static_argnums=(0,))
    def model(self, theta):
        """ Construct the model for the likelihood evaluation

        Parameters
        ----------
        theta: jnp.array
            Array of model parameters.

        Returns
        -------
        mod: jnp.array
            Spectrum model.
        """

        return self.l1model.l1model(self.theta_asy, ThetaReg(*theta[:9]), *theta[10:12]) * theta[9] + 1


class MagneticPsd(PSDModel):

    def l1model(self, theta_asy, theta_reg, dnu_mag, a, **kwargs):
        """ Create an l=1 multiplet

        Constructs a triplet of m=-1, 0, and 1, split by a linear combination of
        the core and envelope rotational splitting. The relative power density is
        modulated the inclination of the stellar rotation axis.

        Parameters
        ----------
        theta_asy: data class
            Data class containing a sample of the asymptotic parameters for the p-modes.
        theta_reg: data class
            Data class containing a sample of l=1 mixing model parameters.
        kwargs : dict
            Additional keyword arguments to be passed to _l1model.

        Returns
        -------
        l1model : jnp.array
            The spectrum model of a single l=1 multiplet.
        """

        dnu_g = 10.**theta_reg.log_omega_core /  reggae.nu_to_omega

        dnu_p = 10.**theta_reg.log_omega_env /  reggae.nu_to_omega

        inc = theta_reg.inclination

        numax = 10. ** (theta_asy.log_numax)
        deltaPi0 = UNITS['DPI0'] * theta_reg.dPi0  # in inverse μHz
        epsilon_g = theta_reg.epsilon_g
        if self.n_g is None:
            self.update_n_g(theta_asy, theta_reg)
        n_g = self.n_g
        nu_g = reggae.asymptotic_nu_g(n_g, deltaPi0, jnp.inf, epsilon_g, numax=numax, alpha=alpha_g)

        return (self._l1model(theta_asy, theta_reg, dnu_g=(1-a)*dnu_mag * (numax / nu_g) ** 3, dnu_p=0, **kwargs) * jnp.cos(inc)**2

              + self._l1model(theta_asy, theta_reg, dnu_g=-dnu_g + (1+a/2)*dnu_mag * (numax / nu_g) ** 3, dnu_p=-dnu_p, **kwargs) * jnp.sin(inc)**2 / 2

              + self._l1model(theta_asy, theta_reg, dnu_g= dnu_g + (1+a/2)*dnu_mag * (numax / nu_g) ** 3, dnu_p=dnu_p,**kwargs) * jnp.sin(inc)**2 / 2
               )