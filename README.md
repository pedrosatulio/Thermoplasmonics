# Thermoplasmonics (v1.1.0)

_Thermoplasmonics_ is a library developed during the course of my graduate studies to aid theoretical investigations in the field of photothermal heating of plamosmonic nanoparticles and provide a framework to standarize image/plot formatting and perform data processing.

The library relies on [Mie][#1] and [Mie-Gans][#2] theories to calculate the optical cross sections of NPs (mostly plasmonic) and translate the results to thermal simulation environments for further photothermal conversion analysis. [Figures of merit (FoM)][#3] are extensively used for characterization, and collective temperature increase is also modeled.

Furthermore, a collection of refractive indexes for different optical materials is included, and their permittivities are compensated following the [size dependent Drude model][#4]. While the computation of nanospheres and nanoshells use Mie theory, nanorods are modeled as prolate [spheroids][#5] with optical cross sections determined by [Rayleigh-Gans approximation][#6] with polarizability correction using [Modified Long Wavelength Approximation (MLWA)][#7].

Moreover, the framework integrate optical calculations to a thermal module that uses the [Finite Difference Method (FDM)][#8] to simulate [Direct Absorption Solar Collectors (DASCs)][#9] and extract temperature increase, as well as the device's thermodynamic performance.

Next, a set of curated tutorials are catalogued to facilitate the use of this tool.

## Tutorials

### [1. Simple optical cross-section calculation](Thermoplasmonics.md)
> The optical cross sections of a gold nanosphere of diameter 100 nm in water are calculated. Absorption, scattering, extinction and backscattering are shown. The norm of the electric and magnetic fields, as well as the heat source density are also depicted.

### [2. Parametric sweep #1: 1D sweep](Parametric%20sweep.md)
> Parametric sweep on gold nanospheres for increasing radius (from 5 to 100 nm) is illustrated. Absorption maximum for each size is stored and used to calculate size dependent FoM. The obtained results are plotted.

### [3. Parametric sweep #2: colormaps](Colormap.md)
> Multidimensional parametric sweep is introduced exploring LSPR tunability of nanorods. Gold nanorod length and diameter are swept and absorption cross section extracted. Plasmon peak wavelength and FoM colormaps are ellaborated based length and diameter changes.

### [4. Collective heating in cuvette](Thermal.md)
> Here, a generic solution to the macroscopic transient heating of gold colloid in a cuvette suffering from convective heat losses (_h_) is shown. Three gold nanosphere sizes are modeled (5, 50 and 100 nm) maintaining the same total mass of NPs among calculations. Temperature increase in the steady-state is then compared with FoM.

### [5. FDM for DASCs](DASC.md)
> DASCs are simulated using FMD. Thermo-optical and thermophysical properties of nanofluids are modeled, considering the basefluid and NP characteristics. The solar spectral irradiance is explored and a nanofluid containing 100 nm gold nanospheres is simulated for solar heating. Furthermore, spectral absorption is discussed from Beer attenuation analysis and solar weighted absorption coefficient evaluated. Moreover, all data are plotted and DASC performance exported.

## Future additions

The library is incomplete in its current form. To further support thermoplasmonic investigations, a new set of features are planned to be introduced in future releases. Below, we highlight a non-exhaustive list of intended additions:

- Add support for spheroidal nanoshells;
- Model NP monodispersity: _include a distribution of NP sizes to mimic the spectral broadening seen in experimental data_;
- Add support for direct import of external files of nanofluid spectrum for solar simulation;
- Add [Discrete Dipole Approximation (DDA)][#10] support for NPs of arbitrary shape;
- Expand the FMD implementation to include the thermal simulation of single NP and [plasmonic metamolecules][#11]:
    1. _Steady-state heat transfer_
    2. _Transient heat transfer_
    3. _Add support for different laser pulse regimes_
 - Add support for photoacoustics.


<!--References-->

[#1]: https://en.wikipedia.org/wiki/Mie_scattering
[#2]: https://en.wikipedia.org/wiki/Gans_theory
[#3]: https://doi.org/10.3390/nano12234188
[#4]: https://doi.org/10.1038/s41598-020-63066-9
[#5]: https://en.wikipedia.org/wiki/Spheroid
[#6]: https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Gans_approximation
[#7]: http://dx.doi.org/10.1021/nl060219x
[#8]: https://en.wikipedia.org/wiki/Finite_difference_method
[#9]: https://doi.org/10.1016/j.applthermaleng.2021.116799
[#10]: https://en.wikipedia.org/wiki/Discrete_dipole_approximation
[#11]: https://doi.org/10.1038/nmat4031