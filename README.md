# Synthacc
A Python interface for earthquake ground motion modelling software! It does *not* contain any simulation code.

The current version is 0.0.1, which means *totally not ready to be used by someone else*, but if you want you can try and help me improving the library! For the moment, I will only add the things I need for my PhD.

Bart Vleminckx @ Royal Observatory of Belgium (bart . vleminckx @ observatory . be)


## Installation
I use Synthacc from source on Windows 10 (64-bit) with Python 3.5 in a [Miniconda](http://conda.pydata.org/miniconda.html) virtual environment. It depends on Obspy and the OpenQuake Engine. All other dependencies, except pyproj are shared with them. They can be installed from the [conda-forge](https://conda-forge.org) channel. So no real [dependency hell](https://en.wikipedia.org/wiki/Dependency_hell) anymore thanks to conda! Only Matplotlib must additionally be specified to avoid an error with Qt (Matplotlib 2 is then installed instead of 1.5). Nose is only required for running the tests.

```
conda install -c conda-forge matplotlib pyproj obspy openquake.engine nose
```

Clone the Synthacc repository and add its folder to the PYTHONPATH environment variable. Ready!


## References
* Aki K. and Richards P.G. (2002). Quantitative Seismology (2nd edition). University Science Books.
* Akkar S., Sandikkaya M.A., Senyurt M., Sisi A.A., Ay B.Ö., Traversa P., Douglas J., Cotton F., Luzi L., Hernandez B. and Godey S. (2014). Reference database for seismic ground-motion in Europe (RESORCE). Bulletin of Earthquake Engineering, 12(1), 311-339.
* Ambraseys N., Smit P., Douglas J., Margaris B., Sigbjornsson R., Olafsson S., Suhadolc P. and Costa G. (2004). Internet-Site for European Strong-Motion Data. Bollettino di Geofisica Teorica ed Applicata, 45(3), 113-129.
* Beyreuther M., Barsch R., Krischer L., Megies T., Behr Y. and Wassermann J. (2010). ObsPy: A Python Toolbox for Seismology. Seismological Research Letters, 81(3), 530-533.
* Cotton F. and Coutant O. (1997). Dynamic stress variations due to shear faults in a plane-layered medium. Geophysical Journal International, 128, 676-688.
* Jost M.L. and Herrmann R.B. (1989). A Student's Guide to and Review of Moment Tensors. Seismological Research Letters, 60(2), 37-57.
* Kikuchi M. and Kanamori H. (1991). Inversion of complex body waves - part III. Bulletin of the Seismological Society of America, 81(6), 2335-2350.
* Krischer L., Hutko A., van Driel M., Stähler S.C., Bahavar M., Trabant C. and Nissen-Meyer T. (2017). On-demand custom broadband synthetic seismograms, Seismological Research Lettets, 88(4), 1127-1140.
* Mai, P.M. and Beroza, G.C. (2002). A spatial random field model to characterize complexity in earthquake slip. Journal of Geophysical Research, 107, B11, ESE 10-1–ESE 10-21.
* Minson S.E. and Dreger D.S. (2008). Stable inversions for complete moment tensors. Geophysical Journal International, 174, 585-592.
* Nissen-Meyer T., van Driel M., Stähler S.C., Hosseini K., Hempel S., Auer L., Colombi A. and Fournier A. (2014). AxiSEM: broadband 3-D seismic wavefields in axisymmetric media. Solid Earth, 5, 425-445.
* Pagani M., Monelli D., Weatherill G., Danciu L., Crowley H., Silva V., Henshaw, P. Butler L., Nastasi M., Panzeri L., Simionato M. and Vigano D. (2014). OpenQuake Engine: An Open Hazard (and Risk) Software for the Global Earthquake Model. Seismological Research Letters, 85(3), 692–202.
* Shearer P. (2009). Introduction to Seismology (2nd edition). Cambridge University Press.
* Stein S. and Wyession M. (2003). An Introduction to Seismology, Earthquakes, and Earth Structure. Blackwell Publishing.
* Vallée M. and Douet V. (2016). A new database of source time functions (STFs) extracted from the SCARDEC method. Physics of the Earth and Planetary Interiors, 257, 149-157.
* Vallée M., Charléty J., Ferreira A.M.G., Delouis B. and Vergoz J. (2011). SCARDEC: a new technique for the rapid determination of seismic moment magnitude, focal mechanism and source time functions for large earthquakes using body-wave deconvolution. Geophysical Journal International, 184, 338-358.
* van Driel M., Krischer L., Stähler S.C., Hosseini K. and Nissen-Meyer T. (2015). Instaseis: instant global seismograms based on a broadband waveform database. Solid Earth, 6, 701-717.
* Wang R. (1999). A Simple Orthonormalization Method for Stable and Efficient Computation of Green’s Functions. Bulletin of the Seismological Society of America, 89(3), 733–741.
