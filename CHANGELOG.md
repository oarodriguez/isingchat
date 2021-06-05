# Changelog

## 0.6.0 (2021-06-04)

### Added

- The configuration file now accepts the parameter `num_tm_eigvals` to 
  indicate how many transfer-matrix-eigenvalues to use for estimating the 
  free energy. This may be useful for long chains, where it is impossible 
  or terribly slow to calculate all the eigenvalues of the transfer matrix.

### Changed

- We moved several utility functions from the `ising` module to the `utils` 
  module. Also, we fixed the tests for the `utils` module.

### Fixed

- We corrected the contribution of all eigenvalues to the partition function.

## 0.5.0 (2021-06-03)

### Added

- We added new functionality to calculate the free energy of a finite spin 
  chain, in addition to the energy of a chain in the thermodynamic limit.
  
