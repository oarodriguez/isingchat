# Changelog

Versions follow [CalVer](https://calver.org).

## 2021.3.0.dev0 (Not yet released)

### Added

TODO.

### Changed

- Update `conda` recipe version number. By default, we built the `conda` package
  using the latest commit and set the build number from `git` environment
  variables.

- Calculate all transfer matrix eigenvalues for a finite chain (see routine
  `energy_finite_chain_fast` in `ising` module).

### Deprecated

TODO.

### Removed

TODO.

### Fixed

TODO.

---

## 2021.2.1 (2021-08-17)

### Fixed

- Fix `conda` recipe.

---

## 2021.2.0 (2021-08-17)

### Changed

- Update `numba`.

---

## 2021.1.0 (2021-08-17)

### Added

- Add GitHub action to execute tests.
- Identify code quality issues using `pre-commit`.
- Add tasks to install, uninstall, and upgrade the project package.

### Changed

- Switch to [CalVer](https://calver.org) to define library versions.
- Update `numpy`.
- Do not pin versions for any dependencies in `pyproject.toml` file.

### Fixed

- Fix the calculation of the norms of the eigenvalues.

---

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

---

## 0.5.0 (2021-06-03)

### Added

- We added new functionality to calculate the free energy of a finite spin
  chain, in addition to the energy of a chain in the thermodynamic limit.
