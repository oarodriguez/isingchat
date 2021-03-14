# Isingchat

A library to calculate the thermodynamic properties of a 1D Ising spin chain
with beyond nearest-neighbor interactions.

## Installation

Isingchat dependencies are managed by [poetry][poetry], so poetry should be
installed in the system. In the project root directory,  execute

```shell
poetry install
```

Poetry will take care of the installation process. Afterward, isingchat
packages and command-line interface tools should be available in the current
shell. It is recommended to create a separate virtual environment for
isingchat. If you use [conda][conda], it is enough to make a minimal environment
with Python 3.7 or greater, for instance, 

```shell
conda create -n isingchatdev python=3.7
```

Naturally, other virtual environment managers can be used.

## Authors

### Library

Omar Abel Rodríguez López, [https://github.com/oarodriguez][gh-oarodriguez]

### Theory and main algorithm

José Guillermo Martínez Herrera


[comment]: <> (---)

[gh-oarodriguez]: https://github.com/oarodriguez
[poetry]: https://python-poetry.org
[conda]: https://docs.conda.io/en/latest/
