<div id="top"></div>
<!--
*** Thanks for checking out the Isingchat project. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or send me an email to mhjguillermo@gmail.com.
-->

# Isingchat

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#organization">Organization</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project

A library to calculate the thermodynamic properties of a 1D Ising spin chain
with beyond nearest-neighbor interactions.

We explore the regular and irregular Ising chains with different range 
of interaction

### Built With

You can use the following open source package management system

* [Conda](https://docs.conda.io/en/latest/)
* [Poetry](https://python-poetry.org/)
* [Pip](https://pypi.org/project/pip/)

## Getting Started 

### Prerequisites 

It is recommended to create a separate virtual environment for
isingchat. If you use [conda][conda], it is enough to make a minimal environment
with Python 3.7 or greater, for instance, 

```shell
conda create -n isingchatdev python=3.7
```

Naturally, other virtual environment managers can be used.

### Installation

Isingchat dependencies are managed by [poetry][poetry], so poetry should be
installed in the system. In the project root directory,  execute

```shell
poetry install
```

Poetry will take care of the installation process. Afterward, isingchat
packages and command-line interface tools should be available in the current
shell. 

<p align="right">(<a href="#top">back to top</a>)</p>

## Organization
The project is organized as follows

 + src/isingchat/: Source for principal code
    + cli/ 
    + style/
    + ising.py: Main calculation code
 + conda/recipe/: recipes for conda-build construction package
 + scripts/
 + docs/
    + hardware/
        + cpuinfo: Hardware specifications used
 + test/
 + notebooks/: Folder with all notebook experiment for thermodynamic analysis
 + data/: all data calculated 
   
## Authors

### Library

Omar Abel Rodríguez López, [https://github.com/oarodriguez][gh-oarodriguez]

### Theory and main algorithm

José Guillermo Martínez Herrera

<p align="right">(<a href="#top">back to top</a>)</p>

[comment]: <> (---)

[gh-oarodriguez]: https://github.com/oarodriguez
[poetry]: https://python-poetry.org
[conda]: https://docs.conda.io/en/latest/
