<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">

<h1 align="center">Parallelizing the Krylov Method to Calculate the Mittag-Leffler Function</h1>

</div>



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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#output">Output</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Matrices are used to represent problems across various domains such as physics, computational chemistry, and engineering. Therefore, the efficient computation of matrix functions is generally critical when large-scale simulations are essential. When dealing with certain problems in these fields, Krylov methods such as the generalized minimal residual method and the conjugate gradient are prevailing choices for researchers when finding an approximate solution to large-scale problems.
The Mittag-Leffler function, a special function that has prevalent use in the field of fractional calculus, is noteworthy for its applicability in various physical phenomena.
Given the importance of the Mittag-Leffler function in fractional calculus, it is essential to have an efficient and accurate method to calculate its value over matrices. In this scenario, Krylov subspace methods are an effective tool for the numerical approximation of that value.
The solution presented in this project addresses these inherent challenges by developing a parallelized and distributed Krylov Method, tailored for the Mittag-Leffler function. By using parallelism, the solution adopted in the current work has the potential to noticeably enhance the efficiency of matrix-vector computations. Furthermore, it aims to address scalability limitations by fully optimizing the utilization of available computing resources.

An initial solution has already been developed, which employs the Krylov Method to compute a matrix's exponential.
<p align="right"></p>



### Built With

* [C++](https://cplusplus.com/)
* [OpenMP](https://www.openmp.org/)
* [OpenMPI](https://www.open-mpi.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* Clone the project into your computer
* g++
  * If your system does not have the g++ compiler, you will need to install it.
  * If you are using Ubuntu, you can install it with:
    ```sh
    sudo apt install g++
    ```
* Eigen Library
  * Download the Eigen Library's 3.4.0 [release](https://gitlab.com/libeigen/eigen/-/releases/3.4.0).
  * Place the eigen folder on the project folder.
  * Your folder should look something like this:
    <HTML> <br> &#8618; conjugateGradient </HTML>
    <HTML> <br> &#8618; eigen </HTML>
    <HTML> <br> &#8618; exponentialMatrix </HTML>
    <HTML> <br> &#8618; utils </HTML>
* OpenMP
  * You will need to install OpenMP on your system.
  * Quick install for Ubuntu:
    ```sh
    sudo apt install libomp-dev
    ```
* OpenMPI
  * OpenMPI Quick Start [here](https://docs.open-mpi.org/en/v5.0.x/installing-open-mpi/quickstart.html)
  * Quick install for Ubuntu:
    ```sh
    sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server libopenmpi1.3 libopenmpi-dbg libopenmpi-dev
    ```

### Compilation

If you have followed the previous steps, to compile the project you need to access the desired folder and use the makefile. For example:

```sh
cd exponentialMatrix
make
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

After compiling the program, to run it you will need to give it some arguments.

* k: the number of Krylov iterations
* n: the 2norm value of the vector you want to use to compare accuracy
* m: the matrix path of the desired input matrix. The matrix has to be in the matrix market format.

This project is suitable to run in SLURM Workload Managers. However, if you want to run this project locally, you can do so by following the following steps:

To set the number of threads, you can use the OMP_NUM_THREADS variable. For example, to have 4 threads, you will need to run this command before running the project:
```sh
export OMP_NUM_THREADS=4
```
As for the number of mpi computational nodes, you can just set it up when running the project with the -np option when using mpirun to run the project. For example:
```sh
  mpirun -np 5 (...)
```

As such, to run the project locally for:
* Five krylov iterations
* Accuracy norm value of 1
* Matrix path of the input matrix "matrix.mtx"
* 4 mpi nodes
  
You can use the following command:
```sh
  mpirun -np 4 ./exp -k 5 -n 1 -m "matrix.mtx"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OUTPUT EXAMPLES -->
## Output
As for the ouput, you will get something like this:
```
exec_time_arnoldi: 0.0033148
exec_time_pade: 0.00163207
diff: 0.000000293897008
2Norm: 0.999753510484612
exec_time: 0.00901218
```

* exec_time_arnoldi: time it took to execute the Arnoldi process step
* exec_time_pade: time it took to execute the Padé Approximation step
* diff: difference between the provided 2-norm value and the 2-norm value of the obtained vector
* 2Norm: 2norm of the obtained vector
* exec_time: execution time of the whole program 


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[cplusplus]: https://img.shields.io/badge/-c++-black?logo=c%2B%2B&style=social
[cplusplus-url]: https://cplusplus.com/
