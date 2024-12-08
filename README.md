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
    <li><a href="#Matrix-Exponential-Usage">Matrix Exponential Usage</a></li>
      <ul>
        <li><a href="#Matrix-Exponential-Distributed-Computing">Matrix Exponential Distributed Computing</a></li>
        <li><a href="#Matrix-Exponential-Shared-Memory-Computing">Matrix Exponential Shared Memory Computing</a></li>
        <li><a href="#Matrix-Exponential-Output">Matrix Exponential Output</a></li>
      </ul>
    <li><a href="#Mittag-Leffler-Usage"> Mittag-Leffler Usage</a></li>
      <ul>
        <li><a href="#Mittag-Leffler-Distributed-Computing">Mittag-Leffler Distributed Computing</a></li>
        <li><a href="#Mittag-Leffler-Shared-Memory-Computing">Mittag-Leffler Shared Memory Computing</a></li>
        <li><a href="#Mittag-Leffler-Output">Mittag-Leffler Output</a></li>
      </ul>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Matrices are used to represent problems across various domains such as physics, computational chemistry, and engineering. Therefore, the efficient computation of matrix functions is generally critical whenever large-scale simulations are essential. In addressing certain problems in these fields, Krylov methods such as the generalized minimal residual method and the conjugate gradient are prevailing choices for researchers in finding an approximate solution to large-scale problems.
The Mittag-Leffler function, a special function that has widespread use in the field of fractional calculus, is noteworthy for its applicability in various physical phenomena.
Given the importance of the Mittag-Leffler function in fractional calculus, it is essential to have an efficient and accurate method to calculate its value over matrices. In this scenario, Krylov subspace methods are an effective tool for the numerical approximation of that value.
The solutions presented in this thesis address these inherent challenges by developing a parallelized and distributed Krylov Method, tailored for the Exponential and Mittag-Leffler functions. By using parallelism, the solution adopted in the current work has the potential to enhance the efficiency of matrix-vector computations noticeably. Furthermore, it aims to address scalability limitations by fully optimizing the utilization of available computing resources.
We have developed a solution using the Krylov Method to compute both the matrix exponential and the Mittag-Leffler function. The solution demonstrates fast convergence and execution time across various types of matrices. However, its scalability is limited by rising communication costs between nodes as the number of Krylov iterations grows, as well as by resource constraints.

You can find more information in https://fenix.tecnico.ulisboa.pt/cursos/meic-a/dissertacao/846778572214671
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
* Have the following installed on your system:
  * g++
  * OpenMP
  * OpenMPI
* To install the prerequisites, you can use the following command on the project directory:
  ```sh
  make requirements
  ```

<!-- USAGE EXAMPLES -->
## Matrix Exponential Usage

To compile the program, you can use the makefile provided in the exponentialMatrix folder.
```sh
cd exponentialMatrix; make
```

This will generate an executable file called "exp-distr", for distributed computing
and "exp-shared", for shared memory computing.

After compiling the program, to run it you will need to give it some arguments.

* k: the number of Krylov iterations
* n: the 2norm value of the vector you want to use to compare accuracy
* m: the matrix path of the desired input matrix. The matrix has to be in the matrix market format.

### Matrix Exponential Distributed Computing
This project is suitable to run in SLURM Workload Managers. 
However, if you want to run this project locally, you can do so by following the following steps:

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
  mpirun -np 4 ./exp-distr.out -k 5 -n 1 -m "matrix.mtx"
```

### Matrix Exponential Shared Memory Computing

To set the number of threads, you can use the OMP_NUM_THREADS variable. For example, to have 4 threads, you will need to run this command before running the project:
```sh
export OMP_NUM_THREADS=4
```

To run the shared memory computing version of the project with the following arguments:

* Five krylov iterations
* Accuracy norm value of 1
* Matrix path of the input matrix "matrix.mtx"

You can use the following command:
```sh
  ./exp-shared -k 5 -n 1 -m "matrix.mtx"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OUTPUT EXAMPLES -->
### Matrix Exponential Output
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


## Mittag Leffler Usage

To compile the program, you can use the makefile provided in the MLF folder.
```sh
cd MLF; make
```

This will generate an executable file called "mlf-distr", for distributed computing
and "mlf-shared", for shared memory computing and "mlf-shared-test" for shared memory 
computing with a test vector.

After compiling the program, to run it you will need to give it some arguments.

Regarding "mlf-distr" and "mlf-shared", the arguments are:
* k: the number of Krylov iterations
* m: the matrix path of the desired input matrix. The matrix has to be in the matrix market format.

As for the "mlf-shared-test", the arguments are:
* k: the number of Krylov iterations
* p: problem name. This argument represents the name of the matrix, and the name of the 
    vector to be used in the test. for example, if you want to use the matrix "prob.mtx",
    and the vector "prob-res.txt", you will need to use the argument "prob".

### Mittag Leffler Distributed Computing
This project is suitable to run in SLURM Workload Managers.
However, if you want to run this project locally, you can do so by following the following steps:

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
* Matrix path of the input matrix "matrix.mtx"
* 4 mpi nodes

You can use the following command:
```sh
  mpirun -np 4 ./mlf-distr.out -k 5 -m "matrix.mtx"
```


### Mittag Leffler Shared Memory Computing

To set the number of threads, you can use the OMP_NUM_THREADS variable. For example, to have 4 threads, you will need to run this command before running the project:
```sh
export OMP_NUM_THREADS=4
```

To run the shared memory computing version of the project with the following arguments:

* Five krylov iterations
* Matrix path of the input matrix "matrix.mtx"

You can use the following command:
```sh
  ./mlf-shared -k 5 -m "matrix.mtx"
```

### Mittag Leffler Shared Memory Computing with Test Vector

To set the number of threads, you can use the OMP_NUM_THREADS variable. For example, to have 4 threads, you will need to run this command before running the project:
```sh
export OMP_NUM_THREADS=4
```

To run the shared memory computing version of the project with the following arguments:
* Five krylov iterations
* Problem name "prob"

You can use the following command:
```sh
  ./mlf-shared-test -k 5 -p "prob"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- OUTPUT EXAMPLES -->
### Mittag Leffler Output

As for the ouput, with "mlf-distr.out" and "mlf-shared.out" you will get something like this:
```
exec_time_arnoldi: 0.000635
exec_time_schur: 0.003443
exec_time: 0.004323
result norm: 7.56616e+58
```

* exec_time_arnoldi: time it took to execute the Arnoldi process step
* exec_time_schur: time it took to execute the Schur-Parlett matrix function evaluation step
* exec_time: execution time of the whole program
* result norm: 2norm of the obtained vector

As for the "mlf-shared-test", you will get something like this:
```
Relative error: 1e-06
```

* Relative error: relative error between the obtained vector and the expected vector.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[cplusplus]: https://img.shields.io/badge/-c++-black?logo=c%2B%2B&style=social
[cplusplus-url]: https://cplusplus.com/
