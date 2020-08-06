# Moving from Java to CUDA

If you move from Java to CUDA:

* you change the algorithms
* you change the compilers
* you change the execution environment
* you change the hardware architecture

Preparing the Environment: the Java code.

We want to implement our code 

* on another environment
* using another compiler 

BUT which pieces of code?



Selecting the code

We want to keep part of the code in Java - for interfacing reasons

We want to speed it up in CUDA:

* suited to computationally intensive algorithms
* not suited for I/O- bound algorithms

Isolating the code

* semantic isolation:
  * reduce code dependencies
  * interfacing between classes
* structural isolation:

  * move classes to other packages
  * move constants/fields/methods to other packages

Common code - some pieces of code may have to be kept in both Java and CUDA:

* place them in common package
* make copies of the code
* and dont loose track of them ;-\)



Interfacing

Interfacing is important:

* make access to the code independent of implementation \(Strategy design pattern :-\( \)
* implementations can be selected on-the-fly
* results from different implementations can be compared \(method contracts are the same\)





