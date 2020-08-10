# Testing

We want to test if native code we ported complies with the original Java code.

That is, if it yields the same results as the original piece of code.

#### Software testing

We don't care how the native implementation was conceived

It may be poor in terms of testability in favour of good performance and resource usage.

JNI is beautiful but debugging is very painful



#### Black box testing

Feed certain input, expect certain output.

If the input is the same in both codes, then the output must be as well



input --&gt; \[ black box \] --&gt; output



A great assumptions - determinism: same inputs always yield outputs for a same black box

* what about random number generator
* what about parallel execution flow interface
* what about time dependencies
* what about environment dependencies



#### Avoiding non-determinism



Random number generator:

* actually, pseudorandom or quasi-random: the generated sequence depends on seeds
* seed must be explicitly set to a known constant value - no environment-dependent values!
* known konstant value should be passed as a paremater to the execution:
  * it should be included in the list of input parameters
  * it should be optional, to be used only for testing purposes, so that it is not defined then its value is set to the production, non-testing value

Parallel execution flow interface:

* race condition: the result of a computation depends on the order or synchronisam of events triggered by separate execution flows
* result may not be wrong, but it is probabilisticaly defined \(randomnes\)
* caused by shared states: shared variables accessed concurrently
* solution: mutual exclusion - only one execution flow is allowed to access a shared variable at once
* pay extreme care to :
  * critical sections: pieces of code that access a shared and must not be concurrently accessed by more than one execution flow
  * atomic operations: consist of a set linear steps that either execute as an atomic sequence or not at all \(cannot be hung between two steps\)
* in Java, mutual exclusion is implemented with the kwyword eynchronized, and can be applied:
  * over methods: locks the execution of any synchronized method on the same object
  * over objects: locks the execution of a piece of code synchronized by the same object

Methods:

```java
class MyClass {
    synchronized void firstMethod() {
        ...
    }
    
    synchronized void secondMethod() {
        ...
    }
}
```

Statements:

```java
...
synchronized (object) {
    doThis();
    doSomethingElse();
    doMore();
}
```

* In Java, read and write operations are atomic for all variables qualified as volatile

```java
class MyClass() {
    volatile long variable;
    
    void addVal(int val) {
        variable += val;
    }
}
```



Time dependencies:

* conditions that depend on time are dangerous
  * performing X operations within a time frame
  * executing function X when Z second passed
* time consumed for computations is hardware-dependent: faster hardware, less time needed
* avoid time-related constraints!



Environment dependencies:

* depending on the environment is very tricky:
  * using the current time or the process ID
  * performing an action if certain environment condition is met \(e.g. library is available\)
* a solution: creating flag for testing and surrounding environement-dependent code with a flagged conditional



#### BUGS

If the library is used within Java: printf debbuging

If the library has proper JNI-agnostic API - create an executable code to run the library:

* discard chances that the bug be due to Java
* be able to use debugging tools \(e.g. GNU gdb\)

**Allows library code to be executed, tested and debugged without being enclosed in a Java shell.**









