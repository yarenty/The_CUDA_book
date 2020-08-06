# JNI - Interoperating with Java

JNI - the Java Native Interface

An application programming interface:

* allows Java code from within a JVM to invoke native code \(C, C++, assembly\)
* allows native code to invoke JVM
* implementation is JVM specific



Why we want to call native code from Java?

* Java API does not support platform-dependent features
* Want to use existing library written in native code
* need to run time-critical code ina lower-level language

How to use JNI

1. Declare native method in Java
2. Implement native method in low-level language
3. Compile low-level code to create a dynamic library
4. Add the dynamic library to the JVM library path
5. During runtime, load dynamic library and invoke method.

How JNI works

1. Java code requests library load
2. JVM searches for the named library on its library paths and loads it
3. Native method is invoked in Java
4. JVM looks for the method implementation on its loaded libraries
5. Parameters to the native method are copied and context is switched to run the native implementation outside JVM
6. Method finishes and results are reflected in the JVM
7. Execution flow continues as if the method was run by JVM

