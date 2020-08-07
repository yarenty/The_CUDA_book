# General notes on JNI

Method invocations are synchronous:

* Java to C
* C to Java

The JNIEnv pointer is only valid in the current thread - do not attempt to pass the pointer to other threads.

JNI does not check against NULL parameters.

The set of arguments for native methods follow these requirements:

* the first argument is always the JNIEnv pointer
* if the method is nonstatic, the second argument is a jobject representing the invoking object "this"
* the remaining arguments come from the method aignature in Java, typed with their JNI counterpart

As C does not support overloading, Java overloaded methods are appended with two underscores followed by the argument signature \(so to avoid name clash\)

JNI resolves native methods in the following order:

1. first the short name
2. then the long name, i.e. with the argument signature

#### Overview of JNI

1. Declare methods in Java with the "native" keyword
2. Invoke JavaH to generate the corresponding C header
3. Include the header do a C source file and implement the declared functions, paying attention to cast types,  unwrap objects and modify fields.







