# Architecture Notes

#### Attaching JNI to your code

The existence of JNI in your code must be isolated and self-contained \(best practice: low coupling\):

* a set of Java classes dealing with JNI-related issues \(e.g. declaring the Java native methods\)
* a set of C source files dealing with JNI-related issues \(e.g. converting between JNI and C types and modifying Java object fields

#### Interfacing to the native code

Having an API header in your library is important

Reuse and low coupling: library can be easily attached:

* to JNI and then be called by Java
* to other native applications
* to a standalone piece of native code and then be called directly from C/C++



#### Performance

Avoid the overhead introduced by JNI to the communication between the JVM and native codes:

* perform a full computation on one side and then transmit the result to the other; do not perform in pieces and transmit in pieces unless necessary 
* retrieve and modify JNI arrays by range, not element by element
* look-ups for method and field IDs are very costly











