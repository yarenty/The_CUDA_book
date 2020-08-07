---
description: looking into JNI in details
---

# JNI details

Declaring a native method in Java

* to be implemented in a lower-level language
* "Native": binary is platform dependent
* Use **native** keyword in Java

`native int simpleNativeMethod(int a, double b);`

#### Declaring a JNI header in C

Methods to be declared in a header:

`JNIEXPORT int JNICALL`

`Java_fully_qualified_name_of_Class_simpleMethodNativeMethod(JNIEnv *, jobject, jint, jdouble);`

Must include "jni.h" to use JNI types and macros - this header can be found in $JAVA\_HOME/include

Header can be machine-generated

`$ javah -jni - classpath /path/to/bytecode -force -o generatedHeader.h fully.qualified.name.of.Class`

JavaH - Java Header Generator - binary can be founr iin $JDK\_HOME/bin



#### Implementing a native method in C

1. Include the created JNI header to a C source file and implement tha JNI functions \(do not forget to include "jni.h" as well\).
2. Cast JNI types to C types
3. Tun your algorithms
4. Cast C types back to JNI types for result return

#### Mapping between Java, JNI and C

Mapping between Java and JNI is performed automatically by the JVM

Mapping between types JNI and C to be ensures by type casting - however, most ot JNI types are only "renaming" of their C counterpart \(a.k.a typedef\).



Mapping Java types to JNI types

Primitive types are passed **by copy** to the native functions.

| Java primitive type | JNI type | Format |
| :--- | :--- | :--- |
| boolean | jboolean | unsigned 8 bits |
| byte | jbyte | singed 8 bits |
| char | jchar | unsigned 16 bits |
| short | jshort | signed 16 bits |
| int | jint | signed 32 bits |
| long | jlong | signed 64 bits |
| float | jfloat | 32 bits |
| double | jdouble | 64 bits |
| void | void | N/A |

Composed types are passed **by reference** to the native functions

| Java composed type | JNI type |
| :--- | :--- |
| String | jstring |
| Class | jclass |
| Object\[\] | jobjectArray |
| boolean\[\] | jbooleanArray |
| byte\[\] | jbyteArray |
| char\[\] | jcharArray |
| short\[\] | jshortArray |
| int\[\] | jintArray |
| long\[\] | jlongArray |
| floa\[\] | jfloatArray |
| double\[\] | jdoubleArray |
| Throwable | jthrowable |

Type hierarchy:

* all array types in JNI are subtype of jarray
* all object types in JNI are subtypes of jobject



Mapping JNI types to C/C++ types

| JNI type | Native type |
| :--- | :--- |
| jboolean | unsigned char |
| jbyte | signed char |
| jchar | unsigned short |
| jshort | short |
| jint | int |
| jlong | long or long long |
| jfloat | float |
| jdouble | double |
| void | void |

* types jbyte, jint, jlong are architecture-dependent, so the mapping may vary
* boolean types do not exist in C, but integer types can do the trick with the existing JNI definitions:

```text
#define JNI_FALSE 0
#define JNI_TRUE 1
```

Composed types have to be mapped manually \(no direct cast can be applied\).

The jobject must be explored, each field read and copied to an external structure.

Use of the JNIEnv object to access \(read, write, execute, map\) Java abstractions from within JNI



#### Mapping JNI Arrays to C/C++





