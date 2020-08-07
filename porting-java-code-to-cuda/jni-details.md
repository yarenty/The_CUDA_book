---
description: looking into JNI in details
---

# JNI details

Declaring a native method in Java

* to be implemented in a lower-level language
* "Native": binary is platform dependent
* Use **native** keyword in Java

```cpp
native int simpleNativeMethod(int a, double b);
```

#### Declaring a JNI header in C

Methods to be declared in a header:

```cpp
JNIEXPORT int JNICALL
Java_fully_qualified_name_of_Class_simpleMethodNativeMethod(JNIEnv *, jobject, jint, jdouble);


```

Must include "jni.h" to use JNI types and macros - this header can be found in $JAVA\_HOME/include

Header can be machine-generated



```bash
javah -jni - classpath /path/to/bytecode -force -o generatedHeader.h fully.qualified.name.of.Class
```

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

JNI arrays are directly mapped to C arrays

```text
jdouble * arrayOfJDoubles = jniEnv->GetDoubleArrayElements(aJDoubleArray,0);
jlong * arrayOfJLongs = jniEnv->GetLongArrayElements(aJLongArray,0);
jint * arrayOfJInts = jniEnv-GetIntArrayOfElemnts(aJIntArray,0);

```

However manually mapping has to be performed if one wants to take advantage of C++ bool type:

```text
bool * anArrayOfBools = new bool[n];
jboolean * arrayOfJBooleans = jniEnv->GetBooleanArrayElements(aJBooleanArray,0);
for (int k = 0; k < n; k++) 
    anArrayOfBools[k] = arrayOoJBooleana[k] == JNI_TRUE;
```

#### Unwrapping JNI objects in C/C++

1. Get class of jobject
2. Use class to get field ID using the field information \(name and type\).
3. Get object field using the object and field ID
4. Repeat these steps in the retrived field is an object itself; map the field to C/C++ otherwise

```text
jclass aJClass = jniEnv->GetObjectClass(aJObject);
jfieleID aJFieldID = jniEnv->GetFieldID(aJClass, "fieldName", "[Lfully/qualified/name/of/Class;");
if (aJFieldID == NULL) {
    //this means requested field does not exist!
}
jobject objectField = jniEnv->GetObjectField(aJObject, aJFieldID);
```



#### JNI Type Signatures

| Java Type | Type Signature |
| :--- | :--- |
| boolean | Z |
| byte | B |
| char | C |
| short | S |
| int | I |
| long | J |
| float | F |
| double | D |
| Class | Lfully/qualified/name/of/Class |
| Array of 'type' | \[type |



#### Accessing JNI object fields in C/C++

All composed objects are unwrapped to jobject - they should be explicitly casted to a proper JNI subtype, if applicable \(e.g. jbooleanArray, jintArray\)

Fields of the primitive type must be accessed using each function counterpart

GetObjectField GetIntField GetFloatField GetBooleanField GetLongField GetDoubleField

#### Performing side effects in JNI

As Java objects passed by reference to JNI, changes are reflected in the Java runtime.

Native methods can cause side effects:

* modifying an object field
  * modifying a Java object field
    1. Get class of jobject
    2. use class to get field ID using the fields information \(name and type\)
    3. set a value to the object field using the object, the field ID and the value to be set
  * modifying an int field

```cpp
jclass aJClass = jniEnv->GetObjectClass(aJObject);
jfieldID aJFieldID = jniEnv->GetFieldID(aJClass, "anIntField", "I");
if (aJFieldID == NULL) {
    //this means the requested field does not exist!
}
jobject objectField = jiniEnv->SetIntField(aJObject,aJFieldID,anyIntValue);
```

* cd
  * modifying an int array field

```cpp
jclass aJClass = jniEnv->GetObjectClass(aJObject);
jfieldID aJFieldID = jniEnv->GetFieldID(aJClass, "anIntArrayField", "[I");
if (aJFieldID == NULL) {
    //this means the requested field does not exist!
}
jintArray jniArrayField = (jintArray) jniEnv->GetObjectField(aJObject,aJFieldID);

jiniEnv->SetIntArrayRegion(jniArrayField,initialIdx, finalIdx, anArrayOfIntValues);
```

* cd
  * modyfying an int matrix field

```cpp
jclass aJClass = jniEnv->GetObjectClass(aJObject);
jfieldID aJFieldID = jniEnv->GetFieldID(aJClass, "anIntMateixField", "[[I");
jobjectArray jniMatrixField = (jobjectArray) 
    jniEnv->GetObjectField(aJObject,aJFieldID);

for (int k = 0; k < n; k++) {
    jintArray jniArrayField = (jintArray) 
        jniEnv->GetObjectArrayElement(jniMatricField, k);
    jniEnv->SetIntArrayRegion(jniArrayfield, initialIdx, finalIdx, anArrayOfIntValues); 
}

```

* invoking an object method

  1. Get class of jobject
  2. use class to get method ID using the method information \(name and type signature\)
  3. call the object method using the object, the method ID and set of arguments \(if any\)

```cpp
jclass aJClass = jniEnv->GetObjectClass(aJObject);
jmethodID aJMethodID = jniEnv->GetMEthodID(aJclass, "methodName", "(IFJ)D");
if (aJMethodID == NULL {
    //requested methond does not exist
}

jdouble result = jniEnv->CallDoubleMethod(aJObject, aJMethodID, 
            jintArg, jfloatArg, jlongArg);
```

Method signatures are in the following format: "\(contiguous-list-of-argument-types\)return-type"

```cpp
long simpleMethod(int n, String s, float[] ff);
(ILjava/lang/String;[F)J

double anotherMethod(String[] dict, int[][] map);
([Ljava/lang/String;[[I)D

```











