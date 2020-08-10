# Using JNI

### Compiling and running

#### Compiling to dynamic library

Compiling mysourcefile.c to shared object libmylibrary.so whose headers should be found in directories ".,$JDK\_HOME/include" and "$JDK\_HOME/include/linux"

```bash
$ gcc -shared -I. -I$JDK_HOME/include -I$JDK_HOME/include/linux -o libmylibrary.so mysourcefile.c
```



#### Running from Java

Add generated library to the linker's dynamic libraries lookup path:

1. To a bash environment variable \(valid only for that shell session\):

```bash
$ export LD_LIBRARY_PAHT=$LD_LIBRARY_PATH:/path.to/dir/that/contains/mylib/
```

2. As a parameter to the JVM call

```bash
$ java -Djava.library.path=/path/to/dir/that/contains/mylib -cp /path/to/my/bytecodes/package.tree.to.MainClass params
```



#### Loading the library in Java

Must be loaded prior to the method invocation \(at any point in code\).

Should be loaded just once.

Is generally loaded in static block by the same class that has the native methods declarations \( so to ensure it is loaded when the class is initialised and only once, before any method call\).

```java
class MyClass {
    static {
        System.loadlibrary("mylibrary");
    }
    
    native double myMethod(int i, float f);
}    
```



#### Library naming standards

| OS | Naming standard |
| :--- | :--- |
| Windows | mylibrary.dll |
| Unix | libmylibrary.so |
| OS X | libmylibrary.dylib or libmylibrary.jnilib |



#### Choosing between different libraries

At launch, in bash: same lib name, different dirs:

```bash
$ if [ $selectThisLib ]; then MYLIB=/path1/; else MYLIB=/path2/; fi
$ java -Djava.library.path-$MYLIB MainClass
```

At runtime, in Java: different lib names

```java
String libraryName = (selectThisLib ? "firstlib" : "secondlib");
System.loadlibrary(libraryName);
```





{% hint style="info" %}
1. Declare native method in Java
2. Implement native method in low-level language
3. Compile low-level code to create a dynamic library
4. Add the low-level library to JVM library path
5. During runtime, load dynamic library and invoke the method
{% endhint %}







