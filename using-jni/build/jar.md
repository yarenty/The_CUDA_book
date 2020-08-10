# JAR

#### JAR - the Java Archive

The standard file used for distribution Java applications and libraries, for easy deployment

Generally comes with important metadata and resources associated

The data are compressed with the ZIP algorithm

Can be executable if a mani class is pointed

```text
myApp.jar
    |--- META-INF
    |        |-- MANIFEST.MF
    |--- packagefoo
             |-- Class1.class
             |-- packagebar
                     |-- Class2.class
                  
```

Inside a JAR comes the package tree for a Java application or library

There may be any associated resources as well \(images, music, video\)

There can be an optional directory "META-INF", with an optional file inside, "MANIFEST.MF", used to point a main class for runnable JARs.

Simple manifest file:

```text
Manifest-Version: 1.0
Main-Class: the.package.tree.to.my.MainClass
```

JAR files are ZIP files, so can be made with most compression software

JARs are more commonly .made with the jar utility that comes with the JDK \($JDK\_HOME/bin\)

Most IDEs make up JARs, generally invoking the JDK jar utility for doing so.



#### Creating a JAR

```text
$ jar cmfv manifest.mf myapop.jar -C /dir/to/bytrcode/package/tree/ .
```

{% hint style="info" %}
* c - create 
* m - manifest
* f - filename
* v -verbose

the order between 'm' and 'f' must reflect the order these parameters are passed
{% endhint %}

#### Running JARs

JARs can be run with the jave utility , as usual:

```text
$ java -cp myapp.jar the.package.tree.to.my.MainClass
```

 More conveniently, with the jar argument \(if JAR is executable\):

```text
java -jar myapp.jar
```















