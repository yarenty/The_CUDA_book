# Make



#### The make tool

It's fame: it is included in Unix since 1977

Its input: it reads a script called "Makefile"

Its operation: it is intimately bound to the execution environment shell  

Its support: environment dependency checkers, IDEs, makefile generators

Its handy feature: defaults exist in the absence of a target or the whole makefile

Its limit: makefile portability and language syntax

> Make is to C what Ant to Java \(update: Maven or Gradle\)



```c
# this is a simple makefile

CC=gcc
CFLAGS=-g -O3 #-v

OBJS=hello.o
EXE=hello

all: $(EXE)

$(EXE): $(OBJS)
    $(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
    rm -rf $(OBJS) $(EXE)
    
.SUFFIXES: .c .o

%.0: %.c
    $(CC) -c $(CFLAGS( $<

```

{% hint style="info" %}
* line comments starts with hash '\#'
* macro declaration: MACRO=line of values
* macro use: $\(MACRO\)
* targets:
  * `target: dep1 dep2`
  * traditionally the first target is called 'all'
  * target commands come in a line following target
  * command line begins with &lt;TAB&gt;!
  * command line is batch in Windows, bash in Unix
  * multiple commands come line after line, always &lt;TAB&gt; indented
  * one command line = one shell session \(how to change shell environment then?\)
  * the internal macro $@ contains the name of the current target
* traditionally there is a target 'clean' to reset the environment
* %.o: %.c  - implicit/suffix rules based on file extensions
* the internal macro $&lt; fives the "from" value on implicit rules
* the .SUFFIXES list the set of file extensions the implicit rules work on
{% endhint %}



#### Running make

Simplest form:

```c
$ make
```

* Searches for files 'makefile', then 'Makefile' in the current directory
* Executes the first target declared in the makefile

 

Specify the makefile:

```c
$ make -f myMakefile
```

Pointing the targets to be executed in order:

```c
$ make clean all
```

Give environment variables precedence:

```c
$ make -e
```

Ignore all errors in command execution:

```c
$ make -i
```

Silent mode - do  not output te command lines :

```c
$ make -s
```

