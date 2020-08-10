# Build

#### Build automation

Scripting task for software development:

* compiling code
* packaging runnable
* running tests
* deploying executables on production systems
* generating documentation

In order to provide the following advantages

* accelerating common task execution
* eliminating redundant tasks
* minimising the chances of slips or mistakes
* documenting the build process
* organising a history of builds and releases

{% page-ref page="make.md" %}

{% page-ref page="jar.md" %}

#### JARs and native libraries

Native libraries do NOT go inside JARs \(by default\):

* they are kept outside and pointed to for execution, just lke JARs are:

```text
$ java - jar myapp.jar -Djava.library.path=/dir/that/holds/lib/
```





