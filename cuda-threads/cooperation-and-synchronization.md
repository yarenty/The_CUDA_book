# Cooperation and synchronization

#### Thread cooperation

{% hint style="info" %}
90% of bugs, are due to:
{% endhint %}

Threads within a block can synchronise

A thread cannot synchronise with threads in other blocks

![threads cooperation](../.gitbook/assets/thread_cooperation.jpeg)



#### Thread synchronisation

Within a block you can:

* exchange data vie shared memory \(or other\)
* synchronise threads: \_\_syncthreads\(\)

![threads synchronisation](../.gitbook/assets/thread_eynchronisation.jpeg)



