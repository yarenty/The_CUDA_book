# Porting

Code cleanup 

In order to port efficiently, we need to clean up the code beforehand.

Get rid of unnecessary code and clarify confusing code \(this should be done always!\).

Time consumed here radically decreases the time taken during the port



Cleaning code:

* dead code - executed but has no effect on the result
  * perform identity arithmetic operations
  * write to variables or fields that are never read
  * invoke methods that neither yield results nor cause collateral effects:
  * `z += 0;`
  * `int func(int a) {int b = 2; return a; }`
* redundant code - executed more than once
  * recompute values previously calculated
  * assign a same value to different variables / fields
  * store intermediate values only neeeded once
  * compute known values:
  * `int func(int a) { int b= 2 * a; return 2*a;}`
  * `int a = 2; int b = 2; int c = a*b*5;`
  * `int func(int a) { int b=2; return a*b; }`
* unreachable code - never executed
  * code placed after return statement
  * conditions always evaluated to false
  * unused variables, fields, methods, classes:
  * `class Clazz {private int a;}`
  * `boolean b = false; if(b) {doNothing();}`
  * `in doNothing() {return 2; int x = 0;}`

Automatic cleanup: some compilers may perform these clean-ups automatically using static analysis \(FindBugs\).



