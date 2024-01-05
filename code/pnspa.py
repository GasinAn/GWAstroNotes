from sympy import *

M, F, f = symbols('M, F, f', postive=True)

t = Rational(5,256)*M*(pi*M*F)**Rational(-8,3)
Phi = Rational(1,16)*(pi*M*F)**Rational(-5,3)

dt_dF = Rational(5,96)*pi*M**2*(pi*M*f)**Rational(-11,3)

A = sqrt(2*pi)
A /= sqrt(2*pi*f*diff(t,F,2).subs(F,f)-diff(Phi,F,2).subs(F,f))
A *= M*(pi*M*f)**Rational(2,3)*Rational(1,2)*dt_dF

print(simplify(A))
print(simplify(A.subs(M,1).subs(f,1)))
