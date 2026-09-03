import math

F = input("Input force function F(x, y, vx, vy, t):")

f = lambda x, y, vx, vy, t : eval(F)

v0=float(input("initial magnitude of velocity"))

theta=float(input("initial angle with x-axis"))

x=float(input("initial x"))

y=float(input("initial y"))

vy= math.cos(math.radians(theta))*v0

vx= math.sin(math.radians(theta))*v0

g= -9.8 

dt = 0.01

while y>0:
    y = y+ vy * dt
    vy = vy + g * dt
    x = x + vx * dt

print(x,y) 

