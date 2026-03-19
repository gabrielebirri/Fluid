from phi.flow import *
import math

# Create velocity field
v = StaggeredGrid((1, 0), extrapolation.ZERO, x=64, y=64, bounds=Box(x=(0,1), y=(0,1)))

# Create an oriented box
b1 = Box(x=(0.2, 0.4), y=(0.2, 0.4)).rotated(math.pi/4)
o1 = Obstacle(b1)

# Create another box
b2 = Box(x=(0.6, 0.8), y=(0.6, 0.8)).rotated(-math.pi/6)
o2 = Obstacle(b2)

try:
    v, p = fluid.make_incompressible(v, (o1, o2))
    print("Success with multiple oriented obstacles!")
except Exception as e:
    print("Error:", e)
