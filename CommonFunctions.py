import random
from deap import creator

def generateParticle(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))  # Create two positions for x, y coordinates
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Create the x, y speeds
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part