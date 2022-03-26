import operator
import random
import numpy
import matplotlib.pyplot as plt
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


# adapted stuff from https://deap.readthedocs.io/en/master/index.html

def makeCreator():
    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.Fitness, speed=list, smin=None, smax=None, best=None, pmin=None, pmax=None)
    return creator

def generateParticle(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))  # Create two positions for x, y coordinates
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Create the x, y speeds
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part


def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part))) # The amount of pull the particle's best known
    # location has
    u2 = (random.uniform(0, phi2) for _ in range(len(part))) # The amount of pull the best particle's pull has
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part)) # Based on the best this particle has been at
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part)) # Based on the overall best particle
    part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2))) # The combination of the pull
    # of the particle's best known spot and the best particle's spot plus the current speed
    for i, speed in enumerate(part.speed): # Limits the speed to the potential min/max
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add, part, part.speed))
    for i, position in  enumerate(part): # Limits the position of the particle to within bounds
        if position < part.pmin:
            part[i] = part.pmin
        elif position > part.pmax:
            part[i] = part.pmax


def schwefel_arg0(sol):
    return benchmarks.schwefel(sol)[0]


def graphsingle(pop):
    xs = []
    ys = []
    for part in pop:
        xs.append(part[0])
        ys.append(part[1])
    X = numpy.arange(-510, 510, 10)
    Y = numpy.arange(-510, 510, 10)
    X, Y = numpy.meshgrid(X, Y)
    Z = numpy.fromiter(map(schwefel_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    fig, (ax1) = plt.subplots(1, 1)
    fig.subplots_adjust(hspace=10)
    plot1 = ax1.contourf(X, Y, Z, cmap='Greens')
    plt.colorbar(plot1, ax=ax1)
    ax1.scatter(xs, ys, c='Black')
    ax1.set_title("Schwefel Function")
    fig.tight_layout()
    plt.show()


def makeToolbox(smin, smax, pmin, pmax):
    toolbox = base.Toolbox()
    toolbox.register("particle", generateParticle, size=2, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluatesingle", benchmarks.schwefel)
    return toolbox


def singleObj(smin, smax, pmin, pmax, bestPerGen):
    creator = makeCreator()
    toolbox = makeToolbox(smin, smax, pmin, pmax)
    pop = toolbox.population(n=25)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
    GEN = 250
    genNum = 0
    graphsingle(pop)
    best = None
    for generation in range(GEN):
        if bestPerGen:
            best=None
        genNum+=1
        for part in pop:
            part.fitness.values = toolbox.evaluatesingle(part) # The fitness is the average of the two functions
            if not part.best or part.best.fitness.values > part.fitness.values:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness.values > part.fitness.values:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        logbook.record(gen=generation, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    graphsingle(pop)


# The algorithm function takes 5 parameters
# 1;minspeed 2;maxspeed 3;minposition 4;maxposition 5;Reset Best each generation
singleObj(-1, 1, -500, 500, False)


# Uncomment the two lines below to run the multi-function algorithm; takes same parameters as above,
# should still work if the single was ran before, might give a warning though
from Multi import *
multiObj(-0.5, 0.5, -5, 5, False)


