import operator
import random
import numpy
import matplotlib.pyplot as plt
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

# adapted stuff from https://deap.readthedocs.io/en/master/index.html


def makeCreatorMulti():
    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Particle", list, fitness=creator.Fitness, speed=list, smin=None, smax=None, best1=None, best2=None, pmin=None, pmax=None)
    return creator

def generateParticleMulti(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))  # Create two positions for x, y coordinates
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Create the x, y speeds
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part


def updateParticleMulti(part, best1, best2, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in range(len(part))) # The amount of pull the particle's best known
    # location has
    u2 = (random.uniform(0, phi2) for _ in range(len(part))) # The amount of pull the best particle's pull has
    u3 = (random.uniform(0, phi1) for _ in range(len(part)))
    u4 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best1, part)) # Based on the best this particle has been at
    v_u2 = map(operator.mul, u2, map(operator.sub, best1, part)) # Based on the overall best particle
    v_u3 = map(operator.mul, u3, map(operator.sub, part.best2, part))
    v_u4 = map(operator.mul, u4, map(operator.sub, best2, part))
    pull1 = map(operator.add, v_u1, v_u2)
    pull2 = map(operator.add, v_u3, v_u4)
    part.speed = list(map(operator.add, part.speed, map(operator.add, pull1, pull2)))
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


def himmelblau_arg0(sol):
    return benchmarks.himmelblau(sol)[0]


def rastrigin_arg0(sol):
    return benchmarks.rastrigin(sol)[0]


def graphmulti(pop):
    xs = []
    ys = []
    for part in pop:
        xs.append(part[0])
        ys.append(part[1])
    X = numpy.arange(-5, 6, 1)
    Y = numpy.arange(-5, 6, 1)
    X, Y = numpy.meshgrid(X, Y)
    Z1 = numpy.fromiter(map(himmelblau_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    Z2 = numpy.fromiter(map(rastrigin_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    Z3 = (Z1+Z2)/2
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.subplots_adjust(hspace=10)
    plot1 = ax1.contourf(X, Y, Z1, cmap='Greens')
    plt.colorbar(plot1, ax=ax1)
    ax1.scatter(xs, ys, c='Black')
    ax1.set_title("Function 1")
    plot2 = ax2.contourf(X, Y, Z2, cmap='Greens')
    plt.colorbar(plot2, ax=ax2)
    ax2.scatter(xs, ys, c='Black')
    ax2.set_title("Function 2")
    plot3 = ax3.contourf(X, Y, Z3, cmap='Greens')
    plt.colorbar(plot3, ax=ax3)
    ax3.scatter(xs, ys, c='Black')
    ax3.set_title("Average of the Functions")
    fig.tight_layout()
    plt.show()


def makeToolboxMulti(smin, smax, pmin, pmax):
    toolbox = base.Toolbox()
    toolbox.register("particle", generateParticleMulti, size=2, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticleMulti, phi1=2.0, phi2=2.0)
    toolbox.register("evaluatemulti", benchmarks.kursawe)
    toolbox.register("evaluatehimmelblau", benchmarks.himmelblau)
    toolbox.register("evaluaterastrigin", benchmarks.rastrigin)
    return toolbox


def multiObj(smin, smax, pmin, pmax, bestPerGen):
    creator = makeCreatorMulti()
    toolbox = makeToolboxMulti(smin, smax, pmin, pmax)
    pop = toolbox.population(n=25)
    stats1 = tools.Statistics(lambda ind: ind.fitness.values[0]) # Logbook will print out stats for the fitness for both functions
    stats1.register("avg1", numpy.mean)
    stats1.register("std1", numpy.std)
    stats1.register("min1", numpy.min)
    stats1.register("max1", numpy.max)
    stats2 = tools.Statistics(lambda ind: ind.fitness.values[1])
    stats2.register("avg2", numpy.mean)
    stats2.register("std2", numpy.std)
    stats2.register("min2", numpy.min)
    stats2.register("max2", numpy.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats1.fields + stats2.fields
    GEN = 250
    genNum = 0
    graphmulti(pop)
    best1 = None
    best2 = None
    for generation in range(GEN):
        if bestPerGen:
            best1 = None
            best2 = None
        genNum+=1
        for part in pop:
            values = toolbox.evaluatehimmelblau(part)[0], toolbox.evaluaterastrigin(part)[0]
            part.fitness.values = values[0], values[1]
            if not part.best1 or part.best1.fitness.values[0] > part.fitness.values[0]:
                part.best1 = creator.Particle(part)
                part.best1.fitness.values = part.fitness.values
            if not part.best2 or part.best2.fitness.values[1] > part.fitness.values[1]:
                part.best2 = creator.Particle(part)
                part.best2.fitness.values = part.fitness.values
            if not best1 or best1.fitness.values[0] > part.fitness.values[0]:
                best1 = creator.Particle(part)
                best1.fitness.values = part.fitness.values
            if not best2 or best2.fitness.values[1] > part.fitness.values[1]:
                best2 = creator.Particle(part)
                best2.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best1, best2)
        logbook.record(gen=generation, evals=len(pop), **stats1.compile(pop), **stats2.compile(pop))
        print(logbook.stream)
    graphmulti(pop)
