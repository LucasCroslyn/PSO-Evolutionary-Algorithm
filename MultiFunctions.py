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
    '''
    Makes the needed classes to define how the particles in the population are structured.
    Fitness is multi-objective minimization.
    The particle is a list which are the positions for the particle. The class also has the current speed (separated into each axis), the min/max position/speed, and the best location this particle has found for both objectives.
    The limits of position/speed are the same for each dimension in this implementation.

    :return: Returns the creator object with the created classes.
    '''

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("ParticleMulti", list, fitness=creator.FitnessMulti, speed=list, smin=None, smax=None, best1=None, best2=None, pmin=None, pmax=None)
    return creator


def generateParticleMulti(size, pmin, pmax, smin, smax):
    '''
    This function is used to generate a new, random particle confined to the limits of what the position and speed can be.
    The limits of position/speed are the same for each dimension in this implementation.
    
    :param size: The number of dimensions for the particle (2D, 3D, etc.)
    :param pmin: The minimum position for the particle.
    :param pmax: The maximum position for the particle.
    :param smin: The minimum speed for the particle.
    :param smax: The maximum speed for the particle.
    :return: Returns the particle object after it has been made.
    '''

    part = creator.ParticleMulti(random.uniform(pmin, pmax) for _ in range(size))  # Create two positions for x, y coordinates
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Create the x, y speeds
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part


def updateParticleMulti(part, best1, best2, phi1, phi2):
    '''
    Updates the particle's speed and then position with the newly calculated speed.
    The speed changes based on the best position this current particle has found and the best overall particle for both objectives.

    :param particle: The particle object that is currently being updated.
    :param best1: The best particle that has been found overall for the first objective.
    :param best1: The best particle that has been found overall for the second objective.
    :param phi1: The maximum weight this particle's best position's should have on the updated speed
    :param phi2: The maximum weight the overall best particles's positions should have on the updated speed
    '''

    u1 = (random.uniform(0, phi1) for _ in range(len(part))) # Generate a random weight for this particle's best position (for each dimension) based on the first objective.
    u2 = (random.uniform(0, phi2) for _ in range(len(part))) # Generate a random weight for the overall best particle's position (for each dimension) based on the first objective.
    u3 = (random.uniform(0, phi1) for _ in range(len(part))) # Generate a random weight for this particle's best position (for each dimension) based on the second objective.
    u4 = (random.uniform(0, phi2) for _ in range(len(part))) # Generate a random weight for the overall best particle's position (for each dimension) based on the second objective.
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best1, part)) # The amount of pull to this particle's best positon based on the first objective.
    v_u2 = map(operator.mul, u2, map(operator.sub, best1, part)) # The amount of pull to the best particle's position based on the first objective.
    v_u3 = map(operator.mul, u3, map(operator.sub, part.best2, part)) # The amount of pull to this particle's best positon based on the second objective.
    v_u4 = map(operator.mul, u4, map(operator.sub, best2, part)) # The amount of pull to the best particle's position based on the second objective.
    pull1 = map(operator.add, v_u1, v_u2) # The total amount of pull for the first objective.
    pull2 = map(operator.add, v_u3, v_u4) # The total amount of pull for the second objective.
    part.speed = list(map(operator.add, part.speed, map(operator.add, pull1, pull2))) # The total combination of the particle's current speed plus the pulls to both objectives.
    
    # Limits the speed to the potential min/max
    for i, speed in enumerate(part.speed): 
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    
    # Calculates the new position of the particle
    part[:] = list(map(operator.add, part, part.speed))
    
    # Limits the position of the particle to within bounds
    for i, position in  enumerate(part):
        if position < part.pmin:
            part[i] = part.pmin
        elif position > part.pmax:
            part[i] = part.pmax


def himmelblau_arg0(sol):
    '''
    Calculates the error between the Himmelblau function and a particle

    :param particle: The particle being examined
    :return: Returns the error between the particle and the Himmelblau function
    '''

    return benchmarks.himmelblau(sol)[0]


def rastrigin_arg0(sol):
    '''
    Calculates the error between the Rastrigin function and a particle

    :param particle: The particle being examined
    :return: Returns the error between the particle and the Rastrigin function
    '''

    return benchmarks.rastrigin(sol)[0]


def graphmulti(pop):
    '''
    Graphs various heatmaps of the Himmelblau function, the Rastrigin function, the average of the two, and plots where the particles are on top of them.
    This gives a visual indicator of how accurate the particles are at finding the optimums between the two objectives.

    :param pop: The list of all the particles in the swarm
    :return: Nothing is fully returned as plots are shown instead.
    '''
    # The range of the X, Y values are small, just between -5 and 6 for both.
    # The color is based on the error for the function being examined for those X, Y positions (lighter is better for both as it is closer to 0)
    X = numpy.arange(-5, 6, 1)
    Y = numpy.arange(-5, 6, 1)
    X, Y = numpy.meshgrid(X, Y)
    Z1 = numpy.fromiter(map(himmelblau_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    Z2 = numpy.fromiter(map(rastrigin_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    Z3 = (Z1+Z2)/2 # Just basic avg between the two functions
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.subplots_adjust(hspace=10)
    plot1 = ax1.contourf(X, Y, Z1, cmap='Greens')
    plt.colorbar(plot1, ax=ax1)
    plot2 = ax2.contourf(X, Y, Z2, cmap='Greens')
    plt.colorbar(plot2, ax=ax2)
    plot3 = ax3.contourf(X, Y, Z3, cmap='Greens')
    plt.colorbar(plot3, ax=ax3)
    
    # Plots a scatter plot overtop of the heatmaps to show where the particles are
    # The closer to the lighter spots on the heapmaps the better.
    xs = []
    ys = []
    for part in pop:
        xs.append(part[0])
        ys.append(part[1])
    
    ax1.scatter(xs, ys, c='Black')
    ax1.set_title("Himmelblau Function")
    
    
    ax2.scatter(xs, ys, c='Black')
    ax2.set_title("Rastrigin Function")
    
    ax3.scatter(xs, ys, c='Black')
    ax3.set_title("Average of the Functions")
    
    fig.tight_layout()
    plt.show()


def makeToolboxMulti(smin, smax, pmin, pmax):
    '''
    Makes the toolbox which contains the functions needed to perform the particle swarm optimization.

    :param smin: The minimum speed for a particle. 
    :param smax: The maximum speed for a particle.
    :param pmin: The minimum position for a particle.
    :param pmax: The maximum position for a particle.
    '''

    toolbox = base.Toolbox()
    toolbox.register("particle", generateParticleMulti, size=2, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticleMulti, phi1=2.0, phi2=2.0)
    toolbox.register("evaluatehimmelblau", benchmarks.himmelblau)
    toolbox.register("evaluaterastrigin", benchmarks.rastrigin)
    return toolbox


def multiObj(smin, smax, pmin, pmax, n_pop, n_gen, bestPerGen):
    '''
    Performs the multiple minimization objectives particle swarm optimization for the Himmelblau and Rastrigin Functions

    :param smin: The minimum speed for a particle. 
    :param smax: The maximum speed for a particle.
    :param pmin: The minimum position for a particle.
    :param pmax: The maximum position for a particle.
    :param n_pop: Number of particles to have in the population
    :param n_gen: Number of generations to iterate (number of times particles move)
    :param bestPerGen: Bool for if the best particle should be reset each generation
    :return: Nothing is returned as the results are printed. Could add a way to save the results.
    '''

    # Sets up the needed classes/functions to perform the algorithm
    creator = makeCreatorMulti()
    toolbox = makeToolboxMulti(smin, smax, pmin, pmax)
    
    
    # Sets up the stats to track for both functions
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

    # Makes the initial particles and grapgs where they start
    pop = toolbox.population(n=n_pop)
    graphmulti(pop)
    
    best1 = None
    best2 = None
    for generation in range(n_gen):
        # If enabled, only the best current particle will have an influence on the others. Not the overall best one found 
        if bestPerGen:
            best1 = None
            best2 = None
        
        for part in pop:
            part.fitness.values = toolbox.evaluatehimmelblau(part)[0], toolbox.evaluaterastrigin(part)[0]
            
            # Keeping track of the best position based on the first function for each unique particle
            if not part.best1 or part.best1.fitness.values[0] > part.fitness.values[0]:
                part.best1 = creator.ParticleMulti(part)
                part.best1.fitness.values = part.fitness.values
            
            # Keeping track of the best position based on the second function for each unique particle
            if not part.best2 or part.best2.fitness.values[1] > part.fitness.values[1]:
                part.best2 = creator.ParticleMulti(part)
                part.best2.fitness.values = part.fitness.values
            
            # Keeping track of the best particle found for the first function
            if not best1 or best1.fitness.values[0] > part.fitness.values[0]:
                best1 = creator.ParticleMulti(part)
                best1.fitness.values = part.fitness.values
            
            # Keeping track of the best particle found for the first function
            if not best2 or best2.fitness.values[1] > part.fitness.values[1]:
                best2 = creator.ParticleMulti(part)
                best2.fitness.values = part.fitness.values
        
        # Updates the positions (and speeds) of each particle
        for part in pop:
            toolbox.update(part, best1, best2)
        
        # Records the stats about the fitnesses for the population each generation
        logbook.record(gen=generation, evals=len(pop), **stats1.compile(pop), **stats2.compile(pop))
        print(logbook.stream)
    
    # Re-graph the particles after they have moved around
    graphmulti(pop)
