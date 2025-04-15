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
    '''
    Makes the needed classes to define how the particles in the population are structured.
    Fitness is single-objective minimization.
    The particle is a list which are the positions for the particle. The class also has the current speed (separated into each axis), the min/max position/speed, and the best location this particle has found.
    The limits of position/speed are the same for each dimension in this implementation.

    :return: Returns the creator object with the created classes.
    '''

    creator.create("Fitness", base.Fitness, weights=(-1.0,))
    creator.create("Particle", list, fitness=creator.Fitness, speed=list, smin=None, smax=None, best=None, pmin=None, pmax=None)
    return creator


def generateParticle(size, pmin, pmax, smin, smax):
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

    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))  # Create two positions for x, y coordinates
    part.speed = [random.uniform(smin, smax) for _ in range(size)] # Create the x, y speeds
    part.smin = smin
    part.smax = smax
    part.pmin = pmin
    part.pmax = pmax
    return part


def updateParticle(particle, best, phi1, phi2):
    '''
    Updates the particle's speed and then position with the newly calculated speed.
    The speed changes based on the best position this current particle has found and the best overall particle.

    :param particle: The particle object that is currently being updated.
    :param best: The best particle that has been found overall.
    :param phi1: The maximum weight this particle's best position should have on the updated speed
    :param phi2: The maximum weight the overall best particle's position should have on the updated speed
    '''
    
    u1 = (random.uniform(0, phi1) for _ in range(len(particle))) # Generate a random weight for this particle's best position (for each dimension)
    u2 = (random.uniform(0, phi2) for _ in range(len(particle))) # Generate a random weight for the overall best particle's position (for each dimension)
    v_u1 = map(operator.mul, u1, map(operator.sub, particle.best, particle)) # The amount of pull to this particle's best positon
    v_u2 = map(operator.mul, u2, map(operator.sub, best, particle)) # The amount of pull to the best particle's position
    particle.speed = list(map(operator.add, particle.speed, map(operator.add, v_u1, v_u2))) # The combination of the pull
    # of the particle's best known spot and the best particle's spot plus the current speed
    
    # Limits the speed to the potential min/max
    for i, speed in enumerate(particle.speed): 
        if speed < particle.smin:
            particle.speed[i] = particle.smin
        elif speed > particle.smax:
            particle.speed[i] = particle.smax
    
    # Calculates the new position of the particle
    particle[:] = list(map(operator.add, particle, particle.speed))
    
    # Limits the position of the particle to within bounds
    for i, position in  enumerate(particle): 
        if position < particle.pmin:
            particle[i] = particle.pmin
        elif position > particle.pmax:
            particle[i] = particle.pmax


def schwefel_arg0(particle):
    '''
    Calculates the error between the Schwefel function and a particle

    :param particle: The particle being examined
    :return: Returns the error between the particle and the Schwefel function
    '''

    return benchmarks.schwefel(particle)[0]


def graphsingle(pop):
    '''
    Graphs a heatmap of the Schwefel function and plots where the particles are on top of it.
    This gives a visual indicator of how accurate the particles are at the given problem.

    :param pop: The list of all the particles in the swarm
    :return: Nothing is fully returned as a plot is shown instead.
    '''
    # Creates the heatmap for the values of -510 - 510 for the X, Y values. 
    # The color is based on the error for the Schwefel function of that X, Y position (lighter is better as it is closer to 0)
    X = numpy.arange(-510, 510, 10)
    Y = numpy.arange(-510, 510, 10)
    X, Y = numpy.meshgrid(X, Y)
    Z = numpy.fromiter(map(schwefel_arg0, zip(X.flat,Y.flat)), dtype=float, count=X.shape[0]*X.shape[1]).reshape(X.shape)
    fig, (ax1) = plt.subplots(1, 1)
    fig.subplots_adjust(hspace=10)
    plot1 = ax1.contourf(X, Y, Z, cmap='Greens')
    plt.colorbar(plot1, ax=ax1)

    # Plots a scatter plot overtop of the above heatmap to show where the particles are
    # The closer they are to lighter spots on the heatmap the better. 
    xs = []
    ys = []
    for part in pop:
        xs.append(part[0])
        ys.append(part[1])
    
    ax1.scatter(xs, ys, c='Black')
    ax1.set_title("Schwefel Function")
    fig.tight_layout()
    plt.show()


def makeToolbox(smin, smax, pmin, pmax):
    '''
    Makes the toolbox which contains the functions needed to perform the particle swarm optimization.

    :param smin: The minimum speed for a particle. 
    :param smax: The maximum speed for a particle.
    :param pmin: The minimum position for a particle.
    :param pmax: The maximum position for a particle.
    '''

    toolbox = base.Toolbox()
    toolbox.register("particle", generateParticle, size=2, pmin=pmin, pmax=pmax, smin=smin, smax=smax)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluatesingle", benchmarks.schwefel)
    return toolbox


def singleObj(smin, smax, pmin, pmax, n_pop, n_gen, bestPerGen):
    '''
    Performs the single minimization objective particle swarm optimization for the Schwefel function

    :param smin: The minimum speed for a particle. 
    :param smax: The maximum speed for a particle.
    :param pmin: The minimum position for a particle.
    :param pmax: The maximum position for a particle.
    :param n_pop: Number of particles to have in the population
    :param n_gen: Number of generations to iterate (number of times particles move)
    :param bestPerGen: Bool for if the best particle should be reset each generation
    :return: Nothing is returned as the results are printed. Could add a way to save the results.
    '''

    # Sets up the needed classes/functions and the stats to track over the generations
    creator = makeCreator()
    toolbox = makeToolbox(smin, smax, pmin, pmax)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields
    
    # Makes the intitial particles and graphs where they start
    pop = toolbox.population(n=n_pop)
    graphsingle(pop)
    
    best = None
    for generation in range(n_gen):
        # If enabled, only the best current particle will have an influence on the others. Not the overall best one found 
        if bestPerGen:
            best=None
        for part in pop:
            part.fitness.values = toolbox.evaluatesingle(part)

            # Keeping track of the best position for each unique particle
            if not part.best or part.best.fitness.values > part.fitness.values:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            
            # Keeping track of the best particle found
            if not best or best.fitness.values > part.fitness.values:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        
        # Updates the positions (and speeds) of each particle
        for part in pop:
            toolbox.update(part, best)
        
        # Records various stats about the fitness for the population each generation
        logbook.record(gen=generation, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    # Re-graph the particles after they have moved around
    graphsingle(pop)