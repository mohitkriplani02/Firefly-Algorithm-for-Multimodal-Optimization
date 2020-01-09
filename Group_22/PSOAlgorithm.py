import numpy as np
import multiprocessing as mp
import inspect
from collections import namedtuple
from functools import partial
from attr import attrib, attrs
from attr.validators import instance_of
from matplotlib import animation, cm, colors
import matplotlib.pyplot as plt
from functools import reduce

@attrs
class Designer(object):
    # Overall plot design
    figsize = attrib(type=tuple, validator=instance_of(tuple), default=(10, 8))
    title_fontsize = attrib(validator=instance_of((str, int, float)), default="large")
    text_fontsize = attrib(validator=instance_of((str, int, float)), default="medium")
    legend = attrib(validator=instance_of(str), default="Cost")
    label = attrib(validator=instance_of((str, list, tuple)),default=["x-axis", "y-axis", "z-axis"],)
    limits = attrib(validator=instance_of((list, tuple)),default=[(-1, 1), (-1, 1), (-1, 1)],)
    colormap = attrib(validator=instance_of(colors.Colormap), default=cm.viridis)
@attrs
class Animator(object):
    interval = attrib(type=int, validator=instance_of(int), default=80)
    repeat_delay = attrib(default=None)
    repeat = attrib(type=bool, validator=instance_of(bool), default=True)
@attrs
class Mesher(object):
    func = attrib()
    # For mesh creation
    delta = attrib(type=float, default=0.001)
    limits = attrib(validator=instance_of((list, tuple)), default=[(-1, 1), (-1, 1)])
    levels = attrib(type=list, default=np.arange(-2.0, 2.0, 0.070))
    # Surface transparency
    alpha = attrib(type=float, validator=instance_of(float), default=0.3)
    def compute_history_3d(self, pos_history):
        fitness = np.array(list(map(self.func, pos_history)))
        return np.dstack((pos_history, fitness))

def plot_cost_history(cost_history, ax=None, title="Cost History", designer=None, **kwargs):
    iters = len(cost_history)
    if designer is None:
        designer = Designer(legend="Cost", label=["Iterations", "Cost"])
    if ax is None:
        x, ax = plt.subplots(1, 1, figsize=designer.figsize)
    ax.plot(np.arange(iters), cost_history, "k", lw=2, label=designer.legend)
    ax.set_title(title, fontsize=designer.title_fontsize)
    ax.legend(fontsize=designer.text_fontsize)
    ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
    ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
    ax.tick_params(labelsize=designer.text_fontsize)
    return ax
def _animate(i, data, plot):
    current_pos = data[i]
    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)
def plot_contour(pos_history,canvas=None,title="Trajectory",mark=None,designer=None,mesher=None,animator=None,**kwargs):
    if designer is None:
        designer = Designer(limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"])

    if animator is None:
        animator = Animator()
    if canvas is None:
        fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
    else:
        fig, ax = canvas

    n_iters = len(pos_history)
    # Customize plot
    ax.set_title(title, fontsize=designer.title_fontsize)
    ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
    ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
    ax.set_xlim(designer.limits[0])
    ax.set_ylim(designer.limits[1])

    if mesher is not None:
        xx, yy, zz, = _mesh(mesher)
        ax.contour(xx, yy, zz, levels=mesher.levels)
    if mark is not None:
        ax.scatter(mark[0], mark[1], color="red", marker="x")
    plot = ax.scatter(x=[], y=[], c="black", alpha=0.6, **kwargs)
    # Do animation
    anim = animation.FuncAnimation(fig=fig,func=_animate,frames=range(n_iters),fargs=(pos_history, plot),interval=animator.interval,repeat=animator.repeat,repeat_delay=animator.repeat_delay,)
    return anim
def _mesh(mesher):
    """Helper function to make a mesh"""
    xlim = mesher.limits[0]
    ylim = mesher.limits[1]
    x = np.arange(xlim[0], xlim[1], mesher.delta)
    y = np.arange(ylim[0], ylim[1], mesher.delta)
    xx, yy = np.meshgrid(x, y)
    xypairs = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
    # Get z-value
    z = mesher.func(xypairs)
    zz = z.reshape(xx.shape)
    return (xx, yy, zz)
@attrs
class Swarm(object):
    position = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    velocity = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    n_particles = attrib(type=int, validator=instance_of(int))
    dimensions = attrib(type=int, validator=instance_of(int))
    options = attrib(type=dict, default={}, validator=instance_of(dict))
    pbest_pos = attrib(type=np.ndarray, validator=instance_of(np.ndarray))
    best_pos = attrib(type=np.ndarray,default=np.array([]),validator=instance_of(np.ndarray),)
    pbest_cost = attrib(type=np.ndarray,default=np.array([]),validator=instance_of(np.ndarray),)
    best_cost = attrib(type=float, default=np.inf, validator=instance_of((int, float)))
    current_cost = attrib(type=np.ndarray,default=np.array([]),validator=instance_of(np.ndarray),)

    @n_particles.default
    def n_particles_default(self):
        return self.position.shape[0]
    @dimensions.default
    def dimensions_default(self):
        return self.position.shape[1]

    @pbest_pos.default
    def pbest_pos_default(self):
        return self.position

def generate_swarm(n_particles, dimensions, bounds=None, center=1.00, init_pos=None):
    if (init_pos is not None) and (bounds is None):
        pos = init_pos
    elif (init_pos is not None) and (bounds is not None):
        if not (np.all(bounds[0] <= init_pos) and np.all(init_pos <= bounds[1])):
            raise ValueError("User-defined init_pos is out of bounds.")
        pos = init_pos
    elif (init_pos is None) and (bounds is None):
        pos = center * np.random.uniform(low=0.0, high=1.0, size=(n_particles, dimensions))
    else:
        lb, ub = bounds
        min_bounds = np.repeat(np.array(lb)[np.newaxis, :], n_particles, axis=0)
        max_bounds = np.repeat(np.array(ub)[np.newaxis, :], n_particles, axis=0)
        pos = center * np.random.uniform(low=min_bounds, high=max_bounds, size=(n_particles, dimensions))
    return pos

def generate_discrete_swarm(n_particles, dimensions, binary=False, init_pos=None):
    if (init_pos is not None) and binary:
        if not len(np.unique(init_pos)) <= 2:
            raise ValueError("User-defined init_pos is not binary!")
        pos = init_pos
    elif (init_pos is not None) and not binary:
        pos = init_pos
    elif (init_pos is None) and binary:
        pos = np.random.randint(2, size=(n_particles, dimensions))
    else:
        pos = np.random.random_sample(size=(n_particles, dimensions)).argsort(axis=1)
    return pos

def generate_velocity(n_particles, dimensions, clamp=None):
    min_velocity, max_velocity = (0, 1) if clamp is None else clamp
    velocity = (max_velocity - min_velocity) * np.random.random_sample(size=(n_particles, dimensions)) + min_velocity
    return velocity

def create_swarm(n_particles,dimensions,discrete=False,binary=False,options={},bounds=None,center=1.0,init_pos=None,clamp=None,): 
    if discrete:
        position = generate_discrete_swarm(n_particles, dimensions, binary=binary, init_pos=init_pos)
    else:
        position = generate_swarm(n_particles,dimensions,bounds=bounds,center=center,init_pos=init_pos,)
    velocity = generate_velocity(n_particles, dimensions, clamp=clamp)
    return Swarm(position, velocity, options=options)


class HandlerMixin(object):
    """ A HandlerMixing class
    This class offers some basic functionality for the Handlers.
    """
    def _merge_dicts(self, *dict_args):
        """Backward-compatible helper method to combine two dicts"""
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def _out_of_bounds(self, position, bounds):
        """Helper method to find indices of out-of-bound positions
        This method finds the indices of the particles that are out-of-bound.
        """
        lb, ub = bounds
        greater_than_bound = np.nonzero(position > ub)
        lower_than_bound = np.nonzero(position < lb)
        return (lower_than_bound, greater_than_bound)

    def _get_all_strategies(self):
        """Helper method to automatically generate a dict of strategies"""
        return {k: v for k, v in inspect.getmembers(self, predicate=inspect.isroutine) if not k.startswith(("__", "_"))}


class BoundaryHandler(HandlerMixin):
    def __init__(self, strategy):
        
        self.strategy = strategy
        self.strategies = self._get_all_strategies()
        self.memory = None

    def __call__(self, position, bounds, **kwargs):
        new_position = self.strategies[self.strategy](
            position, bounds, **kwargs
        )
        return new_position
    def periodic(self, position, bounds, **kwargs):
        lb, ub = bounds
        lower_than_bound, greater_than_bound = self._out_of_bounds(
            position, bounds
        )
        bound_d = np.tile(
            np.abs(np.array(ub) - np.array(lb)), (position.shape[0], 1)
        )
        ub = np.tile(ub, (position.shape[0], 1))
        lb = np.tile(lb, (position.shape[0], 1))
        new_pos = position
        if lower_than_bound[0].size != 0 and lower_than_bound[1].size != 0:
            new_pos[lower_than_bound] = ub[lower_than_bound] - np.mod(
                (lb[lower_than_bound] - new_pos[lower_than_bound]),
                bound_d[lower_than_bound],
            )
        if greater_than_bound[0].size != 0 and greater_than_bound[1].size != 0:
            new_pos[greater_than_bound] = lb[greater_than_bound] + np.mod((new_pos[greater_than_bound] - ub[greater_than_bound]),
                bound_d[greater_than_bound],
            )
        return new_pos
    

class VelocityHandler(HandlerMixin):
    def __init__(self, strategy):
        self.strategy = strategy
        self.strategies = self._get_all_strategies()
        self.memory = None

    def __call__(self, velocity, clamp, **kwargs):
            new_position = self.strategies[self.strategy](velocity, clamp, **kwargs)
            return new_position

    def __apply_clamp(self, velocity, clamp):
        """Helper method to apply a clamp to a velocity vector"""
        clamped_vel = velocity
        min_velocity, max_velocity = clamp
        lower_than_clamp = clamped_vel <= min_velocity
        greater_than_clamp = clamped_vel >= max_velocity
        clamped_vel = np.where(lower_than_clamp, min_velocity, clamped_vel)
        clamped_vel = np.where(greater_than_clamp, max_velocity, clamped_vel)
        return clamped_vel

    def unmodified(self, velocity, clamp=None, **kwargs):
        """Leaves the velocity unchanged"""
        if clamp is None:
            new_vel = velocity
        else:
            if clamp is not None:
                new_vel = self.__apply_clamp(velocity, clamp)
        return new_vel

class op(object):
    def compute_pbest(swarm):
        dimensions = swarm.dimensions
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(~mask_cost, swarm.pbest_cost, swarm.current_cost)
        return (new_pbest_pos, new_pbest_cost)
    
    
    def compute_velocity(swarm, clamp, vh, bounds=None):
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        cognitive = (c1* np.random.uniform(0, 1, swarm_size)*(swarm.pbest_pos - swarm.position))
        social = (c2* np.random.uniform(0, 1, swarm_size)* (swarm.best_pos - swarm.position))
        temp_velocity = (w * swarm.velocity) + cognitive + social
        updated_velocity = vh(temp_velocity, clamp, position=swarm.position, bounds=bounds)
        return updated_velocity
    
    
    def compute_position(swarm, bounds, bh):
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        position = temp_position
        return position
    
    
    def compute_objective_function(swarm, objective_func, pool=None, **kwargs):
        if pool is None:
            return objective_func(swarm.position, **kwargs)
        else:
            results = pool.map(partial(objective_func, **kwargs),np.array_split(swarm.position, pool._processes),)
            return np.concatenate(results)
    
class Star(object):
    def __init__(self, static=None, **kwargs):
        self.static = static
        self.neighbor_idx = None
        
    def compute_gbest(self, swarm, **kwargs):
        if self.neighbor_idx is None:
            self.neighbor_idx = np.tile(np.arange(swarm.n_particles), (swarm.n_particles, 1))
        if np.min(swarm.pbest_cost) < swarm.best_cost:
            best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
            best_cost = np.min(swarm.pbest_cost)
        else:
            best_pos, best_cost = swarm.best_pos, swarm.best_cost
        return (best_pos, best_cost)

    def compute_velocity(self,swarm,clamp=None,vh=VelocityHandler(strategy="unmodified"),bounds=None,):
        return op.compute_velocity(swarm, clamp, vh, bounds=bounds)

    def compute_position(self, swarm, bounds=None, bh=BoundaryHandler(strategy="periodic")):
        return op.compute_position(swarm, bounds, bh)
    
class GlobalBestPSO(object):
    """For running Schwefel and set bounds = (min_bound,max_bound)"""
#     max_bound = 480*np.ones(2)
#     min_bound = 390*np.ones(2)
    def __init__(self,n_particles,dimensions,options,bounds=None,bh_strategy="periodic",velocity_clamp=None,vh_strategy="unmodified",center=1.00,ftol=-np.inf,init_pos=None,):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.velocity_clamp = velocity_clamp
        self.swarm_size = (n_particles, dimensions)
        self.options = options
        self.center = center
        self.ftol = ftol
        self.init_pos = init_pos
        self.ToHistory = namedtuple(
            "ToHistory",
            [
                "best_cost",
                "mean_pbest_cost",
                "mean_neighbor_cost",
                "position",
                "velocity",
            ],
        )
        self.reset()
        # Initialize the topology
        self.top = Star()
        self.bh = BoundaryHandler(strategy=bh_strategy)
        self.vh = VelocityHandler(strategy=vh_strategy)
        self.name = __name__
    def _populate_history(self, hist):
        self.cost_history.append(hist.best_cost)
        self.mean_pbest_history.append(hist.mean_pbest_cost)
        self.mean_neighbor_history.append(hist.mean_neighbor_cost)
        self.pos_history.append(hist.position)
        self.velocity_history.append(hist.velocity)
    
    def reset(self):
        # Initialize history lists
        self.cost_history = []
        self.mean_pbest_history = []
        self.mean_neighbor_history = []
        self.pos_history = []
        self.velocity_history = []
        # Initialize the swarm
        self.swarm = create_swarm(n_particles=self.n_particles,dimensions=self.dimensions,bounds=self.bounds,center=self.center,init_pos=self.init_pos,clamp=self.velocity_clamp,options=self.options,)

    def optimize(self, objective_func, iters, n_processes=None, **kwargs):

        print("Optimize for {} iters with {}".format(iters, self.options))
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position
        pool = None if n_processes is None else mp.Pool(n_processes)

        self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        for i in range(iters):
            self.swarm.current_cost = op.compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = op.compute_pbest(self.swarm)
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            hist = self.ToHistory(best_cost=self.swarm.best_cost,mean_pbest_cost=np.mean(self.swarm.pbest_cost),mean_neighbor_cost=self.swarm.best_cost,position=self.swarm.position,velocity=self.swarm.velocity,)
            self._populate_history(hist)
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            if (np.abs(self.swarm.best_cost - best_cost_yet_found)< relative_measure):
                break
            self.swarm.velocity = self.top.compute_velocity(self.swarm, self.velocity_clamp, self.vh, self.bounds)
            self.swarm.position = self.top.compute_position(self.swarm, self.bounds, self.bh)
        # Obtain the final best_cost and the final best_position
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()].copy()
        # Write report in log and return final cost and position
        print("Optimization finished | best cost: {}, best pos: {}".format(final_best_cost, final_best_pos))

        return (final_best_cost, final_best_pos)

class fx(object):
    def michalewicz(y):
        x = y.T
        result = reduce(lambda acc, x: acc + np.sin(x) * np.power(np.sin((0 + 1) * np.power(x, 2) / np.pi), 2 * 10), x, 0.)
        return -1.0*result
    
    def ackley(x):
        d = x.shape[1]
        j = (-20.0 * np.exp(-0.2 * np.sqrt((1 / d) * (x ** 2).sum(axis=1)))- np.exp((1 / float(d)) * np.cos(2 * np.pi * x).sum(axis=1))+ 20.0+ np.exp(1))
        return j
 
    def easom(x):
        x_ = x[:, 0]
        y_ = x[:, 1]
        j = (-1* np.cos(x_)* np.cos(y_)* np.exp(-1 * ((x_ - np.pi) ** 2 + (y_ - np.pi) ** 2)))
        return j
    
    def rastrigin(x):
        #x between -5.12 to 5.12
        d = x.shape[1]
        j = 10.0 * d + (x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)
        return j
    
    def rosenbrock(x):
        r = np.sum(100 * (x.T[1:] - x.T[:-1] ** 2.0) ** 2 + (1 - x.T[:-1]) ** 2.0, axis=0)
        return r
    def yang(y):
        temp1 = 0
        temp2 = 0
        x = y.T
        for i in range(min(len(x),16)):
            temp2+=np.sin(np.power(x[i],2))
            temp1+=np.absolute(x[i])
        ans =  temp1*np.exp(-1*temp2)
        return ans
    def griewank(y):
        x = y.T
        ans = 1.0
        inner_product = 1.0
        inner_sum = 0.0
        for i in range(len(x)):
            inner_sum += x[i] ** 2
            inner_product *= np.cos(x[i] / np.sqrt(i + 1))
        ans += inner_sum * (1.0 / 4000.0) - inner_product
        return ans
    def schwefel(y):
        x=y.T
        alpha=418.982887
        fitness=alpha*len(x)
        for i in range(min(len(x),128)):
            fitness-=x[i]*np.sin(np.sqrt(np.abs(x[i])))
        return fitness

    def shubert(y):
        x = y.T
        temp1 = 0
        temp2 = 0
        i=0
        while i < (5): #Change to while
            temp1 += ((i+1)*np.cos((i+1)+(i+2)*x[0]))
            temp2 += ((i+1)*np.cos((i+1)+(i+2)*x[1]))
            i+=1
        return temp1*temp2
    def dejong(y):
        x = y.T
        ans = 0
        for i in range(min(len(x),256)):
            ans+=np.power(x[i],4)
        return ans
    
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(0*np.ones(2),4*np.ones(2)))
cost, pos = optimizer.optimize(fx.michalewicz, iters=50)

plot_cost_history(cost_history=optimizer.cost_history)
gmin = -1.8013
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin)/abs(gmin))))
plt.show()
m = Mesher(func=fx.michalewicz)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2D_michalewicz.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-20*np.ones(2),20*np.ones(2)))
cost, pos = optimizer.optimize(fx.ackley, iters=50)

plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.ackley)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2D_ackley.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-7*np.ones(2),13*np.ones(2)))
cost, pos = optimizer.optimize(fx.easom, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin = -1
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin)/abs(gmin))))
plt.show()
m = Mesher(func=fx.easom)
anim = plot_contour(pos_history=optimizer.pos_history,mesher=m,mark=(0,0))
anim.save("PSO2D_easom.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-5.12*np.ones(2),5.12*np.ones(2)))
cost, pos = optimizer.optimize(fx.rastrigin, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.rastrigin)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2D_rastrigin.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-5.12*np.ones(2),5.12*np.ones(2)))
cost, pos = optimizer.optimize(fx.rosenbrock, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.rosenbrock)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2D_rosenbrock.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-6*np.ones(2),6*np.ones(2)))
cost, pos = optimizer.optimize(fx.yang, iters=50)

plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.yang)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2D_yang.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-3*np.ones(2),3*np.ones(2)))
cost, pos = optimizer.optimize(fx.griewank, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.griewank)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2Ds_griewank.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(390*np.ones(2),480*np.ones(2)))
cost, pos = optimizer.optimize(fx.schwefel, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.schwefel)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2Ds_schwefel.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options,bounds=(-5*np.ones(2),5*np.ones(2)))
cost, pos = optimizer.optimize(fx.dejong, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin=0
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin))))
plt.show()
m = Mesher(func=fx.dejong)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2Ds_dejong.gif")

options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = GlobalBestPSO(n_particles=50, dimensions=2, options=options)
cost, pos = optimizer.optimize(fx.shubert, iters=50)
plot_cost_history(cost_history=optimizer.cost_history)
gmin = -186.7309
print("Accuracy: {}".format(100*abs(1-abs(cost - gmin)/abs(gmin))))
plt.show()
m = Mesher(func=fx.shubert)
anim = plot_contour(pos_history=optimizer.pos_history,
                         mesher=m,
                         mark=(0,0))
anim.save("PSO2Ds_shubert.gif")