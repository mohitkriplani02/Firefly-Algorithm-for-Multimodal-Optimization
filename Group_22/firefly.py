import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,cm
import random
import operator
from functools import reduce
def michalewicz(x):
    result = reduce(lambda acc, x: acc + np.sin(x) * np.power(np.sin((0 + 1) * np.power(x, 2) / np.pi), 2 * 10), x, 0.)
    return -1.0*result

def ackley(x):
    return -20*np.exp(-0.2*np.sqrt((x[0]**2+x[1]**2)/2)) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*np.pi*x[1]))) + 20 + np.exp(1)

def dejong(x):
    ans = 0
    for i in range(min(len(x),256)):
        ans+=np.power(x[i],4)
    return ans

def yang(x):
#     Yang N.2 function
    temp1 = 0
    temp2 = 0
    for i in range(min(len(x),16)):
        temp2+=np.sin(np.power(x[i],2))
        temp1+=np.absolute(x[i])
    ans =  temp1*np.exp(temp2*-1)
    return ans

def rosenbrock(x):
    ans=0.0
    for i in range(min(len(x),16)):
        ans+=(100.0*(x[i]-x[i]**2)**2 + (1-x[i])**2)
    return ans

def griewank(x):
    ans = 1.0
    inner_product = 1.0
    inner_sum = 0.0
    i=0
    while i < (len(x)):#change to while
        inner_sum += (x[i] ** 2)
        inner_product *= (np.cos(x[i] / np.sqrt(i + 1)))
        i+=1
    ans += (inner_sum * (1. / 4000.) - inner_product)
    return ans

def shubert(x):
        temp1 = 0
        temp2 = 0
        i=0
        while i < (5): 
            temp1 += ((i+1)*np.cos((i+1)+(i+2)*x[0]))
            temp2 += ((i+1)*np.cos((i+1)+(i+2)*x[1]))
            i+=1
        return temp1*temp2

def rastrigin(x):
    ans = 0.0
    i=0
    while i< (len(x)):
        ans += (x[i] ** 2 - (10. * np.cos(2 * np.pi * x[i])))
        i+=1
    ans += 10. * len(x)
    return ans

def schwefel(x):
    ans=418.982887*len(x)
    for i in range(min(len(x),128)):
        ans-=x[i]*np.sin(np.sqrt(np.abs(x[i])))
    return ans

def easom(x):
    return -1.0*np.cos(x[0])*np.cos(x[1])*np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)

class Firefly():
    def __init__(self, alpha, beta, gamma, ub, lb, function_dimension):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.brightness = None
        self.lb = lb
        self.ub = ub
        self.posn = np.array([random.uniform(self.lb, self.ub) for x in range(function_dimension)])
    
    
    @property
    def intensity(self):
        return self.brightness

        
    @property
    def position(self):
        return self.posn
    
    @position.setter
    def position(self, value):
        self.posn = value

    def check_bounds(self):
        for i, cord in np.ndenumerate(self.posn):
            if cord < self.lb:
                self.posn[i] = self.lb
            elif cord > self.ub:
                self.posn[i] = self.ub
            else:
                self.posn[i] = cord
    def update_intensity(self, func):
        self.brightness = -1.*func(self.posn)

    def rand_walk(self, area):
        self.posn = np.array([random.uniform(cord-area, cord+area) for x, cord in np.ndenumerate(self.posn)])

    def move_towards(self, better_position):
        distance = np.linalg.norm(self.posn - better_position)
        self.posn +=   self.beta*np.exp(-self.gamma*(distance**2)) * (better_position-self.posn) +  self.alpha*(random.uniform(0, 1)-0.5)
        self.check_bounds()


class FireflyProblem():
    
    def __init__(self, function, firefly_number, ub=5.12, lb=-5.12, alpha=2, beta=2, gamma=0.97, iteration_number=50, interval=500, continuous=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.function_dimension = 2
        self.ub = ub
        self.lb = lb
        self.iteration_number = iteration_number
        self.fireflies = [Firefly(self.alpha,self.beta,self.gamma,self.ub,self.lb, self.function_dimension) for x in range(firefly_number)]
        self.function = function
        self.interval = interval
        self.best = None
        self.continuous = continuous
        self.cost=[]
        i=0
        while i<(len(self.fireflies)): 
            (self.fireflies)[i].update_intensity(self.function)
            i+=1
    def run(self):
        y = np.linspace(self.lb, self.ub, 100)
        x = np.linspace(self.lb, self.ub, 100)
        X, Y = np.meshgrid(x, y)
        x_init = []
        y_init = []
        z = self.function([X, Y])        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cs = ax.contourf(X, Y, z, cmap=cm.PuBu_r) 
        fig.colorbar(cs)
        i=0
        while i<(len(self.fireflies)):
            x_init.append((self.fireflies)[i].position[0])
            y_init.append((self.fireflies)[i].position[1])
            i+=1
        particles, = ax.plot(x_init, y_init, 'ro', ms=6)
        rectangle = plt.Rectangle([self.lb, self.lb],self.ub-self.lb,self.ub-self.lb, ec='none', lw=2, fc='none')
        ax.add_patch(rectangle)
        def init():
                particles.set_data([], [])
                rectangle.set_edgecolor('none')
                return particles, rectangle

        def animate(i): #Generate animation and visualization
            x = []
            y = []
            ms = int(50.0 * fig.get_figwidth()/fig.dpi)
            rectangle.set_edgecolor('k')
            fig.canvas.set_window_title('Iteration %s/%s' % (i, self.iteration_number))
            if i ==0:
                print("reset the fireflies")
                self.best = None
            for index , fly in enumerate(self.fireflies):
                if i == 0:
                    fly.posn = np.array([x_init[index], y_init[index]])
                    fly.update_intensity(self.function)
                    fig.canvas.set_window_title('Initialization')
                x.append(fly.position[0])
                y.append(fly.position[1])
            self.step()
            particles.set_data(x, y)
            particles.set_markersize(ms)
            return particles, rectangle        
        graph = animation.FuncAnimation(fig, animate, frames=self.iteration_number+1, interval=self.interval, blit=True, init_func=init, repeat=self.continuous)
        if (not self.best or self.fireflies[0].intensity > self.best):
            self.best = self.fireflies[0].intensity
        if(self.function==michalewicz):
            graph.save('firefly_michalewicz.gif')
        elif(self.function==easom):
            graph.save('firefly_easom.gif')            
        elif(self.function==shubert):
            graph.save('firefly_shubert.gif')            
        elif(self.function==ackley):
            graph.save('firefly_ackley.gif')
        elif(self.function==rosenbrock):
            graph.save('firefly_rosenbrock.gif')
        elif(self.function==dejong):
            graph.save('firefly_dejong.gif')
        elif(self.function==griewank):
            graph.save('firefly_griewank.gif')
        elif(self.function==yang):
            graph.save('firefly_yang.gif')            
        elif(self.function==rastrigin):
            graph.save('firefly_rastrigin.gif')
        elif(self.function==schwefel):
            graph.save('firefly_schwefel.gif')
        plt.show()

    def step(self):
        (self.fireflies).sort(key=operator.attrgetter('intensity'), reverse=True)
        for x in self.fireflies:
            for y in self.fireflies:
                if y.intensity > x.intensity:
                    x.move_towards(y.position)
                    x.update_intensity(self.function)
     
        if (not self.best or (self.fireflies[0].intensity > self.best) ):
            self.best = self.fireflies[0].intensity
        if(self.function==michalewicz):
            gmin = -1.8013
            self.cost.append(abs(self.best - abs(gmin))/abs(gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin)/abs(gmin))))             
        elif(self.function==easom):
            gmin = -1
            self.cost.append(abs(self.best - abs(gmin))/abs(gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin)/abs(gmin))))             
            
        elif(self.function==shubert):
            gmin = -186.7309
            self.cost.append(abs(self.best - abs(gmin))/abs(gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin)/abs(gmin))))             
            
        elif(self.function==ackley):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        
        elif(self.function==rosenbrock):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))

        elif(self.function==dejong):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        
        elif(self.function==griewank):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        
        elif(self.function==yang):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        
        elif(self.function==rastrigin):
            gmin = 0
            self.cost.append(abs(self.best - gmin))
            print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        elif(self.function==schwefel):
                gmin = 0
                self.cost.append(abs(self.best - gmin))
                print("Overall best intensity: {}, Overall best Accuracy: {}".format(self.best,100*abs(1-abs(self.best - gmin))))
        (self.fireflies)[0].rand_walk(0.1)
        (self.fireflies)[0].update_intensity(self.function)
    
    def plot(self):
        X_Axis=range(1,self.iteration_number+2)
        plt.ylabel('Cost')
        plt.xlabel('No. of iterations')
        plt.plot(X_Axis,self.cost)
        plt.show()
def run_algorithm(optimizer,fireflies_number = 40,ub=5.12, lb=-5.12, alpha=2, beta=2, gamma=0.97, iteration_number=10, interval=500):
    out = FireflyProblem(optimizer, fireflies_number,ub, lb, alpha, beta, gamma, iteration_number, interval)
    out.run()
    out.plot()

run_algorithm(shubert,40,50,-50,2, 2,0.97, 50,100)
run_algorithm(ackley,40,20,-20,1,5,0.5,200,50)
run_algorithm(rosenbrock,40,10,-5,1,5,0.5,200,50)
run_algorithm(dejong,40,5,-5,1,5,0.5,200,50)
run_algorithm(griewank,40,3,-3,1,5,0.5,200,50)
run_algorithm(yang,40,6,-6,1,5,0.5,200,100)
run_algorithm(rastrigin,40,5.12,-5.12,1,5,0.5,150,100)
run_algorithm(easom,40,13,-7,1,5,0.5,100,100)
run_algorithm(michalewicz,40,4,0,1,5,0.5,150,100)
run_algorithm(schwefel,40,480,390,1,5,0.5,150,100)
