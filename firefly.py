# -*- coding: utf-8 -*-
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
        while i < (5): #Change to while
            temp1 += ((i+1)*np.cos((i+1)+(i+2)*x[0]))
            temp2 += ((i+1)*np.cos((i+1)+(i+2)*x[1]))
            i+=1
        return temp1*temp2

def rastrigin(x):
    ans = 0.0
    i=0
    while i< (len(x)):#Change to while
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
    def __init__(self, alpha, beta, gamma, upper_boundary, lower_boundary, function_dimension):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.__intensity = None
        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.__position = np.array([random.uniform(self.lower_boundary, self.upper_boundary) for x in range(function_dimension)])
    
    
    @property
    def intensity(self):
        return self.__intensity

        
    @property
    def position(self):
        return self.__position
    
    @position.setter
    def position(self, value):
        self.__position = value

    def move_towards(self, better_position):
        distance = np.linalg.norm(self.__position - better_position)
        self.__position = self.__position + self.beta*np.exp(-self.gamma*(distance**2)) * (better_position-self.__position) +  self.alpha*(random.uniform(0, 1)-0.5)
        self.check_boundaries()

    def random_walk(self, area):
        self.__position = np.array([random.uniform(cord-area, cord+area) for x, cord in np.ndenumerate(self.__position)])
    def update_intensity(self, func):
        self.__intensity = -1*func(self.__position)
    def check_boundaries(self):
        for i, cord in np.ndenumerate(self.__position):
            if cord < self.lower_boundary:
                self.__position[i] = self.lower_boundary
            elif cord > self.upper_boundary:
                self.__position[i] = self.upper_boundary
            else:
                self.__position[i] = cord
class FireflyProblem():
    
    def __init__(self, function, firefly_number, upper_boundary=5.12, lower_boundary=-5.12, alpha=2, beta=2, gamma=0.97, iteration_number=50, interval=500, continuous=False):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.function_dimension = 2
        self.upper_boundary = upper_boundary
        self.lower_boundary = lower_boundary
        self.iteration_number = iteration_number
        self.fireflies = [Firefly(self.alpha,self.beta,self.gamma,self.upper_boundary,self.lower_boundary, self.function_dimension) for x in range(firefly_number)]
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
        y = np.linspace(self.lower_boundary, self.upper_boundary, 100)
        x = np.linspace(self.lower_boundary, self.upper_boundary, 100)
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
        rectangle = plt.Rectangle([self.lower_boundary, self.lower_boundary],self.upper_boundary-self.lower_boundary,self.upper_boundary-self.lower_boundary, ec='none', lw=2, fc='none')
        ax.add_patch(rectangle)
        def init():
                particles.set_data([], [])
                rectangle.set_edgecolor('none')
                return particles, rectangle

        def animate(i): #Generate animation and visualization
            x = []
            y = []
            ms = int(50. * fig.get_figwidth()/fig.dpi)
            rectangle.set_edgecolor('k')
            fig.canvas.set_window_title('Iteration %s/%s' % (i, self.iteration_number))
            if i ==0:
                print("reset the fireflies")
                self.best = None

            for idx, firefly in enumerate(self.fireflies):
                if i == 0:
                    firefly.__position = np.array([x_init[idx], y_init[idx]])
                    firefly.update_intensity(self.function)
                    fig.canvas.set_window_title('Initialization')
                x.append(firefly.position[0])
                y.append(firefly.position[1])
            self.step()
            particles.set_data(x, y)
            particles.set_markersize(ms)
            return particles, rectangle        
        graph = animation.FuncAnimation(fig, animate, frames=self.iteration_number+1, interval=self.interval, blit=True, init_func=init, repeat=self.continuous)
        plt.show()
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
    
        (self.fireflies)[0].random_walk(0.1)
        (self.fireflies)[0].update_intensity(self.function)
    
    def plotter(self):
        X_Axis=range(1,self.iteration_number+2)
        plt.plot(X_Axis,self.cost)
        plt.xlabel('No. of iterations')
        plt.ylabel('Cost')
        plt.show()
def run_firefly_algorithm(optimizer,fireflies_number = 40,upper_boundary=5.12, lower_boundary=-5.12, alpha=2, beta=2, gamma=0.97, iteration_number=10, interval=500):
    out = FireflyProblem(optimizer, fireflies_number,upper_boundary, lower_boundary, alpha, beta, gamma, iteration_number, interval)
    out.run()
    out.plotter()

run_firefly_algorithm(shubert,40,50,-50,2, 2,0.97, 50,100)
run_firefly_algorithm(ackley,40,20,-20,1,5,0.5,200,50)
run_firefly_algorithm(rosenbrock,40,10,-5,1,5,0.5,200,50)
run_firefly_algorithm(dejong,40,5,-5,1,5,0.5,200,50)
run_firefly_algorithm(griewank,40,3,-3,1,5,0.5,200,50)
run_firefly_algorithm(yang,40,6,-6,1,5,0.5,200,100)
run_firefly_algorithm(rastrigin,40,5.12,-5.12,1,5,0.5,150,100)
run_firefly_algorithm(easom,40,13,-7,1,5,0.5,100,100)
run_firefly_algorithm(michalewicz,40,4,0,1,5,0.5,150,100)
run_firefly_algorithm(schwefel,40,480,390,1,5,0.5,150,100)
 