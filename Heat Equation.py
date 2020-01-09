"""Heat_Equation.py--Solves the heat equationby the Crank_Nicolson numerical method
                     for a square sheet with intial temperature conditions on the
                     boundaries. Also validates the solution for heat equation
                     on 2D square, by comparison to the anaylitical solution for the 2d unit
                     square with temperature bounday conditions of zero and inital temperature
                     funciton f(x,y) using fourier analysis.
                     solution.

   Language: Python 33
   Erich Wanzek
   University of Notre Dame
   Written for Computational Methods in Physics, Spring 2016.
   Last modified April 6, 2016.
"""
####################################################################################################
####################################################################################################
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg
import pylab
import scipy.integrate
####################################################################################################
def initial(temp_1,temp_2,temp_3,temp_4,steps):
    """This function creates a initial column vector called T_initial which consits of
       the initialtemperature values along the perimeter of the 2D square grid
    Arguments:
       temp1(float):
       temp2(float):
       temp3(float):
       temp4(float):
       steps(integer):
    Returns:
       T_initisl(numpy array): column vector with initial temperatures along perimeter
    """
    T_initial1 = np.zeros((steps,steps))
    for i in range(steps):
        T_initial1[i,0] = temp_1
        T_initial1[i,steps-1] =temp_2
    for j in range(steps):
        T_initial1[1,j] = temp_3
        T_initial1[steps-1,j] =temp_4

    
    T_initial2 = T_initial1[::-1]
    
    T_initial = np.reshape(T_initial2,(steps**2,1))

    return T_initial
####################################################################################################
def initial_function(steps):
    """This funciton creates an initial funciton for the grid for the validation test case, this
       initial function is f(x+y)=x+y
    Arguments:
       steps(integer): number of grid steps
    Returns:
       T_initial(numpy array): column vector with initial values
    """
    T_initial1 = np.zeros((steps,steps))
    x_val = np.linspace(0,1,steps)
    y_val = np.linspace(0,1,steps)
    for i in range(steps):
        for j in range(steps):
            T_initial1[i,j]= x_val[i]+y_val[j]

    T_initial2 = T_initial1[::-1]
    
    T_initial = np.reshape(T_initial2,(steps**2,1))

    return T_initial
####################################################################################################
def configure_matrix(steps, timestep, dimensions, a):
    """This function creates matrices A and B in the linear system to solved of the form
       Au(t+h)=Bu(t)
    Arguments:
       step(integer): number of x-y steps
       timestep(float): change in time, time_step value
       dimensions(float): size of square sheet
       a(float): Thermal diffusivity constant
    Returns:
       A,B (numpy array): martices A and B
    """
    # set up constants
    dx=dimensions/steps
    cfl = (a*timestep)/(dx**2)
    u1 = 1 + 2 * cfl
    u2 = -cfl/2
    u3 = 1 - 2 * cfl
    u4 = cfl/2
    
    A = np.zeros((steps**2,steps**2))
    B = np.zeros((steps**2,steps**2))
   
    #set up matrix elements
    
    a1_block=  u1*np.identity(steps)
    for i in range (steps):
        for j in range (steps):
            if j-i == 1:
                a1_block[i,j]= u2
            if i-j == 1:
                a1_block[i,j]= u2

    a2_block = u2*np.identity(steps)
    
    b1_block = u3*np.identity(steps)
    for i in range (steps):
        for j in range (steps):
            if j-i == 1:
                b1_block[i,j]= u4
            if i-j == 1:
                b1_block[i,j]= u4
    
    b2_block = u4*np.identity(steps)
        
    #Broadcast elemnts into main penta-diaganol banded matrices A and B
    for i in range(steps**2):
        for j in range(steps**2):
            if i == j and i%steps == 0:
                A[i:i+a1_block.shape[0], j:j+a1_block.shape[1]] = a1_block
                B[i:i+b1_block.shape[0], j:j+b1_block.shape[1]] = b1_block
    for i in range(steps**2):
        for j in range(steps**2):
             if j%steps==0 and j-i == steps :
                A[i:i+a2_block.shape[0], j:j+a2_block.shape[1]] = a2_block
                B[i:i+b2_block.shape[0], j:j+b2_block.shape[1]] = b2_block
             if i%steps==0 and i-j == steps :
                A[i:i+a2_block.shape[0], j:j+a2_block.shape[1]] = a2_block
                B[i:i+b2_block.shape[0], j:j+b2_block.shape[1]] = b2_block
    return A, B
####################################################################################################
def solve(matrices,T_initial,t_steps,steps):
    """This function solves the liner sytem for the crank-nicolson shceme for each time step
       and then compiles the grid temperature solution for each time step into a storage
       array called solution
    Arguments:
       matrices(tuple): matrices A and B
       T_initial(numpy_array): Column vector of inital temperatures at every grid point
       t_steps(integer): number of time steps
       steps(integer): number of grid steps
    Returns:
       solution(numpy_array): a matrix with column vectors consiting of the grid solution
                              for each time step
    """
    A,B = matrices   #unpack matrices
    solution = np.zeros((steps**2,t_steps))
    T = T_initial
    for k in range(t_steps):
        v=np.dot(B,T)               #matrixmultiply B by T to get column vector v
        T = numpy.linalg.solve(A,v) #solve matrix equation using 
        T=T
        for i in range(steps**2):   #store grid temp values for each time step
            solution[i,k]=T[i]
    return solution

####################################################################################################
def validation(steps,interval,t_steps,diffusivity):
    """Set up fourier anayliss solution to heat equation to serve as a validation
    Arguments:
       steps(integer): number of grid steps
       interval(tuple): time interval
       t_steps(integer): number of time steps
       diffusivity(float): Thermal diffusivity constant
    Return:
       solution(numpy array): a matrix with column vectors consiting of the grid solution
                              for each time step.
    """
    (a,b) = interval
    k=diffusivity
    time=np.linspace(a,b,t_steps)
    x_val = np.linspace(0,1,steps)
    y_val = np.linspace(0,1,steps)
    solution = np.zeros((steps**2,t_steps))
    xy_grid  = np.zeros((steps,steps))
    for t in time:
        for x in range(steps):
            for y in range(steps):
                u=0
                for m in range(1,5): # approx fourier solution to 10
                    for n in range(1,5):
                        bmn=4*scipy.integrate.dblquad(lambda x,y: (x_val[x]+y_val[y])*math.sin(m*math.pi*x_val[x])*math.sin(n*math.pi*y_val[y]),0,1,lambda x:0,lambda x:1)
                        bmn2 = bmn[0]
                        u=bmn2*math.exp(-k*(m**2+n**2)*((math.pi)**2)*t)*math.sin(m*(math.pi)*x_val[x])*math.sin(n*(math.pi)*y_val[y])
                        u+=u
                        
                xy_grid[x,y]=u

        xy_grid_clm = np.reshape(xy_grid,(steps**2,1)) 
        for i in range(steps**2):
            solution[i,t]=xy_grid_clm[i]
    return solution
####################################################################################################
def run(temp1,temp2,temp3,temp4,dimensions,steps,timesteps, time_interval,diffusivity):
    """This function graphs the solution to the heat equation of a square box of dimensions for
       the initial conditions on the perimeter given by temp1,2,3,4.Calls all main funcitons in program
    Arguments:
       temp1(float): temp on left side of square
       temp2(float): temp on right side of square
       temp3(float): temp on top of square
       temp4(flaot): temp on bottom of square
       dimensions(float): x-y dimesnion value of the square
       steps(integer): number of grid steps
       timesteps(integer):number of time steps
       time_interval(tuple): interval of time simulation starting from time of initial condition
       diffusivity(float): Thermal diffusivity of material
    Returns:
       None
    """    
    (a,b) = time_interval
    times = np.linspace(a,b,timesteps)
    timestep =(b-a)/timesteps
    T_initial = initial(temp1,temp2,temp3,temp4,steps)
    matrices  = configure_matrix(steps,timestep,dimensions,diffusivity)
    solution = solve(matrices,T_initial,timesteps,50)

    for i in range(timesteps):
           
        data =solution[:,i]   
        data=np.reshape(data,(steps,steps))
        
        pylab.imshow(data)
        plt.hot()
        cbar=plt.colorbar()
        cbar.set_label('Temperature')
        title = 'Time:' + str(times[i]) + 's'
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        pylab.show()
####################################################################################################
def run_valid(dimensions,steps,timesteps, time_interval,diffusivity):
    """This function serves as a validation of the Crank-Nicolson numerical method implemented in
       this program. This function produces both the numerical results and the analytical results
       for the solution of the heat equation of a 2Dimensional unit square with the initial tempereture
       given by the funciton T(x,y)=x+y. The analytical solution used here is the fourier analysis
       solution for the heat equation in the unit square.
    Arguments:
       dimensions(float): dimension of square(1 for this funciton)
       steps(integer): number of grid steps
       timesteps(integer): number of time steps
       time_interval(tuple): interval of time starting from time of initial condition
       diffusivity(float): Thermal diffusivity
    Returns:
       None
    """
    
    (a,b) = time_interval
    times = np.linspace(a,b,timesteps)
    timestep =(b-a)/timesteps
    T_initial = initial_function(steps)
    matrices  = configure_matrix(steps,timestep,dimensions,diffusivity)
    solution  = solve(matrices,T_initial,timesteps,50)
    solution_valid = validation(steps,time_interval,timesteps,diffusivity)

    for i in range(timesteps):

        data =solution[:,i]   
        data=np.reshape(data,(steps,steps))
        
        pylab.imshow(data)
        plt.hot()
        cbar=plt.colorbar()
        cbar.set_label('Temperature')
        title = 'Time:' + str(times[i]) + 's'
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        pylab.show()   
       
    for i in range(timesteps):

        data =solution_valid[:,i]   
        data=np.reshape(data,(steps,steps))
        
        pylab.imshow(data)
        plt.hot()
        #cbar=plt.colorbar()
        #cbar.set_label('Temperature')
        title = 'VALID solution' +'Time:' + str(times[i]) + 's'
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        pylab.show()      
        
            
####################################################################################################
#test case k=4.35^-4, diffusivity of steel
#run(100,1,1,1,1,50,10, (0,1),4.35e-4)

#validation run
run_valid(1,50,10,(0,1),4.25e-4)
####################################################################################################
####################################################################################################





















