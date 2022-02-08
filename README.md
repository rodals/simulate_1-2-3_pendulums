## **Program to simulate a Triple or Double or Single pendulum with various method of numerical integration.**   
There are 3 modality to run:   
1. Simple, with just the pendulum and nothing more. Fast to run.   
2. Simple + Energy, visualizing the simulation and the variation of total energy over time.     
3. Detailed, with the pendulum and other graphics. Slow to run.   
4. The Butterfly Effect, to visualize the chaos of the double and triple pendulum if the pendulum are given slight perturbation to the initial conditions.   
Atm the conditions that can be perturbed are initial angles, initial angular velocities, lengths, masses (only the first mass) and gravity .

Some visual examples:
Triple Pendulum:
Simple example RK4: ![animate_pendulum_simple_runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151343981-362260c2-08f7-4fc1-b5ce-454d76fcdae0.gif)
Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151346162-3db437d2-b2b8-436e-9eda-5fb31c2b49a0.gif)
The Butterfly Effect example (perturbation of 1e-6 deg of angle) RK4: ![the_butterfly_effect-perturb_angles-1e-6-runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151345660-2e970dfc-baf7-490e-a4e0-a464400e9a20.gif)
TBE example (perturbation of 1e-2m of length) RK4:  
![tbe_length_40](https://user-images.githubusercontent.com/28791454/151368860-0bd1cf36-5c7e-4a28-8dc6-185bcc17cda3.gif)


Double Pendulum:

Simple example RK4: ![animate_pendulum_simple_runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151352572-a687dc11-b931-4cf3-81bf-6491f6bce609.gif)

Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151353775-e38c26ec-9ee2-47c2-a050-f01057cd547d.gif)

The Butterfly Effect example (perturbation of 1e-6 deg of angle) RK4: ![the_butterfly_effect-perturb_angles-1e-6-runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151354141-292f7a5e-0ef1-4693-a74f-46cfe2af706a.gif)



Single Pendulum:
Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_single](https://user-images.githubusercontent.com/28791454/151354608-5c673e02-3491-49a0-a6ce-2d56709c8625.gif) 


#### How to run

First install the requisites from requirements.txt: 
```
$ pip install -r requirements.txt 
```
the requisites are:
- ffmpeg for saving the animation
- numpy  for number manipulation
- matplotlib for plotting the simulation

For installing ffmpeg on your machine:
https://www.ffmpeg.org/download.html

Then the program can be executed in two ways:   
1. Execute in the directory simulate_pendulum the file simulate_pendulum.py:
    ```
    $ python3 simulate_pendulum.py
    ```
    Then the menu will be displayed on the terminal, press enter to use default (things in [*]).   
    For the output two different types of files are supported: .mp4 and .gif .   

2. Execute giving in input file(s) txt, there are some examples in the input_examples folder.
    ```
    $ python3 simulate_pendulum.py input1.txt [input2.txt, ...]
    ```
### Method of integration
The Hamiltonian of the simple pendulum is separable (which means that V(q) and T(p)) so it was possible to use explicit symplectic propagation method which conserves the energy without using Tao's splitting methods for general nonseparable Hamiltonians.
So for the simple pendulum the method Velocity Verlet and Semi Implicit euler have been implemented (but all the methods which have been used for the double and triple pendulums can be used for the simple).
Meanwhile the hamiltonian of the double and triple pendulum is not separable so two different approaches have been taken.
1. Using the Euler-Lagrange equations, inverting the system of differential equation and then using explicit non symplectic integration methods.  
2. Using the Hamiltonian of the system, then getting derivatives with respect to q_i and p_i and using implicit symplectic integration methods (Symplectic Euler and Stormer Verlet).

So for the double and triple pendulum the following methods of integrations are avaiable:
1. Forward Euler    (explicit, order 1)
2. Backward Euler   (implicit, order 1)
4. Symplectic Euler (symplectic, implicit, order 1)
5. Stormer Verlet   (symplectic, implicit, order 2)
7. Two-step Adams–Bashforth (explicit, order 2)
8. Crank Nicolson   (implicit, order 2)
9. Runge kutta 4    (explicit, order 4)

For the simple, the previous methods plus:
3. Semi-implicit euler (symplectic, explicit, order 1)
4. Velocity Verlet     (symplectic, explicit, order 2)


For reference for the Stormer Verlet and Symplectic Euler:
https://www.unige.ch/~hairer/poly_geoint/week2.pdf
