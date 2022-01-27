**Program to simulate a Triple or Double or Single pendulum with various method of numerical integration.**   
There are 3 modality to run:   
1. Simple, with just the pendulum and nothing more. Fast to run.   
2. Detailed, with the pendulum and other graphics. Slow to run.   
3. The Butterfly Effect, to visualize the chaos of the double and triple pendulum if the pendulum are given slight perturbation to the initial conditions.   
Atm the conditions that can be perturbed are initial angles, initial angular velocities, lengths, masses (only the first mass).

Triple Pendulum:
Simple example RK4: ![animate_pendulum_simple_runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151343981-362260c2-08f7-4fc1-b5ce-454d76fcdae0.gif)
Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151346162-3db437d2-b2b8-436e-9eda-5fb31c2b49a0.gif)
The Butterfly Effect example (perturbation of 1e-6 rad of angle) RK4: ![the_butterfly_effect-perturb_angles-1e-6-runge_kutta4_triple](https://user-images.githubusercontent.com/28791454/151345660-2e970dfc-baf7-490e-a4e0-a464400e9a20.gif)
TBE example (perturbation of 1e-2m of length) RK4:  
https://user-images.githubusercontent.com/28791454/151366861-8622c2bd-14c0-432a-9325-99187d90ff77.mp4

Double Pendulum:

Simple example RK4: ![animate_pendulum_simple_runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151352572-a687dc11-b931-4cf3-81bf-6491f6bce609.gif)

Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151353775-e38c26ec-9ee2-47c2-a050-f01057cd547d.gif)

The Butterfly Effect example (perturbation of 1e-6 rad of angle) RK4: ![the_butterfly_effect-perturb_angles-1e-6-runge_kutta4_double](https://user-images.githubusercontent.com/28791454/151354141-292f7a5e-0ef1-4693-a74f-46cfe2af706a.gif)



Single Pendulum:
Detailed example RK4: ![animate_pendulum_detailed_runge_kutta4_single](https://user-images.githubusercontent.com/28791454/151354608-5c673e02-3491-49a0-a6ce-2d56709c8625.gif) 
