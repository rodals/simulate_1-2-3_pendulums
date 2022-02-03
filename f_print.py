### Print functions

def running(anim_name, pendulum_type, numerical_integration_name ):
    print(f"\n Running {anim_name} {pendulum_type} pendulum propagated with {numerical_integration_name}")

def percentage(i, frameforsec, time_simulation):
    print(f"  {int(100*i/min(frameforsec * time_simulation, int(time_simulation/h)))} % Processing  ", end="\r") 
    
def bye():
    print("Non valid input. Bye")
    exit()
