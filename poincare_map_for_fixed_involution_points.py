# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys

def duffing_oscillator(t, y, p1, p2):
    x, v = y
    
    # First derivative: dx/dt = v
    dxdt = v
    
    # Second derivative: dv/dt = -x - x³ + p1*x*v + p2*sin(3t)
    # This comes from the equation: d²x/dt² + x + x³ = p1*x*dx/dt + p2*sin(3t)
    # Rearranged: d²x/dt² = -x - x³ + p1*x*v + p2*sin(3t)
    dvdt = -x - x**3 + p1 * x * v + p2 * np.sin(3 * t)
    
    return [dxdt, dvdt]

def construct_poincare_map(p1=0.1, p2=0.5):
    """
    Construct Poincare map for forced, damped Duffing oscillator
    
    The Poincare map is constructed by:
    1. Starting from initial conditions on the Y-axis (x = 0)
    2. Integrating the system for many periods to reach steady state
    3. Recording the system state at each Poincare section (T = 2p/3)
    4. Plotting the collected points in phase space
    
    Parameters:
    -----------
    p1 : float
        Damping parameter (default: 0.1)
    p2 : float
        Forcing amplitude (default: 0.5)
    """
    
    # ============================================
    # PARAMETER SETUP
    # ============================================
    
    # Duffing equation parameters
    damping_param = p1      # p1: damping coefficient
    forcing_param = p2      # p2: forcing amplitude
    
    # Initial condition parameters
    x0 = 0.0  # Fixed initial position (Y-axis in phase space)
    v_min, v_max = -10.0, 10.0  # Velocity range
    v_step = 0.01  # Step size for initial velocities
    
    # Generate initial velocities
    v_values = np.arange(v_min, v_max + v_step, v_step)
    
    # Poincare section parameters
    forcing_frequency = 3.0  # Forcing frequency
    """
    period = 2 * np.pi / forcing_frequency  # Poincare section period (2p/3)
    num_periods = 2  # Number of periods to integrate # KC : 1st period = 0 (axis Y) , 2nd period = 2Pi/3 (image of the fixed points of involution)
    
    # Integration parameters
    t_span = (0, num_periods * period)  # Total integration time
    t_eval = np.arange(0, num_periods * period, period)  # Evaluation times at Poincare sections
    """
    
    period = 2 * np.pi / forcing_frequency  # Poincare section period (2p/3)
    num_periods = 2  # Number of periods to integrate # KC : 1st period = 0 (axis Y) , 2nd period = 2Pi/3 (image of the fixed points of involution)
    
    # Integration parameters
    t_span = (0, num_periods * period)  # Total integration time
    t_eval = np.arange(period, num_periods * period, period)  # Evaluation times at Poincare sections
    
    # ============================================
    # Poincare MAP CONSTRUCTION
    # ============================================
    
    poincare_points = []
    total_points = len(v_values)
    
    print(f"Constructing Poincare map for forced, damped Duffing oscillator")
    print(f"Equation: d^2x/dt^2 + x + x^3 = p1*x*dx/dt + p2*sin(3t)")
    print(f"Poincare section: T = 2p/3 = {period:.6f}")
    print(f"Initial conditions: x0 = {x0}, v0 = [{v_min}, {v_max}]")
    print(f"Number of initial conditions: {total_points}")
    print(f"Integration: {num_periods} periods per trajectory (for steady state)")
    print(f"\nProcessing initial conditions...")
    
    # Iterate through all initial velocities
    for i, v0 in enumerate(v_values):
        # Display progress
        progress = (i + 1) / total_points * 100
        sys.stdout.write(f"\rProgress: {progress:.1f}%")
        sys.stdout.flush()
        
        # Set initial conditions
        y0 = [x0, v0]
        
        try:
            # Integrate the differential equation
            # Using RK45 (Runge-Kutta-Fehlberg) method for adaptive step size
            sol = solve_ivp(
                duffing_oscillator,      # Differential equation
                t_span,                 # Time interval
                y0,                     # Initial conditions
                args=(p1, p2),          # Additional parameters (p1, p2)
                t_eval=t_eval,           # Evaluation times
                method='RK45',           # Integration method
                rtol=1e-8,              # Relative tolerance # KC - can be changed 1e-10
                atol=1e-10              # Absolute tolerance # KC - can be changed 1e-12
            )
            
            # Check if integration was successful
            if sol.success:
                # Extract position and velocity from solution
                x_values = sol.y[0]      # Position at each Poincare section
                v_values_result = sol.y[1]  # Velocity at each Poincare section
                
                # Store Poincare points (x, v) at each section
                # These points represent the asymptotic behavior after transients
                for j in range(len(x_values)):
                    poincare_points.append((x_values[j], v_values_result[j]))
                    
        except Exception as e:
            # Handle any integration errors
            pass
    
    print(f"\n\nCollected {len(poincare_points)} Poincare points")
    
    # ============================================
    # DATA PROCESSING
    # ============================================
    
    # Convert list of tuples to numpy array for easier manipulation
    poincare_array = np.array(poincare_points)
    x_poincare = poincare_array[:, 0]  # x-coordinates of Poincare points
    v_poincare = poincare_array[:, 1]  # v-coordinates of Poincare points
    
    # ============================================
    # VISUALIZATION
    # ============================================
    
    print("Creating visualizations...")
    
    # Figure 1: Multiple zoomed views of Poincare map
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Poincare Map for Forced, Damped Duffing Oscillator (T = 2p/3)\n"
                    f"p1 = {p1}, p2 = {p2}", fontsize=14, fontweight='bold')
    
    # Subplot 1: Full Poincare map
    ax1 = fig.add_subplot(221)
    ax1.scatter(x_poincare, v_poincare, s=0.5, alpha=0.6, c='blue')
    ax1.set_xlabel('x (position)', fontsize=12)
    ax1.set_ylabel('v = dx/dt (velocity)', fontsize=12)
    ax1.set_title('Full Poincare Map', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Initial x = 0')
    ax1.legend(loc='upper right')
    
    # Subplot 2: Zoomed view of central region
    ax2 = fig.add_subplot(222)
    ax2.scatter(x_poincare, v_poincare, s=0.5, alpha=0.6, c='blue')
    ax2.set_xlabel('x (position)', fontsize=12)
    ax2.set_ylabel('v = dx/dt (velocity)', fontsize=12)
    ax2.set_title('Central Region (Zoomed)', fontsize=12)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Subplot 3: Medium zoom view
    ax3 = fig.add_subplot(223)
    ax3.scatter(x_poincare, v_poincare, s=0.5, alpha=0.6, c='blue')
    ax3.set_xlabel('x (position)', fontsize=12)
    ax3.set_ylabel('v = dx/dt (velocity)', fontsize=12)
    ax3.set_title('Medium Region (Zoomed)', fontsize=12)
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Subplot 4: Outer region view
    ax4 = fig.add_subplot(224)
    ax4.scatter(x_poincare, v_poincare, s=0.5, alpha=0.6, c='blue')
    ax4.set_xlabel('x (position)', fontsize=12)
    ax4.set_ylabel('v = dx/dt (velocity)', fontsize=12)
    ax4.set_title('Outer Region (Zoomed)', fontsize=12)
    ax4.set_xlim(-6, 6)
    ax4.set_ylim(-6, 6)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'poincare_map_forced_duffing_p1_{p1}_p2_{p2}.png', dpi=300, bbox_inches='tight')
    print(f"Poincare map saved as 'poincare_map_forced_duffing_p1_{p1}_p2_{p2}.png'")
    
    # Figure 2: Complete Poincare map with detailed annotations
    fig2 = plt.figure(figsize=(14, 10))
    ax5 = fig2.add_subplot(111)
    ax5.scatter(x_poincare, v_poincare, s=0.5, alpha=0.6, c='blue')
    ax5.set_xlabel('x (position)', fontsize=14)
    ax5.set_ylabel('v = dx/dt (velocity)', fontsize=14)
    ax5.set_title(f'Complete Poincare Map for Forced, Damped Duffing Oscillator\n', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Initial x = 0')
    ax5.legend(loc='upper right', fontsize=12)
    
    # Add text box with parameters
    textstr = f'Parameters:\n'
    textstr += f'  Damping (p1): {p1}\n'
    textstr += f'  Forcing amplitude (p2): {p2}\n'
    textstr += f'  Forcing frequency: 3\n'
    textstr += f'  Poincare section: T = 2p/3\n'
    textstr += f'  Initial conditions: {total_points}\n'
    textstr += f'  Poincare points: {len(poincare_points)}\n'
    textstr += f'  v0 range: [{v_min}, {v_max}]'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(f'poincare_map_forced_duffing_full_p1_{p1}_p2_{p2}.png', dpi=300, bbox_inches='tight')
    print(f"Complete Poincare map saved as 'poincare_map_forced_duffing_full_p1_{p1}_p2_{p2}.png'")
    
    # ============================================
    # STATISTICS
    # ============================================
    
    print(f"\n{'='*60}")
    print("Poincare MAP STATISTICS")
    print(f"{'='*60}")
    print(f"Equation: d^2x/dt^2 + x + x^3 = p1*x*dx/dt + p2*sin(3t)")
    print(f"Damping parameter (p1): {p1}")
    print(f"Forcing amplitude (p2): {p2}")
    print(f"Forcing frequency: 3")
    print(f"Poincare section: T = 2p/3 = {period:.6f}")
    print(f"Initial conditions processed: {total_points}")
    print(f"Total Poincare points collected: {len(poincare_points)}")
    print(f"X-coordinate range: [{np.min(x_poincare):.4f}, {np.max(x_poincare):.4f}]")
    print(f"V-coordinate range: [{np.min(v_poincare):.4f}, {np.max(v_poincare):.4f}]")
    print(f"Average points per trajectory: {len(poincare_points)/total_points:.1f}")
    print(f"{'='*60}")
    
    # Calculate some energy statistics (not conserved due to damping/forcing)
    energies = v_poincare**2/2 + x_poincare**2/2 + x_poincare**4/4
    print(f"Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}]")
    print(f"Average energy: {np.mean(energies):.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) >= 3:
        try:
            p1 = float(sys.argv[1])
            p2 = float(sys.argv[2])
            print(f"Using parameters from command line: p1={p1}, p2={p2}")
            construct_poincare_map(p1, p2)
        except ValueError:
            print("Invalid parameters. Using default values: p1=0.1, p2=0.5")
            construct_poincare_map()
    else:
        print("No parameters provided. Using default values: p1=0.1, p2=0.5")
        print("Usage: python poincare_map_forced_duffing.py <p1> <p2>")
        construct_poincare_map()