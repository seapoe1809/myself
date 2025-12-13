#!/usr/bin/env python3
"""
Cosmic Web Filament Simulation
Generates a sample filament web using stochastic equations from cosmological theory

Uses:
- Gaussian random field for initial density perturbations
- Zel'dovich approximation for structure evolution
- Hessian-based filament detection
- Cox process for galaxy placement
- Network graph for filament connections
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, sobel
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class CosmicWebSimulator:
    """
    Simulates cosmic web structure using stochastic cosmological equations
    """
    
    def __init__(self, box_size=100.0, grid_size=256, n_particles=50000):
        """
        Initialize simulator
        
        Parameters:
        -----------
        box_size : float
            Simulation box size in Mpc/h
        grid_size : int
            Number of grid cells per dimension
        n_particles : int
            Number of dark matter particles
        """
        self.box_size = box_size
        self.grid_size = grid_size
        self.n_particles = n_particles
        self.cell_size = box_size / grid_size
        
        # Cosmological parameters
        self.omega_m = 0.3  # Matter density
        self.sigma_8 = 0.8  # Amplitude of fluctuations
        self.n_s = 0.96     # Spectral index
        self.h = 0.7        # Hubble parameter
        
        # Simulation data
        self.density_field = None
        self.particles = None
        self.velocities = None
        self.halos = None
        self.filaments = None
        
    def generate_power_spectrum(self, k):
        """
        Generate matter power spectrum P(k)
        
        P(k) = A * k^n_s * T^2(k)
        
        Using Eisenstein-Hu transfer function approximation
        """
        # Normalize k
        q = k / (self.omega_m * self.h**2)
        
        # Transfer function (Bardeen et al. approximation)
        T_k = np.log(1 + 2.34*q) / (2.34*q) * \
              (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
        T_k = np.where(q > 0, T_k, 1.0)
        
        # Power spectrum
        A = self.sigma_8**2  # Normalization
        P_k = A * k**self.n_s * T_k**2
        
        return P_k
    
    def generate_gaussian_random_field(self):
        """
        Generate initial Gaussian random density field δ(x)
        
        Using Fourier method with power spectrum P(k)
        """
        print("Generating Gaussian random field...")
        
        # Create k-space grid
        kx = np.fft.fftfreq(self.grid_size, d=self.cell_size) * 2 * np.pi
        ky = np.fft.fftfreq(self.grid_size, d=self.cell_size) * 2 * np.pi
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
        k_mag[0, 0] = 1e-10  # Avoid division by zero
        
        # Get power spectrum
        P_k = self.generate_power_spectrum(k_mag)
        
        # Generate random phases and amplitudes
        # δ(k) = √P(k) * (a + ib) where a,b ~ N(0,1)
        random_field = np.random.randn(self.grid_size, self.grid_size) + \
                       1j * np.random.randn(self.grid_size, self.grid_size)
        
        # Apply power spectrum
        delta_k = np.sqrt(P_k) * random_field
        delta_k[0, 0] = 0  # Zero mean
        
        # Transform to real space
        delta_x = np.real(np.fft.ifft2(delta_k))
        
        # Normalize
        delta_x = (delta_x - delta_x.mean()) / delta_x.std() * self.sigma_8
        
        return delta_x, k_mag, kx_grid, ky_grid
    
    def compute_displacement_field(self, delta_k, kx_grid, ky_grid, k_mag):
        """
        Compute Zel'dovich displacement field Ψ(q)
        
        Ψ(q) = -∇φ(q) where ∇²φ = δ
        In Fourier space: Ψ(k) = -ik/k² * δ(k)
        """
        print("Computing Zel'dovich displacement field...")
        
        # Avoid division by zero
        k_sq = k_mag**2
        k_sq[0, 0] = 1
        
        # Displacement in Fourier space
        psi_x_k = -1j * kx_grid / k_sq * delta_k
        psi_y_k = -1j * ky_grid / k_sq * delta_k
        
        psi_x_k[0, 0] = 0
        psi_y_k[0, 0] = 0
        
        # Transform to real space
        psi_x = np.real(np.fft.ifft2(psi_x_k))
        psi_y = np.real(np.fft.ifft2(psi_y_k))
        
        return psi_x, psi_y
    
    def evolve_particles_zeldovich(self, growth_factor=1.0):
        """
        Evolve particles using Zel'dovich approximation with stochastic noise
        
        x(q,t) = q + D(t)*Ψ(q) + ∫η(q,t')dt'
        
        where D(t) is the growth factor and η is stochastic noise
        """
        print("Evolving particles with Zel'dovich approximation...")
        
        # Generate initial density field
        delta_x, k_mag, kx_grid, ky_grid = self.generate_gaussian_random_field()
        delta_k = np.fft.fft2(delta_x)
        
        # Compute displacement field
        psi_x, psi_y = self.compute_displacement_field(delta_k, kx_grid, ky_grid, k_mag)
        
        # Initialize particles on a grid (Lagrangian coordinates q)
        n_per_dim = int(np.sqrt(self.n_particles))
        q_x = np.linspace(0, self.box_size, n_per_dim, endpoint=False)
        q_y = np.linspace(0, self.box_size, n_per_dim, endpoint=False)
        qx_grid, qy_grid = np.meshgrid(q_x, q_y)
        
        # Interpolate displacement to particle positions
        from scipy.interpolate import RegularGridInterpolator
        
        x_coords = np.linspace(0, self.box_size, self.grid_size, endpoint=False)
        y_coords = np.linspace(0, self.box_size, self.grid_size, endpoint=False)
        
        interp_psi_x = RegularGridInterpolator((y_coords, x_coords), psi_x, 
                                                bounds_error=False, fill_value=0)
        interp_psi_y = RegularGridInterpolator((y_coords, x_coords), psi_y,
                                                bounds_error=False, fill_value=0)
        
        # Flatten particle positions
        q_flat = np.column_stack([qy_grid.ravel(), qx_grid.ravel()])
        
        # Apply Zel'dovich displacement
        displacement_x = interp_psi_x(q_flat) * growth_factor
        displacement_y = interp_psi_y(q_flat) * growth_factor
        
        # Add stochastic noise term: ∫η(q,t')dt'
        noise_amplitude = 0.5 * self.cell_size
        noise_x = np.random.randn(len(q_flat)) * noise_amplitude
        noise_y = np.random.randn(len(q_flat)) * noise_amplitude
        
        # Final Eulerian positions
        x_final = qx_grid.ravel() + displacement_x + noise_x
        y_final = qy_grid.ravel() + displacement_y + noise_y
        
        # Periodic boundary conditions
        x_final = x_final % self.box_size
        y_final = y_final % self.box_size
        
        # Store particles
        self.particles = np.column_stack([x_final, y_final])
        
        # Compute velocities (proportional to displacement for growing mode)
        H = 100 * self.h  # km/s/Mpc
        f = self.omega_m**0.55  # Growth rate
        self.velocities = np.column_stack([
            H * f * displacement_x,
            H * f * displacement_y
        ])
        
        # Compute density field from particles
        self.density_field = self.particles_to_density()
        
        return self.particles, self.density_field
    
    def particles_to_density(self):
        """
        Convert particle distribution to density field using CIC assignment
        """
        density = np.zeros((self.grid_size, self.grid_size))
        
        for x, y in self.particles:
            # Grid indices
            ix = int(x / self.cell_size) % self.grid_size
            iy = int(y / self.cell_size) % self.grid_size
            density[iy, ix] += 1
            
        # Smooth with Gaussian kernel
        density = gaussian_filter(density, sigma=1.5)
        
        # Convert to density contrast δ = (ρ - ρ̄) / ρ̄
        mean_density = density.mean()
        if mean_density > 0:
            density = (density - mean_density) / mean_density
            
        return density
    
    def compute_hessian_eigenvalues(self, smoothing_scale=3.0):
        """
        Compute Hessian matrix eigenvalues for filament detection
        
        H_ij = ∂²δ_s / ∂x_i ∂x_j
        
        Filaments identified where λ₁ < 0, λ₂ < 0, |λ₃| << |λ₁|
        """
        print("Computing Hessian eigenvalues for filament detection...")
        
        # Smooth density field
        delta_s = gaussian_filter(self.density_field, sigma=smoothing_scale)
        
        # Compute second derivatives (Hessian components)
        # H_xx = ∂²δ/∂x²
        H_xx = np.gradient(np.gradient(delta_s, axis=1), axis=1)
        # H_yy = ∂²δ/∂y²
        H_yy = np.gradient(np.gradient(delta_s, axis=0), axis=0)
        # H_xy = ∂²δ/∂x∂y
        H_xy = np.gradient(np.gradient(delta_s, axis=1), axis=0)
        
        # Compute eigenvalues at each point
        lambda1 = np.zeros_like(delta_s)
        lambda2 = np.zeros_like(delta_s)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                H = np.array([[H_xx[i,j], H_xy[i,j]], 
                              [H_xy[i,j], H_yy[i,j]]])
                eigenvalues = np.linalg.eigvalsh(H)
                lambda1[i,j] = eigenvalues[0]  # Smaller eigenvalue
                lambda2[i,j] = eigenvalues[1]  # Larger eigenvalue
        
        # Filamentarity index: F = (λ₂ - λ₁) / (λ₂ + λ₁)
        denom = np.abs(lambda2) + np.abs(lambda1) + 1e-10
        filamentarity = (lambda2 - lambda1) / denom
        
        # Identify filament regions
        # Filaments: both eigenvalues negative (ridge), λ₂ close to zero
        filament_mask = (lambda1 < -0.1) & (np.abs(lambda2) < np.abs(lambda1) * 0.8)
        
        return lambda1, lambda2, filamentarity, filament_mask, delta_s
    
    def identify_halos_peaks(self, threshold=2.0, min_separation=3.0):
        """
        Identify dark matter halos as peaks in smoothed density field
        
        Using excursion set theory: peaks above threshold δ_c
        """
        print("Identifying halos from density peaks...")
        
        # Smooth density field
        delta_s = gaussian_filter(self.density_field, sigma=2.0)
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        local_max = maximum_filter(delta_s, size=int(min_separation))
        peaks = (delta_s == local_max) & (delta_s > threshold)
        
        # Get peak positions
        peak_coords = np.array(np.where(peaks)).T
        
        # Convert to physical coordinates
        halo_positions = peak_coords * self.cell_size
        halo_masses = delta_s[peaks]
        
        # Filter halos that are too close (keep more massive one)
        if len(halo_positions) > 0:
            tree = cKDTree(halo_positions)
            pairs = tree.query_pairs(min_separation * self.cell_size)
            
            to_remove = set()
            for i, j in pairs:
                if halo_masses[i] > halo_masses[j]:
                    to_remove.add(j)
                else:
                    to_remove.add(i)
            
            keep = [i for i in range(len(halo_positions)) if i not in to_remove]
            halo_positions = halo_positions[keep]
            halo_masses = halo_masses[keep]
        
        self.halos = {
            'positions': halo_positions,
            'masses': halo_masses
        }
        
        return self.halos
    
    def generate_filament_network(self, connection_scale=15.0):
        """
        Generate filament network using modified random geometric graph
        
        p_ij = p₀ * (M_i * M_j / M_*²)^α * exp(-d_ij / λ_fil)
        
        Also uses minimum spanning tree for backbone
        """
        print("Generating filament network...")
        
        if self.halos is None or len(self.halos['positions']) < 3:
            print("Not enough halos found!")
            return None
            
        positions = self.halos['positions']
        masses = self.halos['masses']
        n_halos = len(positions)
        
        # Compute distance matrix
        dist_matrix = np.zeros((n_halos, n_halos))
        for i in range(n_halos):
            for j in range(n_halos):
                # Periodic distance
                dx = np.abs(positions[i] - positions[j])
                dx = np.minimum(dx, self.box_size - dx)
                dist_matrix[i,j] = np.sqrt(np.sum(dx**2))
        
        # Connection probability (Eq. from network models)
        M_star = np.median(masses)
        alpha = 0.5
        lambda_fil = connection_scale
        
        # p_ij = p₀ * (M_i * M_j / M_*²)^α * exp(-d_ij / λ_fil)
        mass_term = np.outer(masses, masses) / M_star**2
        p_connect = 0.3 * (mass_term**alpha) * np.exp(-dist_matrix / lambda_fil)
        
        # Generate random connections
        random_vals = np.random.rand(n_halos, n_halos)
        adjacency = (random_vals < p_connect) & (dist_matrix < 2 * lambda_fil)
        np.fill_diagonal(adjacency, False)
        
        # Make symmetric
        adjacency = adjacency | adjacency.T
        
        # Add minimum spanning tree to ensure connectivity
        mst = minimum_spanning_tree(csr_matrix(dist_matrix))
        mst_dense = mst.toarray()
        mst_adjacency = (mst_dense > 0) | (mst_dense.T > 0)
        
        # Combine MST with random connections
        adjacency = adjacency | mst_adjacency
        
        # Extract edges
        edges = []
        for i in range(n_halos):
            for j in range(i+1, n_halos):
                if adjacency[i,j]:
                    edges.append((i, j, dist_matrix[i,j]))
        
        self.filaments = {
            'edges': edges,
            'adjacency': adjacency,
            'positions': positions,
            'masses': masses
        }
        
        return self.filaments
    
    def place_galaxies_cox_process(self, n_galaxies=5000):
        """
        Place galaxies using Cox (doubly stochastic Poisson) process
        
        Λ(x) = Λ₀ * exp[b * δ(x) + ε(x)]
        
        where b is bias and ε is stochastic field
        """
        print("Placing galaxies with Cox process...")
        
        # Galaxy bias parameter
        b = 1.5
        
        # Base intensity
        Lambda_0 = n_galaxies / self.box_size**2
        
        # Stochastic field ε(x)
        epsilon = np.random.randn(self.grid_size, self.grid_size) * 0.3
        epsilon = gaussian_filter(epsilon, sigma=2.0)
        
        # Intensity field
        delta_clipped = np.clip(self.density_field, -0.9, 10)  # Avoid log issues
        Lambda = Lambda_0 * np.exp(b * delta_clipped + epsilon)
        
        # Normalize to get probability
        Lambda_sum = Lambda.sum() * self.cell_size**2
        prob = Lambda / Lambda.sum()
        
        # Sample galaxy positions
        flat_prob = prob.ravel()
        indices = np.random.choice(len(flat_prob), size=n_galaxies, p=flat_prob)
        
        # Convert to positions with random offset within cell
        iy, ix = np.unravel_index(indices, (self.grid_size, self.grid_size))
        x_gal = (ix + np.random.rand(n_galaxies)) * self.cell_size
        y_gal = (iy + np.random.rand(n_galaxies)) * self.cell_size
        
        galaxies = np.column_stack([x_gal, y_gal])
        
        return galaxies
    
    def compute_correlation_function(self, galaxies, n_bins=20, r_max=None):
        """
        Compute two-point correlation function ξ(r)
        
        ξ(r) = (DD/RR) - 1
        """
        if r_max is None:
            r_max = self.box_size / 4
            
        # Compute pair distances
        tree = cKDTree(galaxies)
        
        # Bin edges
        r_bins = np.linspace(0, r_max, n_bins + 1)
        r_centers = 0.5 * (r_bins[:-1] + r_bins[1:])
        
        # Count pairs in each bin
        DD = np.zeros(n_bins)
        for i, pos in enumerate(galaxies):
            distances = tree.query_ball_point(pos, r_max)
            for j in distances:
                if j > i:
                    d = np.sqrt(np.sum((galaxies[i] - galaxies[j])**2))
                    bin_idx = np.searchsorted(r_bins, d) - 1
                    if 0 <= bin_idx < n_bins:
                        DD[bin_idx] += 1
        
        # Random expectation
        n_gal = len(galaxies)
        n_pairs = n_gal * (n_gal - 1) / 2
        volume = self.box_size**2
        
        RR = np.zeros(n_bins)
        for i in range(n_bins):
            r1, r2 = r_bins[i], r_bins[i+1]
            shell_area = np.pi * (r2**2 - r1**2)
            RR[i] = n_pairs * shell_area / volume
        
        # Correlation function
        xi = np.zeros(n_bins)
        valid = RR > 0
        xi[valid] = DD[valid] / RR[valid] - 1
        
        return r_centers, xi
    
    def run_full_simulation(self):
        """
        Run complete cosmic web simulation
        """
        print("="*60)
        print("COSMIC WEB FILAMENT SIMULATION")
        print("="*60)
        print(f"Box size: {self.box_size} Mpc/h")
        print(f"Grid size: {self.grid_size}")
        print(f"Particles: {self.n_particles}")
        print()
        
        # Step 1: Evolve particles with Zel'dovich approximation
        particles, density = self.evolve_particles_zeldovich(growth_factor=2.5)
        
        # Step 2: Compute Hessian for filament detection
        l1, l2, filamentarity, filament_mask, delta_s = self.compute_hessian_eigenvalues()
        
        # Step 3: Identify halos
        halos = self.identify_halos_peaks(threshold=1.5, min_separation=4.0)
        print(f"Found {len(halos['positions'])} halos")
        
        # Step 4: Generate filament network
        filaments = self.generate_filament_network(connection_scale=12.0)
        if filaments:
            print(f"Generated {len(filaments['edges'])} filament connections")
        
        # Step 5: Place galaxies
        galaxies = self.place_galaxies_cox_process(n_galaxies=3000)
        print(f"Placed {len(galaxies)} galaxies")
        
        return {
            'particles': particles,
            'density': density,
            'delta_smoothed': delta_s,
            'filamentarity': filamentarity,
            'filament_mask': filament_mask,
            'halos': halos,
            'filaments': filaments,
            'galaxies': galaxies,
            'eigenvalues': (l1, l2)
        }


def create_visualization(sim, results, output_path):
    """
    Create comprehensive visualization of the cosmic web
    """
    print("\nCreating visualizations...")
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Stochastic Cosmic Web Filament Simulation', fontsize=16, fontweight='bold')
    
    # Color scheme
    cmap_density = 'inferno'
    cmap_filament = 'viridis'
    
    # 1. Particle distribution
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(results['particles'][:, 0], results['particles'][:, 1], 
                s=0.1, alpha=0.3, c='white')
    ax1.set_facecolor('black')
    ax1.set_xlim(0, sim.box_size)
    ax1.set_ylim(0, sim.box_size)
    ax1.set_xlabel('x [Mpc/h]')
    ax1.set_ylabel('y [Mpc/h]')
    ax1.set_title('Dark Matter Particles\n(Zel\'dovich Approximation)')
    ax1.set_aspect('equal')
    
    # 2. Density field
    ax2 = fig.add_subplot(2, 3, 2)
    extent = [0, sim.box_size, 0, sim.box_size]
    im2 = ax2.imshow(results['delta_smoothed'], extent=extent, origin='lower',
                     cmap=cmap_density, vmin=-1, vmax=5)
    plt.colorbar(im2, ax=ax2, label='δ = (ρ-ρ̄)/ρ̄')
    ax2.set_xlabel('x [Mpc/h]')
    ax2.set_ylabel('y [Mpc/h]')
    ax2.set_title('Smoothed Density Field\nP(k) = A·k^n·T²(k)')
    ax2.set_aspect('equal')
    
    # 3. Filamentarity index
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.imshow(results['filamentarity'], extent=extent, origin='lower',
                     cmap=cmap_filament, vmin=0, vmax=1)
    plt.colorbar(im3, ax=ax3, label='F = (λ₂-λ₁)/(λ₂+λ₁)')
    ax3.set_xlabel('x [Mpc/h]')
    ax3.set_ylabel('y [Mpc/h]')
    ax3.set_title('Filamentarity Index\n(Hessian Eigenvalue Analysis)')
    ax3.set_aspect('equal')
    
    # 4. Filament network with halos
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(results['delta_smoothed'], extent=extent, origin='lower',
               cmap='Greys', alpha=0.5, vmin=-1, vmax=3)
    
    # Plot filament edges
    if results['filaments']:
        for i, j, dist in results['filaments']['edges']:
            pos_i = results['filaments']['positions'][i]
            pos_j = results['filaments']['positions'][j]
            ax4.plot([pos_i[1], pos_j[1]], [pos_i[0], pos_j[0]], 
                    'c-', alpha=0.6, linewidth=1.5)
    
    # Plot halos
    if results['halos']:
        halo_pos = results['halos']['positions']
        halo_mass = results['halos']['masses']
        sizes = 50 + 200 * (halo_mass - halo_mass.min()) / (halo_mass.max() - halo_mass.min() + 0.1)
        ax4.scatter(halo_pos[:, 1], halo_pos[:, 0], s=sizes, c='red', 
                   edgecolors='yellow', linewidth=1, alpha=0.8, zorder=5)
    
    ax4.set_xlim(0, sim.box_size)
    ax4.set_ylim(0, sim.box_size)
    ax4.set_xlabel('x [Mpc/h]')
    ax4.set_ylabel('y [Mpc/h]')
    ax4.set_title('Filament Network\np_ij ∝ (M_iM_j)^α exp(-d/λ)')
    ax4.set_aspect('equal')
    
    # 5. Galaxy distribution (Cox process)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor('black')
    ax5.scatter(results['galaxies'][:, 0], results['galaxies'][:, 1],
               s=2, c='white', alpha=0.6)
    
    # Overlay filament network
    if results['filaments']:
        for i, j, dist in results['filaments']['edges']:
            pos_i = results['filaments']['positions'][i]
            pos_j = results['filaments']['positions'][j]
            ax5.plot([pos_i[1], pos_j[1]], [pos_i[0], pos_j[0]], 
                    'orange', alpha=0.4, linewidth=1)
    
    ax5.set_xlim(0, sim.box_size)
    ax5.set_ylim(0, sim.box_size)
    ax5.set_xlabel('x [Mpc/h]')
    ax5.set_ylabel('y [Mpc/h]')
    ax5.set_title('Galaxy Distribution\nΛ(x) = Λ₀·exp[b·δ(x)]')
    ax5.set_aspect('equal')
    
    # 6. Two-point correlation function
    ax6 = fig.add_subplot(2, 3, 6)
    r, xi = sim.compute_correlation_function(results['galaxies'], n_bins=15)
    
    # Plot measured correlation
    valid = xi > 0
    ax6.loglog(r[valid], xi[valid], 'bo-', markersize=8, label='Measured ξ(r)')
    
    # Plot theoretical power law: ξ(r) = (r/r₀)^(-γ)
    r0 = 5.0  # Correlation length in Mpc/h
    gamma = 1.8
    xi_theory = (r / r0)**(-gamma)
    ax6.loglog(r, xi_theory, 'r--', linewidth=2, 
              label=f'Theory: (r/{r0})^(-{gamma})')
    
    ax6.set_xlabel('r [Mpc/h]')
    ax6.set_ylabel('ξ(r)')
    ax6.set_title('Two-Point Correlation Function\nξ(r) = (r/r₀)^(-γ)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(1, 30)
    ax6.set_ylim(0.01, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def create_cosmic_web_beauty_plot(sim, results, output_path):
    """
    Create a beautiful single visualization of the cosmic web
    """
    print("Creating cosmic web beauty visualization...")
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Background density field with custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    colors_cosmic = ['#000011', '#0a0a2e', '#1a1a4e', '#2d2d7a', 
                     '#4444aa', '#6666cc', '#9999ee', '#ccccff', '#ffffff']
    cmap_cosmic = LinearSegmentedColormap.from_list('cosmic', colors_cosmic)
    
    extent = [0, sim.box_size, 0, sim.box_size]
    
    # Show density field
    density_display = np.clip(results['delta_smoothed'], -1, 5)
    ax.imshow(density_display, extent=extent, origin='lower',
             cmap=cmap_cosmic, alpha=0.9)
    
    # Add filament network
    if results['filaments']:
        for i, j, dist in results['filaments']['edges']:
            pos_i = results['filaments']['positions'][i]
            pos_j = results['filaments']['positions'][j]
            # Gradient line based on distance
            alpha = np.exp(-dist / 20) * 0.8
            ax.plot([pos_i[1], pos_j[1]], [pos_i[0], pos_j[0]], 
                   color='#ff6600', alpha=alpha, linewidth=2)
    
    # Add galaxy points
    ax.scatter(results['galaxies'][:, 0], results['galaxies'][:, 1],
              s=1, c='white', alpha=0.4)
    
    # Add halo markers
    if results['halos']:
        halo_pos = results['halos']['positions']
        halo_mass = results['halos']['masses']
        sizes = 30 + 150 * (halo_mass - halo_mass.min()) / (halo_mass.max() - halo_mass.min() + 0.1)
        ax.scatter(halo_pos[:, 1], halo_pos[:, 0], s=sizes, 
                  c='#ffaa00', edgecolors='#ff6600', linewidth=1.5, 
                  alpha=0.9, zorder=5)
    
    ax.set_xlim(0, sim.box_size)
    ax.set_ylim(0, sim.box_size)
    ax.set_xlabel('x [Mpc/h]', fontsize=12)
    ax.set_ylabel('y [Mpc/h]', fontsize=12)
    ax.set_title('Cosmic Web: Galaxy Filaments & Clusters\n' + 
                 'Stochastic Simulation using Zel\'dovich Approximation + Hessian Analysis',
                fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_facecolor('#000011')
    
    # Add text box with equations
    textstr = '\n'.join([
        'Key Equations:',
        r'$\delta(\mathbf{x},t) \sim$ Gaussian Random Field',
        r'$\mathbf{x}(q,t) = q + D(t)\Psi(q) + \eta$',
        r'$H_{ij} = \partial^2\delta/\partial x_i\partial x_j$',
        r'$\Lambda(x) = \Lambda_0 \exp[b\cdot\delta(x)]$',
        r'$\xi(r) = (r/r_0)^{-\gamma}$'
    ])
    props = dict(boxstyle='round', facecolor='black', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', color='white', bbox=props, family='monospace')
    
    # Add scale bar
    scale_length = 10  # Mpc/h
    ax.plot([5, 5 + scale_length], [5, 5], 'w-', linewidth=3)
    ax.text(5 + scale_length/2, 7, f'{scale_length} Mpc/h', 
           color='white', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#000011')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


def create_eigenvalue_analysis_plot(sim, results, output_path):
    """
    Create visualization of Hessian eigenvalue analysis
    """
    print("Creating eigenvalue analysis plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    extent = [0, sim.box_size, 0, sim.box_size]
    
    l1, l2 = results['eigenvalues']
    
    # Lambda 1 (most negative eigenvalue)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(l1, extent=extent, origin='lower', cmap='RdBu', 
                     vmin=-2, vmax=2)
    plt.colorbar(im1, ax=ax1, label='λ₁')
    ax1.set_title('First Eigenvalue λ₁\n(Most negative in filaments)')
    ax1.set_xlabel('x [Mpc/h]')
    ax1.set_ylabel('y [Mpc/h]')
    
    # Lambda 2
    ax2 = axes[0, 1]
    im2 = ax2.imshow(l2, extent=extent, origin='lower', cmap='RdBu',
                     vmin=-2, vmax=2)
    plt.colorbar(im2, ax=ax2, label='λ₂')
    ax2.set_title('Second Eigenvalue λ₂\n(Near zero in filaments)')
    ax2.set_xlabel('x [Mpc/h]')
    ax2.set_ylabel('y [Mpc/h]')
    
    # Classification map
    ax3 = axes[1, 0]
    # Classify regions: void (both +), wall (one -), filament (both -), cluster (both very -)
    classification = np.zeros_like(l1)
    classification[(l1 > 0) & (l2 > 0)] = 0  # Void
    classification[(l1 < 0) & (l2 > 0)] = 1  # Wall/Sheet
    classification[(l1 < 0) & (l2 < 0) & (l2 > -0.5)] = 2  # Filament
    classification[(l1 < -0.5) & (l2 < -0.5)] = 3  # Cluster
    
    cmap_class = plt.cm.get_cmap('Set1', 4)
    im3 = ax3.imshow(classification, extent=extent, origin='lower', cmap=cmap_class,
                     vmin=-0.5, vmax=3.5)
    cbar3 = plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2, 3])
    cbar3.set_ticklabels(['Void', 'Wall', 'Filament', 'Cluster'])
    ax3.set_title('Cosmic Web Classification\n(Based on Hessian Eigenvalues)')
    ax3.set_xlabel('x [Mpc/h]')
    ax3.set_ylabel('y [Mpc/h]')
    
    # Eigenvalue scatter plot
    ax4 = axes[1, 1]
    # Subsample for plotting
    n_sample = 10000
    idx = np.random.choice(l1.size, n_sample, replace=False)
    l1_flat = l1.ravel()[idx]
    l2_flat = l2.ravel()[idx]
    
    ax4.scatter(l1_flat, l2_flat, s=1, alpha=0.3, c='blue')
    ax4.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('λ₁')
    ax4.set_ylabel('λ₂')
    ax4.set_title('Eigenvalue Distribution\nλ₁ ≤ λ₂ by definition')
    ax4.set_xlim(-3, 2)
    ax4.set_ylim(-3, 2)
    
    # Add region labels
    ax4.text(-2, 1, 'Wall\n(λ₁<0, λ₂>0)', fontsize=10, ha='center')
    ax4.text(-2, -1.5, 'Filament/Cluster\n(λ₁<0, λ₂<0)', fontsize=10, ha='center')
    ax4.text(1, 1, 'Void\n(λ₁>0, λ₂>0)', fontsize=10, ha='center')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Hessian Eigenvalue Analysis: $H_{ij} = ∂²δ/∂x_i∂x_j$', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return output_path


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("COSMIC WEB FILAMENT GENERATOR")
    print("Using Stochastic Cosmological Equations")
    print("="*70)
    print()
    
    # Create simulator
    sim = CosmicWebSimulator(
        box_size=100.0,   # 100 Mpc/h box
        grid_size=256,    # 256^2 grid
        n_particles=40000  # 40k particles
    )
    
    # Run simulation
    results = sim.run_full_simulation()
    
    # Create visualizations
    create_visualization(sim, results, '/mnt/user-data/outputs/cosmic_web_analysis.png')
    create_cosmic_web_beauty_plot(sim, results, '/mnt/user-data/outputs/cosmic_web_filaments.png')
    create_eigenvalue_analysis_plot(sim, results, '/mnt/user-data/outputs/hessian_analysis.png')
    
    print()
    print("="*70)
    print("SIMULATION COMPLETE")
    print("="*70)
    print()
    print("Generated files:")
    print("  1. cosmic_web_analysis.png - Multi-panel analysis")
    print("  2. cosmic_web_filaments.png - Beauty visualization")
    print("  3. hessian_analysis.png - Eigenvalue classification")
