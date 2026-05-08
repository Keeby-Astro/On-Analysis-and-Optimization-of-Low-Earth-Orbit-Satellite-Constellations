import matplotlib.pyplot as plt

def ropv_visualization(radius_vectors_np, angular_momentum_vectors_np, 
                           eccentricity_vectors_np, nodal_vectors_np, 
                           velocity_vectors_np, sma, ecc, inc, RAD_TO_DEG, 
                           ROPV, ROPV_PCA):
    """
    Plot 3D visualizations for orbital vectors and PCA/UMAP reduced vectors.
    """

    # First 3D visualization: Orbital Vectors
    print("Do you want to plot the 3D visualization of orbital vectors?")
    plotting_vec = int(input("Enter 0 for no, 1 for yes: "))
    if plotting_vec == 1:
        # Create a figure with 6 subplots for visualizing orbital vectors
        fig = plt.figure(figsize=(18, 12))
        plt.suptitle("3D Visualization of Orbital Vectors", fontsize=16)

        # Subplot 1: Radius Vectors
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.set_title("Radius Vectors")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.scatter(radius_vectors_np[:, 0], radius_vectors_np[:, 1], radius_vectors_np[:, 2],
                    c='b', marker='o', s=20)

        # Subplot 2: Angular Momentum Vectors
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.set_title("Angular Momentum Vectors")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.scatter(angular_momentum_vectors_np[:, 0], angular_momentum_vectors_np[:, 1],
                    angular_momentum_vectors_np[:, 2], c='r', marker='o', s=20)

        # Subplot 3: Eccentricity Vectors
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.set_title("Eccentricity Vectors")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.scatter(eccentricity_vectors_np[:, 0], eccentricity_vectors_np[:, 1],
                    eccentricity_vectors_np[:, 2], c='g', marker='o', s=20)

        # Subplot 4: Nodal Vectors
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.set_title("Nodal Vectors")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        ax4.scatter(nodal_vectors_np[:, 0], nodal_vectors_np[:, 1],
                    nodal_vectors_np[:, 2], c='m', marker='o', s=20)

        # Subplot 5: Velocity Vectors
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.set_title("Velocity Vectors")
        ax5.set_xlabel("X")
        ax5.set_ylabel("Y")
        ax5.set_zlabel("Z")
        ax5.scatter(velocity_vectors_np[:, 0], velocity_vectors_np[:, 1],
                    velocity_vectors_np[:, 2], c='c', marker='o', s=20)

        # Subplot 6: 3D plot of Semi-major Axis, Eccentricity, and RAAN
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.set_title("3D Orbital Elements")
        ax6.set_xlabel("Semi-major Axis (km)")
        ax6.set_ylabel("Eccentricity")
        ax6.set_zlabel("Inclination (deg)")
        ax6.scatter(sma, ecc, inc * RAD_TO_DEG, c='k', marker='o', s=30)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    else:
        pass

    # Second 3D visualization: ROPV vectors (PCA/UMAP)
    print("Do you want to plot the 3D visualization of ROPV vectors?")
    plotting_ROPV = int(input("Enter 0 for no, 1 for yes: "))
    if plotting_ROPV == 1:
        # Create a figure with 6 subplots for different PCA vector combinations
        fig = plt.figure(figsize=(18, 12))
        plt.suptitle("PCA Vector Combinations in 3D", fontsize=16)

        # Subplot 1: Radius/Velocity/Angular Momentum PCA Vectors
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.set_title("Radius/Velocity/Angular Momentum")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.scatter(ROPV[:, 0], ROPV[:, 1], ROPV[:, 2], c='r', marker='o')

        # Subplot 2: Velocity/Angular Momentum/Eccentricity PCA Vectors
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.set_title("Velocity/Angular Momentum/Eccentricity")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        ax2.scatter(ROPV[:, 1], ROPV[:, 2], ROPV[:, 3], c='g', marker='o')

        # Subplot 3: Angular Momentum/Eccentricity/Nodal PCA Vectors
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.set_title("Angular Momentum/Eccentricity/Nodal")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
        ax3.set_zlabel("Z")
        ax3.scatter(ROPV[:, 2], ROPV[:, 3], ROPV[:, 4], c='b', marker='o')

        # Subplot 4: Eccentricity/Nodal/Radius PCA Vectors
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        ax4.set_title("Eccentricity/Nodal/Radius")
        ax4.set_xlabel("X")
        ax4.set_ylabel("Y")
        ax4.set_zlabel("Z")
        ax4.scatter(ROPV[:, 3], ROPV[:, 4], ROPV[:, 0], c='m', marker='o')

        # Subplot 5: Nodal/Radius/Velocity PCA Vectors
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        ax5.set_title("Nodal/Radius/Velocity")
        ax5.set_xlabel("X")
        ax5.set_ylabel("Y")
        ax5.set_zlabel("Z")
        ax5.scatter(ROPV[:, 4], ROPV[:, 0], ROPV[:, 1], c='c', marker='o')

        # Subplot 6: Combined PCA Vectors using UMAP
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        ax6.set_title("Combined PCA Vectors (UMAP)")
        ax6.set_xlabel("X")
        ax6.set_ylabel("Y")
        ax6.set_zlabel("Z")
        ax6.scatter(ROPV_PCA[:, 0], ROPV_PCA[:, 1], ROPV_PCA[:, 2], c='k', marker='o')

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
    
    else:
        pass