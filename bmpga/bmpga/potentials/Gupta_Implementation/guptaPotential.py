import numpy as np
import itertools
import matplotlib.pyplot as plt

class GuptaPotential:
    # Dictionary of base parameters for Gupta Potential
    base_param_A = {"AgAg": 0.1031, "AuAu": 0.2096, "AgAu": 0.1488, "ZnZn": 0.1477}
    base_param_Z = {"AgAg": 1.1895, "AuAu": 1.8153, "AgAu": 1.4874, "ZnZn": 0.8900}
    base_param_p = {"AgAg": 10.85, "AuAu": 10.139, "AgAu": 10.494, "ZnZn": 9.689}
    base_param_q = {"AgAg": 3.18, "AuAu": 4.033, "AgAu": 3.607, "ZnZn": 4.602}
    #base_param_r0 = {"AgAg": 2.8921, "AuAu": 2.885, "AgAu": 2.8885, "ZnZn": 0.8614}
    base_param_r0 = {"AgAg": 2.8921, "AuAu": 2.885, "AgAu": 2.8885, "ZnZn": 0.86}

    def __init__(self, atom_list):
        # Create empty lists for the various parameters
        self.param_A = []
        self.param_Z = []
        self.param_p = []
        self.param_q = []
        self.param_r0 = []
        
        # Iterate over the pairs of interacting atoms
        for atom_pair in itertools.permutations(atom_list, 2):
            # Some string manipulations stuff to get the strings correct
            atom_list = list(atom_pair)
            atom_list.sort()
            atom_str = "".join(atom_list)     
            
            # Lookup the parameters from the dictionarys and append them to the lists 
            self.param_A.append(self.base_param_A[atom_str])
            self.param_Z.append(self.base_param_Z[atom_str])
            self.param_p.append(self.base_param_p[atom_str])
            self.param_q.append(self.base_param_q[atom_str])
            self.param_r0.append(self.base_param_r0[atom_str])
            
        # Turn the parameter lists into numpy arrays so we can do nicely vectorised computation later
        self.param_A = np.array(self.param_A)
        self.param_Z = np.array(self.param_Z)
        self.param_p = np.array(self.param_p)  
        self.param_q = np.array(self.param_q)
        self.param_r0 = np.array(self.param_r0)

        print("A =", self.param_A)
        print("Z =", self.param_Z)
        print("p =", self.param_p)
        print("q =", self.param_q)
        print("r0 = ",self.param_r0)

    def get_energy(self, cluster):
        pass
    
    def minimize(self, cluster):
        pass
    
    # Calculate Rij values for each interacting pair of atoms
    # def calculate_all_rij(self, coordinates, verbose=False):
        
    #     rijs = []

    #     for i in range(len(coordinates)):
    #         for j in range(i + 1, len(coordinates)):

    #             v = coordinates[j] - coordinates[i]
                
    #             rij = np.sqrt(np.sum(v**2))

    #             rijs.append(rij)

    #             if verbose:
    #                 print(f"V = {v}")
    #                 print(i,j)
    #                 print(f"rij = {rij}")
    #     #print(rijs)
    #     return rijs

    def calculate_all_rij(self, coordinates, verbose=False):

        rijs = []
        for c1, c2  in itertools.permutations(coordinates, 2):

            vec = c1-c2
            rijs.append(np.sqrt(np.sum(vec**2)))
        return rijs
    
    # Calculate repulsive energy term for each interacting pair and return total repulsive energy
    def calculate_repulsive(self, rijs):
        
        rep_en = np.sum(self.param_A * np.exp(- 1 * self.param_p * ((rijs / self.param_r0) - 1)))
            
        return rep_en
    
    # Calculate attractive energy term for each interacting pair and return total attractive energy
    def calculate_attractive(self, rijs):
        
        att_en = np.sum(self.param_Z * np.sqrt(np.exp(- 2 * self.param_q * ((rijs / self.param_r0) - 1))))
        
        return att_en

    # Calculate total energy for the cluster
    def calculate_total_energy(self, coordinates):
        
        rijs = self.calculate_all_rij(coordinates)
        rep = self.calculate_repulsive(rijs)
        attract = self.calculate_attractive(rijs)

        print("Rij values =", rijs)
        print("Repulsive Energy (eV) =", rep)
        print("Attractive Energy (eV) =", attract)
        
        return rep - attract

if __name__ == "__main__":
    
    # Ag2 cluster
    a1 = GuptaPotential(["Ag", "Ag"])
    c1 = np.array([[0,0,2.8921],[0,0,0]])

    # Ag3 cluster
    #a1 = GuptaPotential(["Ag", "Ag", "Ag"])
    #c1 = np.array([[0,0,1.676540],[0,1.304549,-0.838270], [0,-1.304549,-0.838270]])
    
    # Zn3 cluster
    #a2 = GuptaPotential(["Zn", "Zn", "Zn"])
    #c2 = np.array([[-0.0025257515,-1.1132616080,-0.1490790579],[0.6823000831,-0.6720141616, 0.1308423179],[0.2779619653,-1.2199766042,0.6583738879]])

    # Calculates Rij
    rij_val = a1.calculate_all_rij(c1)
    #rij_val2 = a2.calculate_all_rij(c2)

    # Calculate repulsive energy term
    repulsive_en = a1.calculate_repulsive(rij_val)
    #repulsive_en2 = a2.calculate_repulsive(rij_val2)

    # Calculate attractive energy term
    att_en = a1.calculate_attractive(rij_val)
    #att_en2 = a2.calculate_attractive(rij_val2)

    # Calculate total energy of the cluster
    tot_en = a1.calculate_total_energy(c1)
    #tot_en2 = a2.calculate_total_energy(c2)

    # Print Rij, repulsive energy, attractive energy, total eneregy
    
    # a1 Rijs and energies
    print("Cluster Energy (eV) =", tot_en)

    # a2 Rijs and energies
    #print("Cluster Energy (eV):", tot_en2)

    # Plot 2D Potential Energy surface

    #Ag2 = GuptaPotential(["Ag", "Ag"])
    #rs = np.linspace(1.8,8, 1000)
    #ens = []
    #for r in rs:
        #ens.append(Ag2.calculate_total_energy(np.array([[0,0,0],[0,0,r]])))
    
    #plt.figure(figsize=(12,8))
    #plt.plot(rs, ens)
    #plt.plot([1.8,8], [0,0], "k:")
    #plt.title('2D Gupta Potential for Ag-Ag interaction')
    #plt.xlabel("r$_{ij}$")
    #plt.ylabel("Energy / eV")
    #plt.show()

    #Zn2 = GuptaPotential(["Zn", "Zn"])
    #rs = np.linspace(0.45,1.6, 1000)
    #ens = []
    #for r in rs:
        #ens.append(Zn2.calculate_total_energy(np.array([[0,0,0],[0,0,r]])))
    
    #plt.figure(figsize=(12,8))
    #plt.plot(rs, ens)
    #plt.plot([0.45,1.6], [0,0], "k:")
    #plt.title('2D Gupta Potential for Zn-Zn interaction')
    #plt.xlabel("r$_{ij}$")
    #plt.ylabel("Energy / eV")
    #plt.show()
