from __future__ import annotations
import numpy as np


def intercalation_chemical_potential(data: dict, number_of_intercalant_per_unit_cell: int)-> np.ndarray:
    """
    This function calculates the chemical potential using DFT predicted values for the formation energies and corresponding compositions (placed in the form of a dictionary)
    
    Arguments
    ---
    data : dict
        A dictionary of crystallogrpahic data that contains the DFT predicted formation energies of differing intercalant-vacancy orderings and corresponding intercalant compositions.
    number_of_intercalant_per_unit_cell : int
        The number of intercalants that are degrees of freedom in the system (generally the number of available intercalant sites). 
    
    Returns
    ---
    chemical_potential : np.ndarray
        The chemical_potential calculated by the given DFT-predicted formation energies and intercalant-vacancy ordering compositions.
    """


def intercalation_voltage(data: dict,  refernce_chemical_potential:float, number_of_intercalant_per_unit_cell:int, electrons_per_species: int)-> np.ndarray, np.nndarry:
    """
    This function calculates voltage from a chemical_potential matrix using the Nernst equation: V=-(\mu-\mu^{0})/(ne) where:
        \mu is the chemical_potential.
        \mu^{0} is the chemical potential for the reference.
        n is the number of electrons_per_species.
        e is accouted for in VASP determined calculations (which are in eV) and are ignored in the usage of this function.
    
    Arguments
    ---
    data : dict
        A dictionary of crystallogrpahic data that contains the DFT predicted formation energies of differing intercalant-vacancy orderings and corresponding intercalant compositions.
    reference_chemical_potential : float 
        The chemical potential of the chemical reference in the Nernset equation (in eV)
    number_of_intercalant_per_unit_cell : int
        The number of potential intercalant ion sites present per unit cell.
    electrons_per_species : int
        Number of electrons released for each intercalant (e.g. 1 Li ion will release 1 e-)
    
    Returns
    ---
    voltage : np.ndarray
        Voltage as calculated from the input chemical_potential and reference chemical_potential. 
    voltage_corresponding_compositions : np.ndarray
        Compositions for each determined vertex of the voltage steps. 
    """


def GCMC_intercalation_voltage(GCMC_data: dict, refernce_chemical_potential:float, free_energy_reference_startstate:float, free_energy_reference_endstate: float, number_of_intercalant_per_unit_cell:int, electrons_per_species: int)-> np.ndarray, np.ndarray:
    """
    This function calculates voltage from a chemical_potential matrix using the Nernst equation: V=-(\mu-\mu^{0})/(ne) where:
        \mu is the chemical_potential.
        \mu^{0} is the chemical potential for the reference.
        n is the number of electrons_per_species.
        e is accouted for in VASP determined calculations (which are in eV) and are ignored in the usage of this function.
    
    Arguments
    ---
    GCMC_data : dict
        A dictionary of crystallogrpahic data that contains the GCMC predicted chemical potentials of differing intercalant-vacancy orderings and corresponding intercalant compositions.
    reference_chemical_potential : float
        The chemical potential of the chemical reference in the Nernset equation (in eV)
    free_energy_reference_startstate : float
        The free energy of the chemical reference with the lower number of intercalants (e.g the pristine structure).
    free_energy_reference_endstate : float
        The free energy of the chemical reference witht the larger number of intercalants
    number_of_intercalant_per_unit_cell : int
        The number of potential intercalant ion sites present per unit cell.
    electrons_per_species : int
        Number of electrons released for each intercalant (e.g. 1 Li ion will release 1 e-)
        
    Returns
    ---
    voltage : np.ndarray
        Voltage as calculated from the input chemical_potential and reference chemical_potential. 
    voltage_corresponding_compositions : np.ndarray
        Compositions for each determined vertex of the voltage steps. 
    """
