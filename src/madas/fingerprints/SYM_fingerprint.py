from ase.spacegroup import get_spacegroup
from ase import Atoms

from madas import Fingerprint
from madas.material import Material


def SYM_similarity(fingerprint1: Fingerprint, fingerprint2: Fingerprint) -> float:
    """
    Compute the similarity between two `SGFingerprint` objects.

    Similarily to the Tanimoto coefficient, the SG similarity is defined as the 
    intersection of symmetry operations of two materials, divided by their union. 
    """
    symops1 = set(fingerprint1.data["symops"])
    symops2 = set(fingerprint2.data["symops"])

    return len(symops1.intersection(symops2)) / len(symops1.union(symops2))


class SYMFingerprint(Fingerprint):
    """
    SYMmetry fingerprint, defined by the symmetry operations of a crystal lattice.

    These are determined using `ASE`s `spacegroup` module. 
    """


    def __init__(self, 
                 name = "SYM", 
                 similarity_function = SYM_similarity,
                 pass_on_exceptions = True) -> None:
        self.set_fp_type("SYM")
        self.set_name(name)
        self.set_similarity_function(similarity_function)
        self.set_pass_on_exceptions(pass_on_exceptions)

    def from_material(self, 
                      material: Material, 
                      symprec: float = 1e-1, 
                      unify_species: bool = False) -> object:
        """
        *Calculatate* fingerprint from a `Material` object.

        See `calculate` method for a description of the parameters.
        """
        self.set_mid(material)
        atoms = material.atoms
        return self.calculate(atoms, symprec=symprec, unify_species=unify_species)

    def calculate(self, 
                  atoms: Atoms, 
                  symprec: float = 1e-1, 
                  unify_species: bool = False) -> object:
        """
        Calculate the SG fingerprint values. Uses `ASE`s `spacegroup` module to compute the space group.

        **Arguments:**

        atoms: `ase.Atoms`
            ASE Atoms object that contains (periodic) structure to determine the symmetry operations.

        **Keyword arguments:**

        symprec: `float`
            Precision, i.e. tolerance, for atom positions

            default: `0.1`

        unify_species: `bool`
            Set all species to the same value before computing symmetry operations.
            
            default: `False`
        """
        atoms = atoms.copy()
        if unify_species:
            atoms.set_atomic_numbers([1 for _ in atoms.get_atomic_numbers()])
        sg_ = get_spacegroup(atoms, symprec=symprec)
        self.set_data("SGN", sg_.no)
        self.set_data("symops", list(map(self._symop_to_string, sg_.get_symop())))
        return self

    def _symop_to_string(symop):
        rot = symop[0].flatten().tolist()
        rot.extend(symop[1])
        return ":".join(map(str, rot))
