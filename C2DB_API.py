import os,sys
from ase.db import connect

class API():

    """
    API to the Coputational 2D Materials Database:
    The Computational 2D Materials Database: High-Throughput Modeling and Discovery of Atomically Thin Crystals

    Sten Haastrup, Mikkel Strange, Mohnish Pandey, Thorsten Deilmann, Per S. Schmidt, Nicki F. Hinsche, Morten N. Gjerding, Daniele Torelli, Peter M. Larsen, Anders C. Riis-Jensen, Jakob Gath, Karsten W. Jacobsen, Jens Jørgen Mortensen, Thomas Olsen, Kristian S. Thygesen

    2D Materials 5, 042002 (2018)
    """

    def __init__(self, atoms_db, logger = None):
        self.set_logger(logger)
        self.atoms_db = atoms_db

    def get_calculation(self):
        pass

    def get_calculations_by_search(self, **kwargs):
        for db_id in range(self.atoms_db.count()):
            self.atoms_db.update(db_id, mid = self.atoms_db.get(db_id).uid)
        return None

    def get_property(self):
        pass

    def set_logger(self, logger):
        self.log = logger

    def _report_error(self, error_message):
        if self.log != None:
            self.log.error(error_message)
        else:
            print(error_message)
