from DOS_fingerprints import DOSFingerprint, Grid
from SYM_fingerprints import SYMFingerprint, get_SYM_sim
from SOAP_fingerprint import SOAPfingerprint
import json

class Fingerprint():

    def __init__(self, fp_type, properties = None, atoms = None, db_row = None):
        self.properties = properties
        self.fp_type = fp_type
        self.atoms = atoms
        if db_row == None:
            self.calculate()
        else:
            self.data = self._get_db_data(db_row)
            self._reconstruct_from_data()
        #TODO Catch: neither data nor properties

    def calculate(self):
        if self.fp_type == 'DOS':
            json_data = self.properties['dos']
            cell_volume = self.properties['cell_volume']
            grid = Grid.create()
            self.fingerprint = DOSFingerprint(json_data, cell_volume, grid)
        elif self.fp_type == "SYM":
            self.fingerprint = SYMFingerprint(self.atoms)
        elif self.fp_type == "SOAP":
            self.fingerprint = SOAPfingerprint(self.atoms)
        self.data = self.fingerprint.get_data()
        return self.data

    def write_to_database(self, row_id, database):
        data = json.dumps(self.calculate())
        if self.fp_type == 'DOS':
            database.update(row_id, DOS = data)
        elif self.fp_type == 'SYM':
            database.update(row_id, SYM = data)
        elif self.fp_type == 'SOAP':
            database.update(row_id, SOAP = data)

    def _get_db_data(self, row):
        if self.fp_type == "DOS":
            data = json.loads(row.DOS)
        elif self.fp_type == "SYM":
            data = json.loads(row.SYM)
        return data

    def calc_similiarity(self, mid, database):
        if self.fp_type == 'DOS':
            if not hasattr(self, 'grid'):
                grid = Grid.create(id = self.data['grid_id'])
            fingerprint = database.get_fingerprint(mid, 'DOS')
            if fingerprint.fingerprint.grid_id != self.fingerprint.grid_id:
                sys.exit('Error: DOS grid types to not agree.') # TODO This is by far no nice solution.
            return grid.tanimoto(self.fingerprint, fingerprint.fingerprint)
        if self.fp_type == 'SYM':
            fingerprint = database.get_fingerprint(mid, 'SYM')
            return get_SYM_sim(self.fingerprint.symop, fingerprint.fingerprint.symop) #, self.fingerprint.sg, fingerprint.fingerprint.sg

    def _reconstruct_from_data(self):
        if self.fp_type == 'DOS':
            self.fingerprint = DOSFingerprint(None, None, None)
            self.fingerprint.bins = bytes.fromhex(self.data['bins'])
            self.fingerprint.indices = self.data['indices']
            self.fingerprint.grid_id = self.data['grid_id']
        elif self.fp_type == 'SYM':
            self.fingerprint = SYMFingerprint(None, symop = self.data['symop'], sg = self.data['sg'])
        return self.data
