from DOS_fingerprints import DOSFingerprint, Grid
from SYM_fingerprints import SYMFingerprint, get_SYM_sim
from SOAP_fingerprint import SOAPfingerprint, get_SOAP_sim
import json
import logging

class Fingerprint():

    def __init__(self, fp_type, mid = None, properties = None, atoms = None, db_row = None, database = None):
        self.properties = properties
        self.mid = mid
        self.fp_type = fp_type
        self.atoms = atoms
        self.log = logging.getLogger('log')
        if db_row == None:
            self.calculate()
        else:
            self.data = self._get_db_data(db_row)
            self._reconstruct_from_data()
        if database != None:
            self.database = database
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

    def get_data_json(self):
        if not hasattr(self, 'data'):
            self.log.error('Reqested data for material '+mid+', but data is not yet calculated.')
            self.data = self.calculate()
        data = json.dumps(self.data)
        return data

    def write_to_database(self, row_id, database):
        data = self.get_data_json()
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
        elif self.fp_type == 'SOAP':
            data = json.loads(row.SOAP)
        return data

    def calc_similiarity(self, mid, database): #TODO Outdated
        if self.fp_type == 'DOS':
            if not hasattr(self, 'grid'):
                self.grid = Grid.create(id = self.data['grid_id'])
            fingerprint = database.get_fingerprint(mid, 'DOS')
            if fingerprint.fingerprint.grid_id != self.fingerprint.grid_id:
                sys.exit('Error: DOS grid types to not agree.') # TODO This is by far no nice solution.
            return self.grid.tanimoto(self.fingerprint, fingerprint.fingerprint)
        if self.fp_type == 'SYM':
            fingerprint = database.get_fingerprint(mid, 'SYM')
            return get_SYM_sim(self.fingerprint.symop, fingerprint.fingerprint.symop) #, self.fingerprint.sg, fingerprint.fingerprint.sg
        if self.fp_type == 'SOAP':
            fingerprint = database.get_fingerprint(mid, 'SOAP')
            return get_SOAP_sim(self.data, fingerprint.data)

    def calc_similiarity_multiprocess(self, mid): #TODO Outdated
        if self.fp_type == 'DOS':
            if not hasattr(self, 'grid'):
                self.grid = Grid.create(id = self.data['grid_id'])
            fingerprint = self.database.get_fingerprint(mid, 'DOS')
            if fingerprint.fingerprint.grid_id != self.fingerprint.grid_id:
                sys.exit('Error: DOS grid types to not agree.') # TODO This is by far no nice solution.
            return self.grid.tanimoto(self.fingerprint, fingerprint.fingerprint)
        if self.fp_type == 'SYM':
            fingerprint = self.database.get_fingerprint(mid, 'SYM')
            return get_SYM_sim(self.fingerprint.symop, fingerprint.fingerprint.symop) #, self.fingerprint.sg, fingerprint.fingerprint.sg

    def get_similarity(self, fingerprint, s = 'tanimoto'):
        if self.fp_type == 'DOS':
            if not hasattr(self, 'grid'):
                self.grid = Grid.create(id = self.data['grid_id'])
            if fingerprint.fingerprint.grid_id != self.fingerprint.grid_id:
                sys.exit('Error: DOS grid types to not agree.') # TODO This is by far no nice solution.
            if s == 'tanimoto':
                try:
                    similarity = self.grid.tanimoto(self.fingerprint, fingerprint.fingerprint)
                except ZeroDivisionError:
                    similarity = 0
                    self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
            elif s == 'earth_mover':
                try:
                    similarity = self.grid.earth_mover_similarity(self.fingerprint, fingerprint.fingerprint)
                except ZeroDivisionError:
                    similarity = 0
                    self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
            return  similarity
        if self.fp_type == 'SYM':
            return get_SYM_sim(self.fingerprint.symop, fingerprint.fingerprint.symop) #, self.fingerprint.sg, fingerprint.fingerprint.sg

    def _reconstruct_from_data(self):
        if self.fp_type == 'DOS':
            self.fingerprint = DOSFingerprint(None, None, None)
            self.fingerprint.bins = bytes.fromhex(self.data['bins'])
            self.fingerprint.indices = self.data['indices']
            self.fingerprint.grid_id = self.data['grid_id']
        elif self.fp_type == 'SYM':
            self.fingerprint = SYMFingerprint(None, symop = self.data['symop'], sg = self.data['sg'])
        elif self.fp_type == 'SOAP':
            self.fingerprint = SOAPfingerprint(None, self.data)
        return self.data
