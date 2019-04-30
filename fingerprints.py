from DOS_fingerprints import DOSFingerprint, Grid
from SYM_fingerprints import SYMFingerprint, get_SYM_sim
from SOAP_fingerprint import SOAPfingerprint, get_SOAP_sim
from EWM_Fingerprint import EWMFingerprint, get_EWM_sim
from PROP_Fingerprint import PROPFingerprint, get_PROP_sym
from IAD_Fingerprint import IADFingerprint, get_IAD_sim
import json, types, copy
import logging

class Fingerprint():

    def __init__(self, fp_type, mid = None, properties = None, atoms = None, db_row = None, log = True, fp_name = None, calculate = True, **kwargs): #database = None,
        self.properties = properties
        self.mid = mid
        self.fp_name = fp_type if fp_name == None else fp_name
        self.fp_type = fp_type
        self.atoms = atoms
        #if log: # The presence of a log does not allow for parallelization of the similarity.
        self.log = logging.getLogger('log') if log else None
        #else:
        #    self.log = None
        if db_row == None:
            if calculate:
                self.calculate(**kwargs)
        else:
            self.data = self._get_db_data(db_row)
            self._reconstruct_from_data()
        #TODO Catch: neither data nor properties

    def calculate(self, **kwargs):
        if self.fp_type == 'DOS':
            json_data = self.properties['dos']
            cell_volume = self.properties['cell_volume']
            grid = Grid.create(**kwargs)
            self.fingerprint = DOSFingerprint(json_data, cell_volume, grid)
        elif self.fp_type == "SYM":
            self.fingerprint = SYMFingerprint(self.atoms)
        elif self.fp_type == "SOAP":
            self.fingerprint = SOAPfingerprint(self.atoms)
        elif self.fp_type == "EWM":
            self.fingerprint = EWMFingerprint(self.atoms)
        elif self.fp_type == "PROP":
            self.fingerprint = PROPFingerprint(self.properties)
        elif self.fp_type == 'IAD':
            self.fingerprint = IADFingerprint(self.atoms)
        self.data = self.fingerprint.get_data()
        return self.data

    def get_data_json(self):
        if not hasattr(self, 'data'):
            self.log.error('Reqested data for material '+mid+', but data is not yet calculated.')
            self.data = self.calculate()
        data = json.dumps(self.data)
        return data

    def _get_db_data(self, row):
        data = json.loads(row[self.fp_name])
        return data

    def set_similarity_function(self, function):
        #import types
        #self.get_similarity = types.MethodType(function, self)
        setattr(self, function.__name__, function) #TODO understand that better!
        setattr(self, 'get_similarity', types.MethodType(function, self))

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
                    if self.log != None:
                        self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
            elif s == 'earth_mover':
                try:
                    similarity = self.grid.earth_mover_similarity(self.fingerprint, fingerprint.fingerprint)
                except ZeroDivisionError:
                    similarity = 0
                    if self.log != None:
                        self.log.error('ZeroDivisionError for '+str(self.mid)+' and '+str(fingerprint.mid))
            elif s == 'mutual_information':
                similarity = self.grid.mutual_information(self.fingerprint, fingerprint.fingerprint)
            return  similarity
        elif self.fp_type == 'SYM':
            return get_SYM_sim(self.fingerprint.symop, fingerprint.fingerprint.symop) #, self.fingerprint.sg, fingerprint.fingerprint.sg
        elif self.fp_type == 'PROP':
            return get_PROP_sym(self.fingerprint, fingerprint.fingerprint)
        elif self.fp_type == 'IAD':
            return get_IAD_sim(self.data, fingerprint.data)
        elif self.fp_type == "SOAP":
            return get_SOAP_sim(self.fingerprint, fingerprint.fingerprint)
        elif self.fp_type == "EWM":
            return get_EWM_sim(self.fingerprint, fingerprint.fingerprint)

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
        elif self.fp_type == 'EWM':
            self.fingerprint = EWMFingerprint(None, self.data)
        elif self.fp_type == 'PROP':
            self.fingerprint = PROPFingerprint(None, self.data)
        elif self.fp_type == 'IAD':
            self.fingerprint = IADFingerprint(None,self.data)
        return self.data
