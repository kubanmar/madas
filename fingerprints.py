from DOS_fingerprints import DOSFingerprint, Grid


class Fingerprint():

    def __init__(self, fp_type, properties = None, parameters = None, data = None):
        self.properties = properties
        self.fp_type = fp_type
        if data == None:
            self.calculate()
        else:
            self.data = data
            self._reconstruct_from_data()
        #TODO Catch: neither data nor properties

    def calculate(self):
        if self.fp_type == 'DOS':
            json_data = self.properties['dos']
            cell_volume = self.properties['cell_volume']
            grid = Grid.create()
            self.fingerprint = DOSFingerprint(json_data, cell_volume, grid)
            self.data = self.fingerprint.get_data()
            #print(self.data) #DEBUG
        return self.data

    def calc_similiarity(self, mid, database):
        if self.fp_type == 'DOS':
            if not hasattr(self, 'grid'):
                grid = Grid.create(id = self.data['grid_id'])
            fingerprint = database.get_fingerprint(mid, 'DOS')
            if fingerprint.fingerprint.grid_id != self.fingerprint.grid_id:
                sys.exit('Error: DOS grid types to not agree.') # TODO This is by far no nice solution.
            return grid.tanimoto(self.fingerprint, fingerprint.fingerprint)

    def _reconstruct_from_data(self):
        if self.fp_type == 'DOS':
            self.fingerprint = DOSFingerprint(None, None, None)
            self.fingerprint.bins = bytes.fromhex(self.data['bins'])
            self.fingerprint.indices = self.data['indices']
            self.fingerprint.grid_id = self.data['grid_id']
        return self.data
