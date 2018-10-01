from DOS_fingerprints import DOSFingerprint, Grid


class Fingerprint():

    def __init__(self, fp_type, properties = None, parameters = None, data = None):
        self.properties = properties
        self.fp_type = fp_type
        if data == None:
            self.calculate()
        else:
            self.reconstruct_from_data()
        #TODO Catch: neither data nor properties

    def dump_data(self):
        """
        Is meant to return all the data that is required to construct the fingerprint again.
        """
        return self.fp_type, self.data #This is not enough, apparently.

    def _reconstruct_from_data(self):
        pass

    def calculate(self):
        if self.fp_type == 'DOS':
            json_data = self.properties['dos']
            cell_volume = self.properties['cell_volume']
            grid = Grid.create()
            fingerprint = DOSFingerprint(json_data, cell_volume, grid)
            self.data = fingerprint.get_data()
        return self.data
