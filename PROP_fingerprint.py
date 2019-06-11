from fingerprint import Fingerprint
from utils import get_lattice_parameters_from_string, report_error

class PROPFingerprint(Fingerprint):

    #def __init__(self, properties, db_data = None, properties_names = ['atomic_density', 'mass_density']): # 'cell_volume', 'lattice_parameters',
    def __init__(self, db_row = None, properties_names = ['atomic_density', 'mass_density']): # 'cell_volume', 'lattice_parameters',
        self.properties_names = properties_names
        self._init_from_db_row(db_row)

    def calculate(self, db_row):
        properties = db_row['data']
        fingerprint_data = {}
        for prop in self.properties_names:
            try:
                fingerprint_data[prop] = properties[prop]
            except KeyError:
                if not hasattr(self, 'log'):
                    self.log = None
                mid = 'unknown' if not hasattr(self, 'mid') else self.mid
                report_error(self.log, 'No property of type %s for material %s' %(prop, mid))
        self.data = fingerprint_data

    def reconstruct(self, db_row):
        data = self._data_from_db_row(db_row)
        self.data = data

    def get_data(self):
        return self.data

def PROP_similarity(prop_fingerprint1, prop_fingerprint2):
    property_names = [x for x in prop_fingerprint1.data.keys()]
    similarity = 0
    n_properties = 0
    for prop_name in property_names:
        fp1_prop = prop_fingerprint1.data[prop_name]
        try:
            fp2_prop = prop_fingerprint2.data[prop_name]
            n_properties += 1
        except KeyError:
            report_error(None, 'Can not calculate similarity of property %s for materials %s and %s. Not available for %s.' %(prop_name, prop_fingerprint1.mid, prop_fingerprint2.mid, prop_fingerprint2.mid))
        if prop_name == 'lattice_parameters':
            overlap = 0
            lcs1 = get_lattice_parameters_from_string(fp1_prop)
            lcs2 = get_lattice_parameters_from_string(fp2_prop)
            for lps in zip(lcs1, lcs2):
                overlap += min(lps)/max(lps)
            overlap = overlap / 6
            similarity += overlap
        else:
            prop_pair = [float(fp1_prop), float(fp2_prop)]
            similarity += min(prop_pair) / max(prop_pair)
    similarity = similarity / n_properties
    return similarity
