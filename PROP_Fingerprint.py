import logging
from utils import get_lattice_parameters_from_string

class PROPFingerprint():

    def __init__(self, properties, db_data = None, properties_names = ['atomic_density', 'cell_volume', 'lattice_parameters', 'mass_density']):
        if db_data != None:
            self.data = db_data
            return
        data = properties
        fingerprint_data = {}
        for prop in properties_names:
            try:
                fingerprint_data[prop] = data[prop]
            except KeyError:
                pass
        self.data = fingerprint_data

    def get_data(self):
        return self.data

def get_PROP_sym(prop_fingerprint1, prop_fingerprint2):
    log = logging.getLogger('log')
    property_names = [x for x in prop_fingerprint1.data.keys()]
    similarity = 0
    n_properties = 0
    for prop_name in property_names:
        fp1_prop = prop_fingerprint1.data[prop_name]
        try:
            fp2_prop = prop_fingerprint2.data[prop_name]
            n_properties += 1
        except KeyError:
            log.error('Can not calculate similarity of property %s for materials %s and %s. Not available for %s.' %(prop_name, prop_fingerprint1.mid, prop_fingerprint2.mid, prop_fingerprint2.mid))
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
