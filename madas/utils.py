from typing import Any
import numpy as np
import sys
from json import JSONEncoder
import logging, json  # noqa: E401

# check which progress bar to use, from: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
is_jupyter = False
try:
    shell = get_ipython().__class__.__name__  # type:ignore
    if shell == 'ZMQInteractiveShell':
        is_jupyter = True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
        is_jupyter = False  # Terminal running IPython
    else:
        is_jupyter = False  # Other type (?)
except NameError:
    is_jupyter = False      # Probably standard Python interpreter

if is_jupyter:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm  # noqa: F401

species_list = 'Vac,H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Nh,Fl,Mc,Lv,Ts,Og'.split(',')

electron_charge = 1.602176565e-19

def rmsle(y_true, y_pred):
    """
    Root means square logarithmic error as used in the NOMAD kaggle competition:

    Sutton, C., Ghiringhelli, L.M., Yamamoto, T. et al. 
    Crowd-sourcing materials-science challenges with the NOMAD 2018 Kaggle competition. 
    npj Comput Mater 5, 111 (2019). https://doi.org/10.1038/s41524-019-0239-3
    
    **Arguments:**

    y_true: `List[float]`
        List of true target values

    y_pred: `List[float]`
        List of predicted target values

    **Returns:**

    rmsle: `float`
        Root mean squared logarithmic error
    """
    if not isinstance(y_true, (list, np.ndarray)):
        y_true = [y_true]
    if not isinstance(y_pred, (list, np.ndarray)):
        y_pred = [y_pred]
    errors = [np.log((np.array(y_p) + 1)/(np.array(y_t) + 1))**2 for y_t, y_p in zip(y_true, y_pred)]
    msle = sum(errors) / len(errors)
    return np.sqrt(msle).tolist()

#def _SI_to_Angstom(length):
#    return np.power(length,10^10)

def report_error(logger: logging.Logger, error_message: str):
    """
    Report error by writing it to a `logging.Logger` instance or to `sys.stderr`.

    **Arguments:**

    logger: `logging.Logger` or `None`
        Log target. Write to log or to stderr if `logger == None`
    
    error_message: `str`
        Message to write

    **Returns:**

    `None`
    """
    if logger is not None:
        logger.error(error_message)
    else:
        print(error_message, file=sys.stderr)

def safe_log(message: str,
             logger: logging.Logger = None, 
             level: str = "error"):    
    """
    Report error by writing it to a `logging.Logger` instance or to `sys.stderr`.

    **Arguments:**
    
    error_message: `str`
        Message to write

    **Keyword arguments:**

    logger: `logging.Logger` or `None`
        Log target. Write to log or to stderr if `logger == None`

        default: `None`

    level: `str`
        Choose target of log. Write to logger.info, logger.error, or, stdout, stderr.

        Options: "error", "info"

        default, and fallback, is "error"

    **Returns:**

    `None`
    """
    if logger is not None:
        if level == "info":
            logger.info(message)
        else:
            logger.error(message)
    else:
        out = sys.stdout if level == "info" else sys.stderr
        print(message, file = out)

def seed_random_number_generators(random_seed):
    """
    Seed Python standard library random generator and numpy random generator.

    **Arguments:**

    random_seed: `Any`
        Seed passed to random number generators.
    """
    from random import seed
    seed(random_seed)
    np.random.seed(random_seed)

def resolve_nested_dict(archive: dict, 
                        path:str, 
                        error_message="failed to resolve path", 
                        fail_on_key_error=False) -> Any:
    """
    Given a nested dictionary (including lists), return the entry at `path` in the dictionary.
    
    **Arguments:**
    
    archive: *Dict[Dict, List]*
        Nested dictionary with str or int keys
        
        Example: `{'a' : {'b' : [{'c':5}]}, 'd' : 'data'}`
        
    path: *str*
        Keys to navigate the dictionary given as `archive`, separated by a '/'
    
        Example: `'a/b/0/c'` # return value `5`
        
    **Keyword arguments:**
    
    error_message: *str*
        Message to display if the path can not be resolved
        
        default: `"failed to resolve path"`
        
    fail_on_key_error: *bool*
        If the path can not be resolved, raise the Exception insted of returning `None`.
        
    **Returns:**
    
    archive: *Any*
        Value at specified location in the dictionary.
        `None`, if the path can not resolved.
        
    **Raises:**
    
    `KeyError`
        Path can not be resolved and `fail_on_key_error==True`
    """
    try:
        for key in path.strip("/").split("/"):
            try:
                key_int = int(key)    
                archive = archive[key_int]
            except ValueError:
                archive = archive[key]
        return archive
    except KeyError as e:
        print(error_message, file=sys.stderr)
        if fail_on_key_error:
            raise e
        else:
            return None

def list_chunks(long_list, chunk_length = 5):
    chunked_list = []
    chunk = []
    for index,item in enumerate(long_list):
        chunk.append(item)
        if index % chunk_length == 0:
            chunked_list.append(chunk)
            chunk = []
    if chunk != []:
        chunked_list.append(chunk)
    return chunked_list

def merge_k_nearest_neighbor_dicts(fp_type_list, dicts):
    merged_dict = {}
    for index, fp_type in enumerate(fp_type_list):
        materials = [key for key in dicts[index].keys()]
        for material in materials:
            data = dicts[index][material]
            if material not in merged_dict.keys():
                merged_dict[material] = {mid:{fp_type:sim} for mid, sim in data}
            else:
                for mid, sim in data:
                    if mid not in merged_dict[material].keys():
                        merged_dict[material][mid] = {fp_type:sim}
                    else:
                        merged_dict[material][mid].update({fp_type:sim})
    return merged_dict


class Float32ToJson(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj,np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def contour_data_from_list(coord_list):
    xs = np.unique([x[0] for x in coord_list])
    ys = np.unique([x[1] for x in coord_list])
    z_matrix = []
    for y in ys:
        row = []
        for x in xs:
            results = [entry for entry in coord_list if (entry[0] == x).all() and (entry[1] == y).all()]
            if len(results) > 1:
                print(results)
                raise AssertionError('Double counting of coordinates.')
            row.append(results[0][2])
        z_matrix.append(row)
    return np.array(z_matrix), xs, ys

class JSONNumpyEncoder(JSONEncoder):
    
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            o=o.tolist()
        return super().default(o)

class BatchIterator():
    """
    A helper class for paralellization of calculation of large batched similarity matrices.

    A `batch` is defined here as a nested list of integers describing the indices of a sub-matrix.

    To parallelize the calculation of large matrices of different nodes in a compute cluster, 
    each node may compute several sub-matrices that can be combined to the complete matrix.

    To reduce the memory footprint, each node loads only a subset of the required fingerprints
    and calculates the (overlap) similarity matrix from these.

    This class iteratively provides the mapping of fingerprint indices in a given list.
    Thus, upon iteration over an object of this class, all combinations of indices that are 
    required to compute a unique sub-matrix for a specific task id are returned.
    """
    
    def __init__(self, 
                 n_entries: int, 
                 batch_size: int, 
                 n_tasks: int = 1, 
                 task_id: int = 0,
                 symmetric: bool = True):
        """
        **Arguments:**

        n_entries: `int`
            Total number of indices to consider
        
        batch_size: `int`
            Range of indices for each batch

        **Keyword arguments:**

        n_tasks: `int`
            Total number of tasks executed in parallel

            default: `1`

        task_id: `int`
            Integer number that specifies which tasks this instance is doing

            Must be smaller than `n_tasks`

            default: `0`

        symmetric: `bool`
            Return only symmetrically unique batches

            default: `True`
        """
        if task_id >= n_tasks:
            raise ValueError("Task ID must be smaller than number of tasks!")
        self.n_tasks = n_tasks
        self.task_id = task_id
        self.symmetric = symmetric
        self._n_entries = n_entries
        self._batch_size = batch_size
        self._batches = self._gen_batches(n_entries, batch_size)
        self._iter_index = 0

    @property
    def batches(self):
        """
        _All_ batches, regardless of the task id.
        """
        return self._batches
    
    @property
    def n_entries(self):
        """
        Total number of entries, i.e. the size of the range of integer to consider.
        """
        return self._n_entries
    
    @property
    def batch_size(self):
        """
        Range of integers in a batch.
        """
        return self._batch_size
    
    def plot_batches(self, figure = True, show = True, text_fontsize=10):
        """
        Generate a plot of the batches for visualization.

        **Keyword arguments:**

        figure: `bool`
            Create new matplotlib.pyplot.figure

            default: `True`

        show: `bool`
            show plot

            default: `True`
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if figure:
            plt.figure(figsize = (10,10))
        plt.xlim(0,self.n_entries)
        plt.ylim(self.n_entries,0)
        for batch in self:
            plt.fill_between([batch[0][0], batch[0][1]],[batch[1][1], batch[1][1]], [batch[1][0], batch[1][0]], color = cm.get_cmap("cool")(self.task_id / self.n_tasks))
            plt.text(sum(batch[0])/2, sum(batch[1])/2, f"{batch}\n{self.task_id}", horizontalalignment='center', fontsize=text_fontsize)
        if show:
            plt.show()
            
    def plot_batch_rows(self, figure = True, show = True, text_fontsize=10):
        """
        Generate a plot of the batches in a row for visualization.

        **Keyword arguments:**

        figure: `bool`
            Create new matplotlib.pyplot.figure

            default: `True`

        show: `bool`
            show plot

            default: `True`
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        if figure:
            plt.figure(figsize = (10,10))
        plt.xlim(0,self.n_entries)
        plt.ylim(self.n_entries,0)
        for idx, batch_row in enumerate(self.get_batch_rows()):
            for batch in batch_row:
                plt.fill_between([batch[0][0], batch[0][1]],[batch[1][1], batch[1][1]], [batch[1][0], batch[1][0]], color = cm.get_cmap("tab20")(idx), alpha = 0.5)
                plt.text(sum(batch[0])/2, sum(batch[1])/2, f"{batch}\n{self.task_id}", horizontalalignment='center', fontsize=text_fontsize)
        if show:
            plt.show()
    
    def get_batches_for_index(self, index: int):
        """
        Return all batches that contain a given index.

        **Arguments:**

        index: `int`
            Index to that should be contained in returned batches.

        **Returns:**

        batches: `List[List[List[int]]]`
            batches containing the given index
        """
        batches = []
        for batch in self.batches:
            if self.symmetric:
                if batch[0][0] <= index < batch[0][1] or batch[1][0] <= index < batch[1][1]:
                    batches.append(batch)
            else:
                if batch[1][0] <= index < batch[1][1]:
                    batches.append(batch)
        return batches
    
    def get_batch_rows(self):
        """
        This is sensitive to the task index!
        Get all rows of batches.
        """
        batch_row_start_indices = list(range(0, self.n_entries, self.batch_size))
        batch_rows_for_task_id = []
        for idx, index in enumerate(batch_row_start_indices):
            if idx % self.n_tasks == self.task_id:
                batch_rows_for_task_id.append(self.get_batches_for_index(index))
        return batch_rows_for_task_id
    
    def _gen_batches(self, size, batch_size):
        batch_list = []
        batch_x_list = self.linear_batch_list(size, batch_size)
        if self.symmetric:
            for idx, batch_x in enumerate(batch_x_list):
                for batch_y in batch_x_list[idx:]:
                    batch_list.append([batch_x, batch_y])
        else:
            for batch_x in batch_x_list:
                for batch_y in batch_x_list:
                    batch_list.append([batch_x, batch_y])            
        return batch_list
    
    @staticmethod
    def linear_batch_list(size, batch_size):
        """
        List of indices that splits a list of size `size` into lists of length `batch_size`.
        """
        len_batched = int(size / batch_size)
        if len_batched * batch_size < size:
            len_batched += 1
        batch_x_list = []
        batch_index = 0
        for _ in range(len_batched):
            if batch_index + batch_size > size:
                batch_x_list.append([batch_index, size])
                break
            else:
                batch_x_list.append([batch_index, batch_index + batch_size])
                batch_index += batch_size
        return batch_x_list
    
    def __len__(self):
        return len(self._batches)
    
    def __iter__(self):
        self._iter_index = 0
        return self
    
    def __next__(self):
        while self._iter_index % self.n_tasks != self.task_id:
            self._iter_index += 1
        if self._iter_index < len(self):
            item = self._batches[self._iter_index]
            self._iter_index += 1
            return item
        else:
            raise StopIteration
            
    def __getitem__(self, index):
        return self.batches[index]

def print_dict_tree(dict_: dict, indent: int=0, indent_symbol: str=" ") -> None:
    """
    Recursively print keys of a dictionary.
    """
    if not isinstance(dict_, dict):
        raise TypeError(f"Please provide a `dict`, type of input is {type(dict_)}")
    for key, values in dict_.items():
        if isinstance(values, dict):
            print(f'{int(indent) * indent_symbol} {key} ' + "{")
            print_dict_tree(values, indent=indent+2, indent_symbol=indent_symbol)
            print(f'{int(indent) * indent_symbol}' + "}")
        elif isinstance(values, list):
            if sum(1 for  _ in values if isinstance(_, dict)) == 0:
                print(f'{int(indent) * indent_symbol} {key} : <value>')
                continue
            print(f'{int(indent) * indent_symbol} {key} [ ')
            has_values = False
            for index, item in enumerate(values):
                if isinstance(item, dict):
                    print(f'{int(indent) * indent_symbol} [{index}] ' + "{")
                    print_dict_tree(item, indent=indent+2, indent_symbol=indent_symbol)
                    print(f'{int(indent) * indent_symbol}' + "}")
                else:
                    if not has_values:
                        print(f'{int(indent+2) * indent_symbol} <value>')
                        has_values = True
                    else:
                        continue
            print(f'{int(indent) * indent_symbol}]')
        else:
            print(f'{int(indent) * indent_symbol} {key} : <value>')
            
def print_key_paths(key_name: str, 
                    dictionary: dict, 
                    parent_path: str="", 
                    child_of: str | None = None) -> None:
    """
    Iterate recursively trough a dict and print all paths that end with a
    given key name.

    **Arguments:**

    key_name: `str`
        key for to search path for

    dictionary: `dict`
        (Nested) dictionary in which the key path is searched.

    **Keyword arguments:**

    parent_path: `str`
        path until current iteration

        default: ""

    child_of: `str` or `None`
        if not `None`, print only paths that contain the string specified here

        default: `None`
    """
    if not isinstance(dictionary, dict):
        raise TypeError(f"Please provide a `dict`, type of input is {type(dictionary)}")
    for key, values in dictionary.items():
        if key == key_name:
            if child_of is not None:
                if child_of in parent_path:
                    print(f"{parent_path}/{key_name}")
            else:
                print(f"{parent_path}/{key_name}")
        if isinstance(values, dict):
            print_key_paths(key_name, values, parent_path=f"{parent_path}/{key}", child_of=child_of)
        elif isinstance(values, list):
            if sum(1 for  _ in values if isinstance(_, dict)) == 0:
                continue
            has_values = False
            for index, item in enumerate(values):
                if isinstance(item, dict):
                    print_key_paths(key_name, item, parent_path=f"{parent_path}/{key}/{index}", child_of=child_of)
                else:
                    if not has_values:
                        has_values = True
                    else:
                        continue    