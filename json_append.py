import json,sys
from os.path import exists

def _seek_terminating_char(f, char = '}'):
    for offset in range(1,11):
        byteoffset = offset * (-1)
        f.seek(byteoffset, 2)
        data = f.read()
        if bytes(char,'utf-8') in data:
            f.seek(byteoffset, 2)
            return
    sys.exit("No valid char found.")

def write_json_file(json_data, filename):
    if not exists(filename):
        with open(filename,'w') as f:
            json.dump(json_data, f, indent = 4)
    else:
        with open(filename, 'rb+') as f:
            _seek_terminating_char(f)
            f.write(bytes(',','utf-8'))
            f.write(bytes(json.dumps(json_data, indent = 4),'utf-8')[1:])
