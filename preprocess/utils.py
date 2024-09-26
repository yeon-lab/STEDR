import pickle

def my_dump(obj, file_name):
    pickle.dump(obj, open(file_name, 'wb'))
    print(f'dumped {file_name}...', flush=True)
    
    
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
