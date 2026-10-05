import time

from TractoPL.data.loader import Dataset

t0 = time.time()
ds = Dataset("/home/ndecaux/NAS_EMPENN/share/projects/actidep/bids")
t1 = time.time()

print(f"Dataset initialized in {t1 - t0:.2f} seconds.")
print(len(ds.get_global(extension="trk")))