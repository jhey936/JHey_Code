import logging
import Pyro4

import numpy as np

from bmpga.mating.selectors import BaseSelector
from bmpga.characterization import SimpleEnergeticCharacterizer
from bmpga.storage import Database, Cluster, Molecule
from bmpga.systems import DefineSystem
from bmpga.optimisation import PoolGA
from bmpga.mutation import Mutate, RandomSingleTranslation, RandomMultipleTranslations

log = logging.getLogger(__name__)

compare = SimpleEnergeticCharacterizer()
database = Database(db="test.db",
                    new_database=True,
                    compare_clusters=SimpleEnergeticCharacterizer(accuracy=5e-7))


lj = Molecule(coordinates=np.array([[0.0, 0.0, 0.0]]), particle_names=["LJ"])
system = DefineSystem(numbers_of_molecules=[38], molecules=[lj], log=log, box_length=6)

# define a mutation scheme
# probabilities is normalised inside Mutate so we can just pass realatve probabilites
mutation = Mutate(mutations=[RandomSingleTranslation(), RandomMultipleTranslations()], relative_probabilities=[1, 1])

daemon = Pyro4.Daemon(port=9939)

GA = PoolGA(database=database,
            min_pool_size=10,
            max_generations=2,
            system=system,
            mutate=mutation,
            daemon=daemon,
            mutation_rate=0.3,
            log=log)

with open("example.uri", "w") as f:
    uri = daemon.register(GA)
    f.write(str(uri))

GA.start_threads()
daemon.requestLoop()
daemon.shutdown()
            
