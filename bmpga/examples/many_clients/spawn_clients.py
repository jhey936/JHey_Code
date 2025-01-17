import logging

from multiprocessing import Process

from bmpga.potentials import LJcPotential
from bmpga.optimisation import QuenchClient


def spawn_client(log):


    
    LJ_pot = LJcPotential()
    
    with open("example.uri") as f:
        uri = f.read().strip()

    quencher = QuenchClient(potential=LJ_pot, URI=uri, max_quenches=250, log=log)

    quencher.run()


if __name__ == "__main__":

    log = logging.getLogger(__name__)

    processes = []
    for proc in range(2):
        p = Process(target=spawn_client, args=(log, ))
        p.start()

    for p in processes:
        p.join()
        p.terminate()
    

    
