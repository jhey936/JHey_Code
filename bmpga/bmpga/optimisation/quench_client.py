# coding=utf-8
"""
Provides the QuenchClient class which is a client of the PoolGAServer class

The quench client can be initialised using either a Pyro4 URI or a Pyro4 name_server
(see: https://pythonhosted.org/Pyro4/tutorials.html for details)

Once the QuenchClient is connected to the GA server, it will start requesting jobs.
These can be either local minimisations or single point energy calculations for Cluster objects.

These jobs will be completed and pass updated Cluster objects back to the server for storage.
(These will be the original Cluster objects with updated coordinates and cost attributes)

"""
import os
import uuid
import Pyro4
import logging

from time import sleep
from base64 import b64decode
from pickle import loads, dumps
from typing import Callable, Union

from Pyro4.errors import ConnectionClosedError

from bmpga.storage import Cluster
from bmpga.errors import ServerNotFoundError, InvalidURIError, ClientError

from bmpga.optimisation.base_quencher import BaseQuencher
from bmpga.potentials.base_potential import BasePotential


# noinspection PyPep8Naming
class QuenchClient(BaseQuencher):
    """
    Provides an interface to the potential that can be run in parallel.
    As the main GA process is running as a
    """

    def __init__(self, potential: BasePotential, max_quenches: int=10,
                 URI: Union[str, bool]=False, name_server: Union[str, bool]=False,
                 log: Callable=False, client_id: int=False,
                 *args, **kwargs) -> None:

        self.potential = potential

        self.max_quenches = max_quenches

        self.log = log or logging.getLogger(__name__)
        self.id = client_id or uuid.uuid4().int
        self.log.debug(self)
        self.log.debug("Initialising: {}\nGA URI: {}".format(self, URI))

        self.GA = self.get_GA_server(URI, name_server)

        super().__init__(potential, *args, **kwargs)

    def get_GA_server(self, URI: str=False, name_server: str=False) -> Pyro4.Proxy:
        """Sets up the interface to the GA server

        Must provide either the direct URI of the GA server instance, a file containing the URI,
           or the location of the Pyro4 name_server.

        Args:
            URI: optional
            name_server:

        Returns:
            Pyro4.Proxy object pointing to the GA

        """
        try:
            assert URI or name_server
        except AssertionError:
            message = "No URI or name_server provided to QuenchClient {}".format(self)
            self.log.exception(message)
            raise 

        t = 0.5
        while t <= 5:

            if URI:
                if os.path.exists(URI):
                    URI = self.uri_from_file(URI)
        
                try:
                        ga = Pyro4.Proxy(URI)
                        # noinspection PyProtectedMember
                        ga._pyroBind()
                        return ga

                except Pyro4.errors.PyroError as error:
                    pyroerror = error
                    t = self.wait(t)
                    
                except TypeError as error:
                    pyroerror = error
                    t = self.wait(t)
                #
                # except CommunicationError as error:
                #     pyroerror = error
                #     t = self.wait(t)
                #
                # except ConnectionClosedError as error:
                #     pyroerror = error
                #     t = self.wait(t)
                
            # elif name_server:  # TODO: implement Pyro4 name_server?
            #     return Pyro4.Proxy(name_server)
        try:
            raise ServerNotFoundError(f"The GA server cannot be reached! Pyro error: {pyroerror}")

        except ServerNotFoundError as error:
            self.log.exception(error)
    
    def wait(self, t) -> float:
        """Implements an exponential back-off to give time for the server to sort itself out"""
        t *= 1.1
        self.log.info(f"Server not found. Waiting {t} seconds to attempt reconnection.")
        sleep(t)
        return t

    def uri_from_file(self, fn: str) -> str:  # TODO: write a test for uri_from_file
        """Returns a uri which has been written to a file

        Args:
            fn: str, required, File name

        Returns:
            str, URI of the GA server

        """
        with open(fn) as f:
            uri = f.read().strip()

        try:             
            assert "PYRO" in uri
        except AssertionError:
            try:
                raise InvalidURIError(f"The URI found in {fn} is not valid! {uri}")
            except InvalidURIError as error:
                self.log.exception(error)
        else:
            return uri

    def minimize(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Provides the interface between the client and the minimise method of the

        Args:
            cluster:

        Returns:

        """
        # noinspection PyUnresolvedReferences
        result_cluster = self.potential.minimize(cluster=cluster)
        return result_cluster

    def get_energy(self, cluster: Cluster, *args, **kwargs) -> Cluster:
        """Provides the interface between the client and the minimise method of the

        Args:
            cluster: Cluster object, required, the Cluster to be minimised

        Returns:

        """
        self.log.debug("Getting energy for cluster: {}".format(cluster))

        # noinspection PyUnresolvedReferences
        result = self.potential.get_energy(cluster)
        cluster.cost = result
        return cluster

    def run(self) -> None:
        """The main loop for the quencher.

        Makes calls out to the GA server to get clusters to minimize or perform single point energy evaluations on,
           and passes back the updated cluster instances to be inserted into the population.

        This is accomplished by invoking the get_job() and return_result() methods of the GA server.

        There is a bit of trickery needed with pickle and encoding/decoding to/from base64 to get
           the Cluster objects to be sent over a socket by the server. It should all be fairly
           self explanatory below.

        Returns:
            None

        """

        self.log.info("Starting: {}".format(self))
        quench = 0

        while quench <= self.max_quenches:
            t = 0.1

            try:
                job = self.GA.get_job(self.id)
            except ConnectionClosedError:
                self.log.info("Connection closed by server")
                break
            except Pyro4.errors.SerializeError:
                self.log.info("""Received empty queue object instead of job. 
                Try increasing the max queue length of the server""")
                continue

            self.log.debug(f"{self}: beginning job {quench}")

            # Decode the serialised job
            job = loads(b64decode(job["data"]))

            input_cluster = job[1]

            self.log.debug(f"""Quench process: {self} received job number {quench}: 
            {job[0]} {input_cluster} from GA server\n""")

            if job[0] == "minimize":
                try:
                    result_cluster = self.minimize(input_cluster)

                except TypeError as error:
                    self.log.error(f"Encountered TypeError: {error}\n during minimisation of job: {job}")
                    continue
                except Exception as error:
                    self.log.error(f"Encountered unexpected error: {error} during minimisation of job {job}")
                    continue

            elif job[0] == "energy":
                try:
                    result_cluster = self.get_energy(input_cluster)

                except TypeError as error:
                    self.log.error(f"Encountered TypeError: {error}\n during minimisation of job: {job}")
                    continue
                except Exception as error:
                    self.log.error(f"Encountered unexpected error: {error} during minimisation of job {job}")
                    continue

            elif job[0] == "NoJob":
                if t >= 5:
                    self.log.error("Waiting too long for jobs. Shutting down.")
                    break
                else:
                    t *= 1.1
                    self.log.warning(f"No Job received. Waiting {t}s to retry")
                    sleep(t)
                    continue

            elif job[0] == "Shutdown":
                self.log.info("Shutdown requested by server")
                break

            else:
                try:
                    raise ClientError("Job {} not recognised!!".format(job))

                except ClientError as error:
                    self.log.exception(error)
                    raise

            # Return the minimised/SPE cluster to the GA
            # noinspection PyUnboundLocalVariable
            self.log.info(f"{self}: returning cluster: {result_cluster} from job: {quench}")

            if isinstance(result_cluster, Cluster):
                self.GA.return_result(dumps(result_cluster), self.id)
            else:
                self.log.info(f"{self} attempted to return {result_cluster} instead of a Cluster. Stopping.")
                break

            self.log.debug(f"{self}: ending job {quench}")
            quench += 1

        self.log.info("Quench process: {} exiting successfully after preforming {} jobs.\n".format(self, quench))
        exit(0)
