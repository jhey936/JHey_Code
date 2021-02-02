# coding=utf-8
"""
Defines the main Database class

Also defines the Cluster class which sets up the schema for the Database clusters table
"""
import os
import copy
import logging

from threading import Lock
from functools import partial
from typing import Callable, Union, List

from sqlalchemy import create_engine
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import sessionmaker, scoped_session

from bmpga.storage.molecule import Molecule
from bmpga.storage.cluster import Cluster, baseSQL, Minimum

# log = logging.getLogger(__name__)
# Set schema version
_schema_version = 0

# This creates a lock to be used throughout the database class and associated methods
# globalLock = Lock()


class LockMethod(object):
    """Decorator to lock and unlock database methods"""
    def __init__(self, f) -> None:
        """Initialises the lock method to be used

        Attributes:
            f: function, required, function to be locked

        Returns:
            None
        """
        self.f = f

    def __get__(self, obj, obj_type) -> partial:
        """Allows support for instance methods

        Attributes:
            obj: method, required, method of object to be locked
            obj_type: type(obj), required, type of the object to be locked

        Returns:
            wrapped_function: functools.partial, the locking function

        """
        return partial(self.__call__, obj)

    def __call__(self, obj, *args, **kwargs) -> Callable:
        """

        Attributes:
            obj: method to be locked
            *args: list, optional, other positional arguments
            **kwargs: dict, optional, other keyword arguments

        Returns:
            locking instance of the passed method

        """
        obj.lock.acquire()
        try:
            res = self.f(obj, *args, **kwargs)
            return res

        finally:
            obj.lock.release()


class Database(object):
    """
    Database storage class
    
    This class handles the connection to the database. 
    It creates new clusters and inserts them into the database. 


    Attributes:
        engine : sqlalchemy database engine
        get_session() : returns the a tread safe sqlalchemy session
        accuracy : float
    """
    engine = None
    session = None
    connection = None
    accuracy = 1e-3

    def __init__(self, db: str=":memory:",
                 accuracy: float=1e-6,
                 connect_string: str="sqlite:///{}",
                 compare_clusters: Callable=None,
                 max_clusters: int=None,
                 new_database: bool=False,
                 verbose: bool=False,
                 log=None) -> None:
        """
        Attributes:
            db : string, optional, filename of new or existing database to connect to. (default:":memory:")
            accuracy : float, optional, energy tolerance to count minima as equal. Default=1e-6
            connect_string : string, optional, connection string. Default is sqlite database
            compare_clusters : callable, `bool = compareClusters(min1, min2)`, optional, called to determine if
                         two minima are identical.  Only called if the energies are within `accuracy` of each
                         other. Default=None
            new_database : boolean, optional, create database if not exists. Default=True
        """

        self.file = db
        self.accuracy = accuracy
        self.max_clusters = max_clusters
        self.compareClusters = compare_clusters
        self.log = log or logging.getLogger(__name__)

        if self.compareClusters is None:
            self.log.warning("compareClusters is None. No checking for uniqueness of clusters will be performed.")

        # Check if db already exists and warn the user
        if self.file == ":memory:":
            self.log.warning("Creating database in memory. This is not persistent and will not be saved.")

        elif os.path.exists(self.file) and not new_database:
            self.log.info("File: {} exists and new_database is False. Attempting to use this database"
                          .format(self.file))

        elif os.path.exists(self.file) and new_database:
            self.log.warning("File {} exists and new_database is True. This file is being overwritten."
                             .format(self.file))
            os.remove(self.file)

        if not os.path.isfile(self.file) or self.file == ":memory:":

            if not new_database:
                try:
                    raise IOError("new_database is False, but database does not exist ({})".format(self.file))
                except IOError as error:
                    self.log.exception(error)
                    raise

        # https://stackoverflow.com/questions/34009296/using-sqlalchemy-session-from-flask-raises-
        # sqlite-objects-created-in-a-thread-c
        # set up the engine which will manage the backend connection to the database
        self.engine = create_engine(connect_string.format(self.file)+"?check_same_thread=False", echo=verbose)

        # set up the tables and check the schema version
        self._update_schema()
        if new_database:
            self._set_schema_version()

        if not new_database and not self._is_bmpga_database():
            try:
                raise IOError
            except IOError:
                message = "existing file ({}) is not a bmpga database.".format(db)
                self.log.exception(message)
                raise IOError(message)

        self._check_schema_version()

        # set up the session which will manage the frontend connection to the database
        # We are using scoped sessions to allow thread_safe interface
        session_factory = sessionmaker(bind=self.engine)
        self.get_session = scoped_session(session_factory)

        self.connection = self.engine.connect()
        self.lock = Lock()
        # self.lock = globalLock
        self.is_bmpga_database = True

    def __repr__(self) -> str:
        return "<bmpga.storage.Database @ {}>".format(self.file)

    def _set_schema_version(self) -> None:
        global _schema_version
        with self.engine.connect() as connection:
            connection.execute("PRAGMA user_version = {};".format(_schema_version))

    def _check_schema_version(self) -> None:
        global _schema_version
        with self.engine.connect() as connection:
            result = connection.execute("PRAGMA user_version;")
            database_schema = result.fetchone()[0]
            result.close()
        # TODO: write a utility to update databases to the latest schema
        if database_schema != _schema_version:
            raise DatabaseError("Database schema used in {} is outdated! {} != {}.".format(self.file,
                                                                                           database_schema,
                                                                                           _schema_version))

    def _update_schema(self) -> None:
        conn = self.engine.connect()
        baseSQL.metadata.create_all(bind=self.engine)
        conn.close()

    def _is_bmpga_database(self) -> bool:
        conn = self.engine.connect()
        result = True
        if not self.engine.has_table("tbl_clusters"):
            result = False
        conn.close()
        return result

    def add_new_cluster(self, cost: float, molecules: [Molecule], *args, **kwargs) -> Cluster:
        """Create and insert a new cluster object into the database

        If a compareClusters method has been

        Args:
            cost: float, required, Value of the cost function corresponding to the cluster to be inserted
            molecules: list of Molecules, required, Molecules making up the Cluster
            *args: other args
            **kwargs: other keyword args

        Returns:
            Cluster object which was inserted into the database

        """

        new_cluster = Minimum(cost=copy.deepcopy(cost), molecules=copy.deepcopy(molecules), *args, **kwargs)
        return self.insert_cluster(new_cluster)

    @LockMethod
    def fast_insert_clusters(self, clusters: list) -> None:
        """Insert a list of Cluster objects into the database

        Please note this method does not check if clusters are unique

        Args:
            clusters: list, required, list of Cluster objects to be inserted

        Returns:
            None

        """  # TODO: Test Database.insert_clusters
        session = self.get_session()
        new_clusters = [Minimum(cluster=c) for c in clusters]

        session.add_all(new_clusters)
        message = """Clusters fast-inserted without checking: {}
        This can lead to non-unique clusters in the database\n""".format(new_clusters)
        self.log.debug(message)
        session.commit()

    @LockMethod
    def insert_cluster(self, new_cluster: Union[Minimum, Cluster], *args, **kwargs) -> Cluster:
        """Insert a single existing_cluster into the database

        Note: this method does check that the new cluster is unique

        Args:
            new_cluster: Cluster, required, object to be inserted

        Returns:
            cluster: Cluster, inserted into database (or the pre-existing cluster already in the database)

        """
        # if isinstance(new_cluster, Cluster):
        cluster = Minimum(cluster=copy.deepcopy(new_cluster))

        session = self.get_session()

        cost = cluster.cost

        if self.compareClusters is not None:
            close_clusters = session.query(Minimum). \
                filter(Minimum.cost > cost - self.accuracy). \
                filter(Minimum.cost < cost + self.accuracy)

            for existing_cluster in close_clusters:
                # compareClusters returns True if clusters are the same
                old_cluster = Cluster(db_cluster=existing_cluster)
                if not self.compareClusters(new_cluster, old_cluster, *args, **kwargs):
                    continue
                else:
                    session.rollback()
                    session.commit()
                    return copy.deepcopy(old_cluster)
                
        else:
            message = f"""Cluster inserted without checking, as self.compareClusters is None: {cluster}
            This can lead to non-unique clusters in the database"""
            self.log.warning(message)

        try:
            session.add(cluster)
        except InvalidRequestError:
            session.rollback()

        session.commit()
        return Cluster(db_cluster=cluster)

    @LockMethod
    def get_global_minimum(self) -> Cluster:
        """
        Returns
        -------
        The Cluster object with lowest energy in the database

        """
        session = self.get_session()
        cluster = session.query(Minimum).order_by(Minimum.cost).first()
        # clusters = sorted(clusters, key=lambda x: x.cost)
        return Cluster(db_cluster=cluster)

    @LockMethod
    def get_minima(self) -> List[Cluster]:
        """

        Returns:
            A list of all the Minimum objects in the database ordered by energy

        """
        session = self.get_session()
        clusters = session.query(Minimum).order_by(Minimum.cost)
        return [Cluster(db_cluster=c) for c in clusters]

    # Setup minima as a property. So it will transparently return and insert minima
    minima = property(get_minima, fast_insert_clusters)

    @property
    @LockMethod
    def number_of_minima(self) -> int:
        """Returns the total number of minima in the database"""
        session = self.get_session()
        return session.query(Minimum).count()

    @LockMethod
    def get_clusters_by_id(self, ids: int or list) -> Cluster or list:
        """Returns clusters selected by id

        Can request single cluster or list of clusters

        If a list of ids is supplied, clusters will be returned in the same order

        Args:
            ids: int or list(int), required, id/s of cluster/s to fetch

        Returns:
            list of clusters with ids from above order is preserved

        """
        assert isinstance(ids, int) or (isinstance(ids, list) and isinstance(ids[0], int))

        session = self.get_session()

        if isinstance(ids, int):
            ids = [ids]

        clusters = []
        for _id in ids:
            clusters.append(Cluster(db_cluster=session.query(Minimum).get(_id)))

        if len(ids) == 1:
            return clusters[0]
        else:
            return clusters

    @LockMethod
    def get_clusters_by_cost(self, cost_max: float or int, cost_min: float or int=None,
                             max_number_of_clusters: int=None) -> list:
        """
        Method to return Clusters ordered by energy under a threshold (and optionally above another threshold)

        Parameters
        ----------
        cost_max: float, required, Clusters above this energy are not returned
        cost_min: float, optional, Clusters below this energy are excluded
        max_number_of_clusters: int, optional, number of lowest energy clusters to be returned (default is all)

        Returns
        -------
        List of Cluster objects ordered by ascending energy

        """
        session = self.get_session()
        if cost_min is not None:
            if cost_min >= cost_max and cost_min is not None:
                raise ValueError("Minimum cost: {} exceeds maximum cost: {}!".
                                 format(cost_min, cost_max))

            result = session.query(Minimum).filter(Minimum.cost <= cost_max).\
                filter(Minimum.cost >= cost_min)

        else:
            result = session.query(Minimum).filter(Minimum.cost <= cost_max)

        result = [Cluster(db_cluster=cluster) for cluster in result]

        return sorted(result[:max_number_of_clusters], key=lambda x: x.cost)


class DatabaseError(Exception):
    """Error called when the database raises an error"""
    pass
