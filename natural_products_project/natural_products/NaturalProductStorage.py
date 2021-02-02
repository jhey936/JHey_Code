import os
import logging

from threading import Lock
from functools import partial
from typing import Callable

from time import sleep


import sqlalchemy
from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.orm import sessionmaker, scoped_session, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, PickleType, create_engine, BLOB, ForeignKey, func#, JSON
from natural_products.smiles import join_set, list_comparison


Base = declarative_base()

# Set schema version
_schema_version = 0

class JSONType(PickleType):
    '''
        JSON DB type is used to store JSON objects in the database
    '''

    impl = BLOB

    def __init__(self, *args, **kwargs):        
        
        #kwargs['pickler'] = json
        super(JSONType, self).__init__(*args, **kwargs)

    def process_bind_param(self, value, dialect):
        
        if value is not None:
            value = json.dumps(value, ensure_ascii=True)
        return value

    def process_result_value(self, value, dialect):

        if value is not None:
            value = json.loads(value)
        return value


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


class NaturalProduct(Base):
    __tablename__ = "natural_product_data"

    np_id = Column(Integer, primary_key=True)
    
    np_name = Column(String)
    cycles = relationship("Cycle", secondary="junction_table")
    file_path = Column(String)
    np_info = Column(JSONType, default = {} )
    file = Column(BLOB)  # trying to find the SQL data type for the files themselves, I think it's this
    graph = Column(PickleType)  # This is he format I think I would use for storing a graph

    def __repr__(self): 
        return f"<Natural_product(np_id='{self.np_id}', np_name='{self.np_name}', cycles='{self.cycles}', " \
               f"np_info='{self.np_info}', file_path='{self.file_path}', file='{self.file}', " \
               f"graph_of_natural_product='{self.graph}')>"

    def __init__(self, np_name, np_info, file_path, file, graph, cycles=None):

        self.np_name = np_name
        self.np_info = np_info
        self.file_path = file_path
        self.file = file
        self.graph = graph

        if cycles is not None:
            self.cycles = cycles


class Cycle(Base):
    __tablename__ = "cycles_data"

    cycle_id = Column(Integer, primary_key=True)

    cycle_representation = Column(PickleType)
    np_id = relationship("NaturalProduct", secondary="junction_table")
    cycle_separation = Column(PickleType)
    frequency = Column(Integer)

    def __repr__(self):
        return f"<Cycles(cycle_id='{self.cycle_id}',cycle_representation='{self.cycle_representation}'," \
               f",cycle_separation='{self.cycle_separation}', frequency='{self.frequency})>"
        # f"np_id='{self.np_id}'" \

    def __init__(self, cycle_representation, cycle_separation, np_id):
        self.cycle_representation = sorted(join_set(cycle_representation))
        self.cycle_separation = cycle_separation
        if isinstance(np_id, list):
            self.np_id = np_id
        else:
            self.np_id = [np_id]
        self.frequency = 1


class JunctionTable(Base):
    __tablename__ = "junction_table"

    relationship_id = Column(Integer, primary_key=True)

    np_id = Column(Integer, ForeignKey("natural_product_data.np_id"))
    cycle_id = Column(Integer, ForeignKey("cycles_data.cycle_id"))

    def __repr__(self):
        return f"<Cycles(relationship_id='{self.relationship_id}'cycle_id='{self.cycle_id}'," \
               f"np_id='{self.np_id}')>"

    def __init__(self, cycle_id, np_id):
        self.cycle_id = cycle_id
        self.np_id = np_id


class Database(object):
    """
    Database storage class
    
    This class handles the connection to the database. 
    It creates new clusters and inserts them into the database. 


    Attributes:
        engine : sqlalchemy database engine
        get_session() : returns the a tread safe sqlalchemy session
    """
    engine = None
    session = None
    connection = None

    def __init__(self, db: str=":memory:",
                 connect_string: str="sqlite:///{}",
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
        self.log = log or logging.getLogger(__name__)

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
        self.engine = create_engine(connect_string.format(self.file)+"?check_same_thread=False", echo=verbose, connect_args={"timeout": 500})

        # if not new_database:
        #     try:
        #         raise IOError
        #     except IOError:
        #         message = "existing file ({}) is not a natural product database.".format(db)
        #         self.log.exception(message)
        #         raise IOError(message)

        self._update_schema()
        if new_database:
            self._set_schema_version()

        # set up the session which will manage the frontend connection to the database
        # We are using scoped sessions to allow thread_safe interface
        self.session_factory = sessionmaker(bind=self.engine)
        #self.get_session = scoped_session(self.session_factory)      

        self.connection = self.engine.connect()
        self.lock = Lock()
        self.is_np_database = True

    def get_session(self):
        return scoped_session(self.session_factory)

    def __repr__(self) -> str:
        return "<np.Storage.Database @ {}>".format(self.file)

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
            raise TypeError("Database schema used in {} is outdated! {} != {}.".format(self.file,
                                                                                           database_schema,
                                                                                           _schema_version))

    def _update_schema(self) -> None:
        conn = self.engine.connect()
        Base.metadata.create_all(bind=self.engine)
        conn.close()

    def _count_natural_products(self, NaturalProduct):

        session = self.get_session()
        rows = session.query(func.count(NaturalProduct.id)).scalar()
        return rows

    def _fraction_of_cycles(self, cycle):

        session = self.get_session()

        freqs = session.query(Cycle.frequency).all()

        tot = 0
        for f in freqs:
            tot += f[0]

        cycle_freq = session.query(Cycle).get(cycle.cycle_id).frequency

        # print("@@", tot, cycle_freq)
        # print("@@", cycle_freq/tot)

    @LockMethod
    def insert(self, item, session=None, *args, **kwargs):
        """Insert a single existing_natural_product into the database

        Note: this method does check that the new natural_product is unique

        Args:
            new_Natural_product: natural_product, required, object to be inserted

        Returns:
            natural_product: Natural_product, inserted into database (or the pre-existing natural_product already in the database)
            """
        close_session = False
        if session is None:
            session = self.get_session()
            close_session = True

        try:
            if isinstance(item, Cycle):
                # epoxy = False
                # if ",".join([str(i) for i in item.cycle_separation]) == ",".join([str(i) for i in [3, 2, 1, 0, 0]]):
                #     epoxy = True
                #     # print("@ == Adding epoxy")

                exists = session.query(Cycle).filter(Cycle.cycle_representation == item.cycle_representation)\
                    .filter(Cycle.cycle_separation == item.cycle_separation).all()
                
                
                if exists != []:
                    # if epoxy:
                    #     # print("@ --- Adding epoxy")
                    
                    new_nps = item.np_id
                    item = exists[0]
                    item.frequency += 1
                    item.np_id.extend(new_nps)

                else:
                    session.add(item)

            elif isinstance(item, NaturalProduct):
                session.add(item)

            session.commit()


        except InvalidRequestError as E:
            session.rollback()
            session.commit()
            raise E

        except Exception as E: 
            session.rollback()
            session.commit()
            raise E

        finally:
            session.commit()
            if close_session:
                session.close()

        return item

    # def bulk_insert_cycles(self, cycle_list):
        

    #     session = self.get_session()
        

    @property
    def number_natural_products(self):
        session = self.get_session()
        return session.query(NaturalProduct).count()

    @LockMethod
    def get_natural_product_by_number(self, np_label):
        session = self.get_session()
        np = session.query(NaturalProduct).filter(NaturalProduct.np_name == np_label)[0]
        return np

    @property
    def natural_products(self):
        session = self.get_session()
        query_result = session.query(NaturalProduct)
        return query_result[:]
    
    @property
    def number_of_natural_products(self):
        session = self.get_session()
        return session.query(NaturalProduct).count()

    def count_cycles(self):
        session = self.get_session()

        return session.query(Cycle).count()


if __name__ == "__main__":
    # import numpy as np
    
    db = Database(new_database=True)
    # print(f"Database created at: {db}")
    n_p = NaturalProduct(np_name="np1", file_path="hello.txt", file=b"werfdserfd", graph="This is a graph")
    np_2 = NaturalProduct(np_name="np2", file_path="yoyo.txt", file=b"dsfsdkf", graph="this is a graph 2.0")
    # print(f"test natural product created: {n_p}")
    np_2 = db.insert(np_2)
    n_p = db.insert(n_p)
    # print("Natural_product inserted into database")
    n_p2 = db.get_natural_product_by_number("np1")
    # print(f"{n_p2} retrieved from database")
    # print(f"{np_2} retrieved from database")
    example_1 = Cycle(cycle_representation="C-C-C-C=", cycle_separation="6,6,6,5,4", np_id=[n_p])
    example_2 = Cycle(cycle_representation="C=C=C=C-", cycle_separation="6,6,5,4", np_id=[np_2],)
    example_3 = Cycle(cycle_representation="C-C-C", cycle_separation="6,6,5,4", np_id=[np_2],)
    example_4 = Cycle(cycle_representation="C-C-C", cycle_separation="6,6,5,4", np_id=[np_2],)
    np_cycles = example_1, example_2, example_3, example_4

    # print(f"test cycles created: {np_cycles}")

    for c in np_cycles:
        db.insert(c)

    # print("================", np_cycles[0].np_id)

    # print("cycle inserted into database")
    # np_cycles2 = db.get_cycles_data_by_number("np_cycle1")
    # # print(f"{np_cycles2} retrieved from database")

    db._fraction_of_cycles(example_1)

    # I have made this a property, 
    # so you can access the result of this function without using the brackets. 
    # i.e. db.number_of_natural_products vs db.number_of_natural_products()
    N_np = db.number_of_natural_products
    # N_cycle_np = db.number_of_cycles
    # print(f"{N_np} Natural_products in database")
    # # print(f"{N_cycles_np} cycle_data in database")

    #
    # # this # prints all the nat_prods in the database
    # # print(f"All Natural_products in database: {db.natural_products}")
    # # print(f"All cycle_data in database:{db.cycles_data}")
