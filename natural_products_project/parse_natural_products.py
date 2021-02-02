import os
import sys
sys.path.append(os.path.abspath(os.curdir))

# import os

import json

import networkx as nx

import traceback

from natural_products.tree_analysis import add_element, tree_creation
from glob import glob
from natural_products.cycleseparation import analysis_of_rings
from natural_products.io_utils import Counter, parse_mol_file
from natural_products.NaturalProductStorage import Cycle, NaturalProduct, Database, JunctionTable
from  natural_products.smiles import smiles

import sqlalchemy as sql
from time import sleep


from multiprocessing import Pool, current_process


def insert_mol_cycles(db, natural_prod, all_cycles, session):
    db.insert(natural_prod, session)
    session.commit()
    
    for cyc in all_cycles:
        db.insert(cyc, session)
        session.commit()

def deal_with_mol_file(mol, db):

    session = db.get_session()

    try:
        print(mol)

        nps = session.query(NaturalProduct).filter(NaturalProduct.file_path == mol).all()
        #session.close()

        if len(nps) >= 1:
            print("MOL file already in db: ", mol)
            session.rollback()
            session.commit()
            session.close()
            return

        
        G, cycles = parse_mol_file(mol, show_graph=False, save_graph=False)
        
        name = mol.split(".")[0].split(r"/")[-1]
        # print(name)

        with open(mol, "rb") as molf:
            mol_file_contents = molf.read()

        # add the np to the tables

        
        cycle_graphs = [nx.Graph(G.subgraph(cycle)) for cycle in cycles]
        

        try:

            json_file = mol.split(".")[0]+".json"
            with open(json_file, "r") as json_f:
                    json_data = json.load(json_f)

            natural_prod = NaturalProduct(np_name=name, np_info=json_data, file_path=mol, file=mol_file_contents, graph=G)

        except AttributeError as E: 
            with open(f"nprod_errors_{current_process()}.error", "a") as E_file:
                E_file.write("@@@@@~~~~~@@@@@\nError whilst parsing: {mol}\n",E)
            session.close()
            return

        # try:                        

        cycle_definitions = analysis_of_rings(G, cycles)

        if len(cycle_graphs) != len(cycle_definitions):
            with open("cycle_graphs.error", "a") as E_file:
                E_file.write(f"len(cycle_graphs) != len(cycle_definitions): {len(cycle_graphs) != len(cycle_definitions)}\n")
            session.close()
            return

        all_cycles = []
        for cycle_graph, cycle_definition in zip(cycle_graphs, cycle_definitions):
            try: 
                smile = smiles(G, cycle_graph)

            except ValueError as E:
                with open(f"cycle_errors_{current_process()}.error", "a") as cycle_errors:
                    cycle_errors.write(f"Problem with generating smiles for cycle: {[str(a) for a in cycle_graph.nodes()]}\nFrom file: {mol}\nCycle not added to database!\n{E}\n@@@@@@@@@@@@")
                session.close()
                return

            except nx.exception.NetworkXNoPath as E:
                with open(f"cycle_errors_{current_process()}.error", "a") as cycle_errors:
                    cycle_errors.write(f"Problem with generating smiles for cycle: {[str(a) for a in cycle_graph.nodes()]}\nFrom file: {mol}\nCycle not added to database!\n{E}\n@@@@@@@@@@@@")
                session.close()
                return

            except nx.NetworkXError as E:
                with open(f"broken_mol_files_{current_process()}.error","a") as broken:
                    broken.write("Failed to handle: "+mol+"\n")
                session.close()
                return
            
            
                

            try:
                cycle_database = Cycle(cycle_representation=smile,
                                   cycle_separation=cycle_definition,
                                   np_id=[natural_prod])
                
                all_cycles.append(cycle_database)

            except AttributeError as E:
                with open(f"nprod_errors_{current_process()}.error", "a") as E_file:
                    E_file.write(f"Problem with adding cycle to database: {[str(a) for a in cycle_graph.nodes()]}\nFrom file: {mol}\nThis generated the following error:\n{E}\nCycle not added to database!\n@@@@@@@@@@@@")
                session.close()
                return

            # else:
            #     add_element(tree_cycles, cycle_definition, cycle_graph, ring_counter)


    except Exception as E:
        with open("broken_mol_files.error","a") as broken:
            broken.write("Failed to handle: "+mol+"\n")
        with open("unhandled_exception.error", "a") as ef:
            ef.write("''''''''''\n")
            ef.write(traceback.format_exc())
            ef.write("\n")
            ef.write(str(E))
            ef.write("''''''''''\n")
        
        session.close()
        return

    else:
        try:
            insert_mol_cycles(db, natural_prod, all_cycles, session)

        except Exception as E:

            with open("unhandled_exception.error", "a") as ef:
                ef.write("''''''''''\n")
                ef.write(traceback.format_exc())
                ef.write("\n")
                ef.write(str(E))
                ef.write("Removing from database")
                ef.write("''''''''''\n")

            for cyc in all_cycles: 
                
                if natural_prod in cyc.np_id:
                    cyc.np_id.remove(natural_prod)
                    cyc.frequency -= 1

            if natural_prod.np_id is not None:
                #session = db.get_session()
                session.delete(natural_prod)
                session.commit()
                #session.close()

            session.rollback()
            session.commit()
            session.close()
            return

        except sqlalchemy.orm.exc.DetachedInstanceError as E:
            session.rollback()
            session.commit()
            session.close()

        else:
            with open(f"successful_mols.out_{current_process()}", "a") as outF:
                outF.write(mol+"\n")
            session.commit()
            session.close()
            return 0


    finally:
        session.close()

if __name__ == "__main__":


    print(f"Searching for mol files in: {sys.argv[1]}")
    
    MolFs = glob(sys.argv[1]+"/*.mol")

    # database set up here
    db = Database("nat_prods.db", new_database=True)

    def wrap_handle_mol_files(MolF):
        deal_with_mol_file(MolF, db)

        
    pool = Pool(4)
    pool.map(wrap_handle_mol_files, MolFs)
