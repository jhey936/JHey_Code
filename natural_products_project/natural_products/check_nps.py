from NaturalProductStorage import *
import sys



db = Database(sys.argv[1])

print(db.number_of_natural_products)
