from rdflib import Graph
import json


def caricaOntologyOrdinataSuFile(urlFile):
    g = Graph()
    g.parse(urlFile,format="xml")

    # Esegui un'interrogazione SPARQL per ottenere i tipi e le superclassi
    query = """
        SELECT ?type ?superclass
        WHERE {
          ?type rdfs:subClassOf* ?superclass .
        }
        """
    results = g.query(query)

    type_hierarchy = {}
    for row in results:
        type_uri = row["type"]
        superclass_uri = row["superclass"]

        if type_uri not in type_hierarchy:
            type_hierarchy[type_uri] = []

        type_hierarchy[type_uri].append(superclass_uri)

    with open("Ontology/orderedTypeList", "w") as fp:
        json.dump(type_hierarchy, fp)

# Funzione per ordinare i tipi in base alla gerarchia
def sort_types(types,type_hierarchy):
    visited = set()
    stack = []

    for type_uri in types:
        if type_uri not in visited:
            topological_sort(type_uri, visited, stack,type_hierarchy)

    return stack   #[::-1]   Inverte l'ordine dello stack per ottenere l'ordine corretto


# Funzione per l'ordinamento topologico
def topological_sort(node, visited, stack,type_hierarchy):
    visited.add(node)
    countNeighbours=0
    if node in type_hierarchy:
        countNeighbours=len(type_hierarchy[node])
        #for neighbor in type_hierarchy[node]:
            #countNeighbours+=1
            # if neighbor not in visited:
            #     topological_sort(neighbor, visited, stack,type_hierarchy)

    stack.append({'key':node,'num':countNeighbours})

#Da far partire per caricare l'ontologia elaborata su file
if __name__ == '__main__':

    caricaOntologyOrdinataSuFile("Ontology/ontology_type=orig.owl")
    print("finito")
