import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina


smiles = [
"CC1=CC(=O)Nc2ccc(cc2)C1CC(=O)Nc3ccc(cc3)C2=CC=CC=C2CC(=O)Nc4ccc(cc4)C(=O)OC(=O)OC(=O)NC(=O)S",
"CC1=CC(=O)Nc2ccc(cc2)C1CC(=O)Nc3ccc(cc3)C2=CC=CC=C2CC(=O)Nc4ccc(cc4)C(=O)OC(=O)OC(=O)C",
"CC1=CC(=O)Nc2ccc(cc2)C1CC(=O)Nc3ccc(cc3)C2=CC=CC=C2CC(=O)Nc4ccc(cc4)C(=O)OC(=O)OC(=O)NC(=O)O",
"CC1=CC(=O)Nc2ccc(cc2)C1CC(=O)Nc3ccc(cc3)C2=CC=CC=C2CC(=O)Nc4ccc(cc4)C(=O)OC(=O)OC",
"CC1=CC=CC=C1Nc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)S(=O)(=O)Nc4ccc(cc4)Nc5ccc(cc5)NCc6ccc(cc6)C",
"CC1=CC(=O)Nc2ccc(cc2)S(=O)(=O)C1CC(=O)Nc3ccc(cc3)S(=O)(=O)C2=CC=CC=C2CC(=O)Nc4ccc(cc4)S",
"CC1=CC=CC=C1Nc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)Nc4ccc(cc4)S(=O)(=O)Nc5ccc(cc5)O",
"CC1=CC=CC=C1Nc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)Nc4ccc(cc4)S(=O)(=O)Nc5ccc(cc5)C",
"CC1=CC=CC=C1Nc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)Nc4ccc(cc4)S(=O)(=O)Nc5ccc(cc5)NC",
"CC1=CC=CC=C1Nc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)Nc4ccc(cc4)S(=O)(=O)Nc5ccc(cc5)Nc6ccc(cc6)N"]

def morgan_fp(mol):
   morgan = AllChem.GetMorganGenerator(radius=2, fpSize=512)
   return morgan.GetFingerprint(mol)

def butina(fingerprints):
   matrix = []

   for i in range(len(fingerprints)):
       for j in range(i):
           similaridade = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
           matrix.append(1 - similaridade)

   clusters = Butina.ClusterData(data=matrix, nPts=len(fingerprints), distThresh=0.5, isDistData=True)
   clusters = sorted(clusters, key=len, reverse=True)

   return clusters

def cluster(smiles):
   df = pd.DataFrame(
      data={
         "smiles": smiles
      }
   )

   PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')

   df['morgan'] = df['ROMol'].apply(morgan_fp)

   indices = butina(df['morgan'])
   
   molecules = []
   for index_cluster in indices:
      molecules.append(smiles[min(index_cluster)])
      
   return molecules
   