# importação da classe do Hierarchical Clustering contida no arquivo importado do módulo de Aprendizado Autônomo
from Neuraline.ArtificialIntelligence.MachineLearning.AutonomousLearning.hierarchical_clustering import HierarchicalClustering
hierarchical_clustering = HierarchicalClustering() # instanciação da classe na variável objeto
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [80, 90]] # dados de entrada, NÃO necessita de saída
hierarchical_clustering.fit(inputs=inputs) # treinamento com os dados de entrada
new_outputs = hierarchical_clustering.predict(clusters=2) # predição com o resultado das entradas divididas em dois grupos/clusters
print(new_outputs) # exibição do resultado da predição, teremos uma matriz tridimensional com uma lista bidimensional para cada grupo classificativo