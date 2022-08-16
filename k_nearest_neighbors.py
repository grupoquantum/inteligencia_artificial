# importação da classe do KNN contida no arquivo importado do módulo de Aprendizado Supervisionado
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.k_nearest_neighbors import KNearestNeighbors
k_nearest_neighbors = KNearestNeighbors() # instanciação da classe na variável objeto

inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [80, 90]] # dados de entrada
outputs = [[0], [1], [0], [1], [0], [1], [0], [1]] # rótulos de saída possíveis
k_nearest_neighbors.fit(inputs=inputs, outputs=outputs, k=0) # treinamento com zero vizinhos mais próximos
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [20, 30], [40, 50], [60, 70], [80, 90]] # dados que não constam no treinamento para testar a predição
new_outputs = k_nearest_neighbors.predict(inputs=new_inputs) # predição do resultado
print(new_outputs) # exibição do resultado da predição