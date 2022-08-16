from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork # importação do algoritmo de IA
neural_network = NeuralNetwork() # criação do objeto da classe NeuralNetwork
''' padrão regressivo, as saídas não são valores categóricos que se repetem '''
inputs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]] # variáveis explicativas
outputs = [[3], [7], [11], [15], [19], [23], [27], [31], [35], [39]] # variáveis de resposta, no padrão somamos os elementos da entrada para obtermos a saída correspondente

neural_network.fit( # fase de treinamento
	inputs=inputs, # matriz bidimensional com as listas numéricas de entrada
	outputs=outputs, # matriz bidimensional com as listas numéricas de saída
	epochs=5, # número de repetições até que a rede aprenda
	activation_function='nonlinear', # a função nonlinear (não linear) é uma função genérica que pode ser aplicada a qualquer tipo de padrão
	show_error=True # exibição do progresso da aprendizagem com a queda da taxa de perda
)
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21]] # entradas da predição
new_outputs = neural_network.predict(inputs=new_inputs) # fase de predição
print(new_outputs) # exibição das respostas