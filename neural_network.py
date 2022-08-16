from Neuraline.ArtificialIntelligence.DeepLearning.neural_network import NeuralNetwork # importação do algoritmo
neural_network = NeuralNetwork() # instanciação da classe para acessar os métodos pela variável neural_network
''' unidades representadas por 0 e dezenas representadas por 1, em redes neurais utilizamos apenas números '''
inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [70, 80]] # exemplos de entrada passados para a rede aprender
outputs = [[0], [1], [0], [1], [0], [1], [0], [1]] # o número de saídas deverá ser sempre igual ao número de entradas, 8 inputs = 8 outputs
''' nas redes neurais é obrigatório o uso de listas para representar os elementos das matrizes de entrada e saída '''

'''
da mesma forma que humanos aprendem com tarefas repetitivas as nedes neurais artificiais também aprendem por repetição.
o número de repetições é definido no parâmetro epochs, neste caso apenas 5 já foram o suficiente.
se os resultados forem imprecisos você deverá aumentar o número de épocas no parâmetro epochs.
'''
neural_network.fit(
	inputs=inputs, # entradas de exemplo
	outputs=outputs, # saídas de exemplo
	epochs=5, # quantidade de repetições na aprendizagem, tarefas repetitivas
	activation_function='binary_step', # função binária para retornar somente 0 ou 1 que é o padrão de resposta dos outputs do treinamento
	show_error=True # exibição do progresso da aprendizagem, quanto maior a aprendizagem menor será a loss (taxa de erro)
)
new_inputs = [[2, 3], [20, 30], [4, 5], [40, 50], [6, 7], [60, 70], [8, 9], [80, 90]] # novos inputs para testar a aprendizagem
new_outputs = neural_network.predict(inputs=new_inputs) # método que retornará o resultado da predição
print(new_outputs) # exibe a matriz com os outputs encontrados para as entradas em new_inputs