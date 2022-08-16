from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.regression import Regression
regression = Regression()
''' o padrão que deverá ser reconhecido consiste em apenas dobrar cada elemento da entrada para obter a saída correspondente '''
inputs = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24], [25, 26, 27], [28, 29, 30]]
outputs = [[2, 4, 6], [8, 10, 12], [14, 16, 18], [20, 22, 24], [26, 28, 30], [32, 34, 36], [38, 40, 42], [44, 46, 48], [50, 52, 54], [56, 58, 60]]
regression.fit(
	inputs=inputs, # variáveis independentes/explicativas/preditoras
	outputs=outputs, # variáveis dependentes/de resposta
	regression_type='linear', # tipo de cálculo regressivo a ser aplicado
	degree=None, # número real utilizado somente quando se quer alterar a inclinação da linha regressiva com base na extremidade final
	alpha=None, # número real utilizado somente quando se quer alterar o grau de horizontalidade da linha regressiva, quanto mais próximo de 0.5 mais horizontal será a linha
	same_output=False, # se definido como True retornará somente resultados que constem nas respostas do treinamento
	only_integers=False, # se definido como True usará o cálculo para regressão de números inteiros forçando somente números inteiros como resposta
	count=False, # se definido como True usará o cálculo de contagem para que os resultados sejam somente inteiros maiores ou iguais a zero
	nonlinear=None, # booleano definido como True somente quando se quer aplicar o cálculo de regressão não linear
	outliers=True # se definido com False removerá os valores discrepantes e tendenciosos
)
'''
nos algoritmos de aprendizado de máquina a dimensionalidade das entradas da predição deverá ser sempre a mesma das entradas do treinamento. 
como as entradas do treinamento possuem 3 elementos as entradas da predição também deverão possuir 3.
'''
new_inputs = [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22], [23, 24, 25], [26, 27, 28], [29, 30, 31]]
new_outputs = regression.predict(inputs=new_inputs)
print(new_outputs)