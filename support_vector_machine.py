from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.support_vector_machine import SupportVectorMachine
support_vector_machine = SupportVectorMachine()
'''
dados de entrada e saída para os exemplos que serão usados na aprendizagem.
* mesmo que não seja obrigatório, procure sempre usar listas bidimensionais para estar de acordo com os padrões internacionais.
evite por exemplo outputs do tipo: ['unidade', 'dezena', 'centena'], ao invés disso use: [['unidade'], ['dezena'], ['centena']].
'''
inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena']]
support_vector_machine.fit(
	inputs=inputs, # lista com os exemplos de entrada
	outputs=outputs, # lista com os exemplos de saída
	gamma=len(inputs) # podemos utilizar a quantidade de dados de entrada ou saída como gamma se quisermos aplicar a separação máxima dos dados
)
support_vector_machine.saveModel('modelo_svm') # salvamento do modelo de aprendizado de máquina, será gerado um arquivo do tipo AI

new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
new_outputs = support_vector_machine.predict(inputs=new_inputs)
print(new_outputs)