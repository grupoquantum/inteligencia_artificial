from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.decision_tree import DecisionTree
decision_tree = DecisionTree()

inputs = [[1, 2], [10, 20], [100, 200], [3, 4], [30, 40], [300, 400], [5, 6], [50, 60], [500, 600]]
outputs = [['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena']]
decision_tree.fit(
	inputs=inputs, # lista com os exemplos de entrada
	outputs=outputs, # lista com os exemplos de saída
	extra_trees=True # podemos aumentar a precisão na separação das classes habilitando árvores extras que serão acrescentadas durante o treinamento (poderá deixar a execução mais lenta)
)

new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
expected_outputs = [['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena']]
result = decision_tree.test(inputs=new_inputs, outputs=expected_outputs) # fase de teste
print(result)