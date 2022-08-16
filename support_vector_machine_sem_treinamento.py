from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.support_vector_machine import SupportVectorMachine
support_vector_machine = SupportVectorMachine()

support_vector_machine.loadModel('modelo_svm') # carregamento do arquivo .ai com os padrões que queremos aplicar
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
new_outputs = support_vector_machine.predict(inputs=new_inputs)
print(new_outputs)