# importação da classe do Q-Learning contida no arquivo importado do módulo de Aprendizado por Reforço
from Neuraline.ArtificialIntelligence.MachineLearning.ReinforcementLearning.q_learning import QLearning
q_learning = QLearning() # instanciação da classe na variável objeto

inputs = ['ação 1', 'ação 2', 'ação 3'] # ações possíveis em uma lista unidimensional
outputs = [[1, 2, 3, 5, 5], [2, 1, 4, 2, 2], [0, 1, 5, 3, 4]] # sequência de estados para cada ação executada
'''
os outputs/estados representam as punições (para valores menores) e as recompensas (para valores maiores) 
e o resultado corresponderá a sequência de ações com os maiores estados
'''
q_learning.fit(actions=inputs, states=outputs) # treinamento do algoritmo
q_learning.saveQTable(url_path='qtable') # salvamento da tabela Q (qualitativa) no diretório local com o nome "qtable"
actions = q_learning.predict() # a predição não contém parâmetros
print(actions) # exibição da sequência de ações necessária para maximizar a recompensa