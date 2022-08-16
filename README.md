<h4>Inteligência Artificial</h4>
<p align="justify">
Inteligência Artificial refere-se a um conjunto de regras e algoritmos matemáticos baseados na lógica da aprendizagem humana com características preditivas. Assim como nós humanos recebemos informações e aprendemos com elas a Inteligência Artificial também aprende com as informações recebidas. Os algoritmos de Inteligência Artificial (ou IA) podem aprender através de três tipos principais de aprendizagem: Aprendizado Supervisionado, Aprendizado NÃO Supervisionado (ou Aprendizado Autônomo) e Aprendizado por Reforço.
</p>
<h5>Aprendizado Supervisionado (Supervised Learning)</h5>
<p align="justify">
O Aprendizado Supervisionado é quando temos exemplos de respostas disponíveis e passamos essas respostas para o treinamento como dados de saída para cada uma das entradas que foram utilizadas. As entradas correspondem aos dados que foram utilizados para se conseguir as saídas definidas. As saídas por sua vez correspondem aos resultados de cada uma das entradas.
</p>
<p align="justify">
Como exemplo de Aprendizado Supervisionado usaremos o K-Nearest Neighbors (ou KNN) que é um algoritmo que calcula a distância euclidiana entre a entrada da predição e as entradas do treinamento e retorna a saída pertencente a entrada do treinamento mais próxima da entrada da predição.
</p>
<p align="justify">
Note que definimos como entrada do treinamento listas com duas unidades ou listas com duas dezenas e estamos informando para o algoritmo que as unidades correspondem ao número 0 e as dezenas ao número 1. Observe também que estamos utilizando listas com um único elemento na saída, nada nos impede de usar somente números na saída mas como a maioria dos dados em análise estatística são salvos como tabelas no formato CSV com linhas e colunas em duas dimensões, é sempre uma boa prática usar uma lista bidimensional na saída para que cada lista interna represente uma linha e cada elemento numérico dessa lista represente uma coluna.
</p>
<br>
<pre>
  <code>
# importação da classe do KNN contida no arquivo importado do módulo de Aprendizado Supervisionado
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.k_nearest_neighbors import KNearestNeighbors
k_nearest_neighbors = KNearestNeighbors() # instanciação da classe na variável objeto

inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [80, 90]] # dados de entrada
outputs = [[0], [1], [0], [1], [0], [1], [0], [1]] # rótulos de saída possíveis
k_nearest_neighbors.fit(inputs=inputs, outputs=outputs, k=0) # treinamento com zero vizinhos mais próximos
new_inputs = [[2, 3], [4, 5], [6, 7], [8, 9], [20, 30], [40, 50], [60, 70], [80, 90]] # dados que não constam no treinamento para testar a predição
new_outputs = k_nearest_neighbors.predict(inputs=new_inputs) # predição do resultado
print(new_outputs) # exibição do resultado da predição
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[0], [0], [0], [0], [1], [1], [1], [1]]
  </code>
</pre>
<br>
<p align="justify">
Observe que através de exemplos de entrada (input) e saída (output) no treinamento o algoritmo foi capaz de prever o resultado de saída das entradas da predição mesmo elas não constando nas entradas do treinamento. Como o KNN é um algoritmo classificativo, as respostas do método “predict” deverão obrigatoriamente serem iguais a uma das saídas do método “fit”. O parâmetro “k=0” quer dizer que a partir da entrada mais próxima encontrada no treinamento nós não iremos selecionar nenhuma entrada mais próxima a ela. Se fosse definido por exemplo “k=3” o algoritmo encontraria o input mais próximo e selecionaria os 3 mais próximas a ele e usaria os 4 (a entrada mais próxima com as 3 mais próximas a ela) para contar o output que mais se repete entre as seleções, retornando esse output como resposta. Porém como foi dito aqui como definimos 0 para o número de vizinhos o algoritmo considerará apenas o output da saída mais próxima como resultado.
</p>
<h5>Aprendizado NÃO Supervisionado/Aprendizado Autônomo (Autonomous Learning)</h5>
<p align="justify">
O Aprendizado NÃO Supervisionado que também é conhecido como Aprendizado Autônomo é quando o algoritmo aprende sem a utilização de saídas (outputs) de exemplo. Nos algoritmo de Aprendizado Autônomo o parâmetro “output” não existe ou quando existe não é obrigatório. Para o nosso exemplo usaremos o algoritmo de agrupamento/clusterização Hierarchical Clustering (Agrupamento Hierárquico ou Clusterização Hierárquica) que não possui o parâmetro “output” e classifica as entradas as agrupando em um número específico de grupos definido no parâmetro “clusters” da predição. Observe que usando as mesmas entradas que usamos no KNN obteremos a mesma separação dos dados, porém agora essa separação é realizada por grupos.
</p>
<br>
<pre>
  <code>
# importação da classe do Hierarchical Clustering contida no arquivo importado do módulo de Aprendizado Autônomo
from Neuraline.ArtificialIntelligence.MachineLearning.AutonomousLearning.hierarchical_clustering import HierarchicalClustering
hierarchical_clustering = HierarchicalClustering() # instanciação da classe na variável objeto

inputs = [[1, 2], [10, 20], [3, 4], [30, 40], [5, 6], [50, 60], [7, 8], [80, 90]] # dados de entrada, NÃO necessita de saída
hierarchical_clustering.fit(inputs=inputs) # treinamento com os dados de entrada
new_outputs = hierarchical_clustering.predict(clusters=2) # predição com o resultado das entradas divididas em dois grupos/clusters
print(new_outputs) # exibição do resultado da predição, teremos uma matriz tridimensional com uma lista bidimensional para cada grupo classificativo
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[[7, 8], [5, 6], [3, 4], [1, 2]], [[80, 90], [50, 60], [30, 40], [10, 20]]]
  </code>
</pre>
<br>
<p align="justify">
Note que da mesma forma que estávamos representando a classificação no KNN aqui os dados também puderam ser separados entre listas com unidades e listas com dezenas, isso por que o agrupamento nada mais é do que uma classificação autônoma onde não conhecemos as respostas previamente.
</p>
<h5>Aprendizado por Reforço/Aprendizado Reforçado (Reinforcement Learning)</h5>
<p align="justify">
O Aprendizado por Reforço que também pode ser chamado de Aprendizado Reforçado é um método de aprendizagem onde o algoritmo aprende através de um sistema de punições e recompensas, da mesma maneira que podemos fazer uma criança a aprender a se comportar chamando a atenção quando ela erra ou dando um presente quando acerta. Ou por exemplo quando treinamos cães que recebem um petisco quando executam uma determinada tarefa durante o treinamento.
</p>
<p align="justify">
Para exemplificar de forma simples um caso de Aprendizado por Reforço, iremos utilizar o algoritmo Q-Learning (Aprendizagem Q, o Q refere-se à aprendizagem qualitativa). Cada ação possível conterá uma qualidade ou pontuação que a descreve em um determinado estado sequencial. No nosso exemplo iremos pressupor que estamos operando um jogo que possui 3 ações possíveis e que armazenamos as pontuações de 3 jogadas que terminam sempre na quinta ação. Agora precisamos que o algoritmo nos retorne a sequência das 5 ações entre as disponíveis que nos resulte na maior pontuação.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
['ação 2', 'ação 1', 'ação 3', 'ação 1', 'ação 1']
  </code>
</pre>
<br>
<p align="justify">
Analisando o arquivo texto da tabela Q podemos constatar que no primeiro estado a maior recompensa está na execução da ação 2, no segundo estado a maior recompensa está na execução da ação 1, no terceiro na ação 3, no quarto na ação 1 novamente e no quinto e último estado a maior recompensa também está na ação 1. Ou seja, para conseguirmos a pontuação máxima no jogo devemos executar os comandos da ação 2, ação 1, ação 3, ação 1 e ação 1 sequencialmente nessa mesma ordem. Essas ações podem representar por exemplo os comandos de um jogo como saltar, atirar e esperar e os estados representariam o ponto do cenário no qual você se encontra.
</p>
<br>
<pre>
  <code>
╒═══════════╤════════════╤════════════╤════════════╤════════════╤════════════╕
│ ACTIONS   │   1º State │   2º State │   3º State │   4º State │   5º State │
╞═══════════╪════════════╪════════════╪════════════╪════════════╪════════════╡
│ ação 1    │          1 │          2 │          3 │          5 │          5 │
├───────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ ação 2    │          2 │          1 │          4 │          2 │          2 │
├───────────┼────────────┼────────────┼────────────┼────────────┼────────────┤
│ ação 3    │          0 │          1 │          5 │          3 │          4 │
╘═══════════╧════════════╧════════════╧════════════╧════════════╧════════════╛
  </code>
</pre>
<br>
<p align="justify">
No nosso exemplo usamos os nomes “ação 1”, “ação 2” e “ação 3” apenas para facilitar mas você poderia substituí-los pelos nomes das ações disponíveis no seu jogo e as pontuações de recompensa poderiam ser qualquer valor inteiro ou real.
</p>
<h5>Salvando e Carregando Modelos</h5>
<p align="justify">
Os algoritmos de inteligência artificial da biblioteca possuem um método de nome “saveModel” para salvar o seu modelo e um método de nome “loadModel” para carregar o modelo que foi salvo eliminando a necessidade de novos treinamentos toda vez que precisarmos executar uma predição. 
</p>
<p align="justify">
Para o nosso exemplo usaremos o algoritmo de classificação Support Vector Machine (SVM) que irá classificar os nossos dados entre unidade, dezena e centena. Observe no código a seguir que também é possível utilizarmos valores do tipo texto nos outputs, mas isso só é permitido em algoritmos exclusivamente classificativos, em dados regressivos isso não seria possível.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena']]
  </code>
</pre>
<br>
<p align="justify">
Da próxima vez que a predição for executada com os mesmos padrões não será mais preciso treinar o algoritmo, bastando carregar o modelo correspondente ao padrão que se quer aplicar. Nós também poderíamos utilizar valores de entrada diferentes dos que foram usados na predição anterior sem problemas.
</p>
<br>
<pre>
  <code>
from Neuraline.ArtificialIntelligence.MachineLearning.SupervisedLearning.support_vector_machine import SupportVectorMachine
support_vector_machine = SupportVectorMachine()

support_vector_machine.loadModel('modelo_svm') # carregamento do arquivo .ai com os padrões que queremos aplicar
new_inputs = [[2, 3], [20, 30], [200, 300], [4, 5], [40, 50], [400, 500], [6, 7], [60, 70], [600, 700]]
new_outputs = support_vector_machine.predict(inputs=new_inputs)
print(new_outputs)
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena'], ['unidade'], ['dezena'], ['centena']]
  </code>
</pre>
<br>
<h5>Testando o Modelo antes da Implantação (Fase/Etapa de Teste)</h5>
<p align="justify">
Nós só vimos até agora as etapas de treinamento e predição, mas também podemos utilizar uma etapa intermediária antes da predição final que será implantada. Essa etapa intermediária é a fase/etapa de testes onde iremos passar entradas que sejam diferentes das que foram passadas no treinamento com valores de saída esperados para essas entradas e o método de teste nos retornará um dicionário com o percentual do grau de assertividade e de erro.
</p>
<p align="justify">
Em algoritmos classificativos o retorno será baseado na contagem dos acertos comparada com a contagem dos erros, porém em algoritmos regressivos este método irá retornar o percentual referente ao grau de semelhança dos valores obtidos com os valores esperados.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
{'assertiveness': 1, 'error': 0}
  </code>
</pre>
<br>
<p align="justify">
Como no caso acima acertamos todos os resultados o retorno para o grau de assertividade será de 1 (100% de acertos) e para o grau de erro 0 (0% de erros). Isso quer dizer que o teste para a configuração atual do nosso modelo retornou todos os outputs que nós esperávamos que ele retornasse.
</p>
<h5>Predição Regressiva</h5>
<p align="justify">
Quando temos valores de saída que não serão fixos na predição, ou seja, quando a predição precisar retornar saídas que possam ser diferentes das encontradas no treinamento, nós deveremos utilizar algoritmos de Regressão, pois os algoritmos classificativos não conseguirão apresentar resultados satisfatoriamente precisos. Na regressão o resultado se adapta as entradas, diferente da classificação onde o resultado deverá ser sempre um dos outputs da fase de treinamento. Se precisarmos, por exemplo, prever se uma pessoa está doente ou não, temos apenas duas possibilidades para todas as predições: doente ou saudável. Neste caso temos uma classificação. Porém se precisarmos prever o preço de uma ação que varia com o tempo podendo apresentar resultados atuais totalmente diferentes dos anteriores nós teremos que aplicar uma regressão para aproximar ao máximo as nossas respostas dos preços futuros que poderão não corresponder a nenhum preço do passado. Resumidamente podemos dizer que a classificação emite respostas seletivas por que irá selecionar uma das respostas do treinamento como resultado da predição. Já a regressão emite respostas adaptativas por que não há uma obrigação em se emitir às mesmas respostas do treinamento podendo adaptar os valores de resultado quando necessário.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[4, 6, 8], [10, 12, 14], [16, 18, 20], [22, 24, 26], [28, 30, 32], [34, 36, 38], [40, 42, 44], [46, 48, 50], [52, 54, 56], [58, 60, 62]]
  </code>
</pre>
<br>
<h5>Redes Neurais Artificiais (Artificial Neural Networks)</h5>
<p align="justify">
Redes neurais artificiais diferentemente dos algoritmos anteriores podem aprender de forma generalista assim como os humanos. Por isso é possível usá-las tanto em casos de classificação quanto em casos de regressão com grande eficiência. Porém elas possuem um treinamento relativamente mais lento do que os algoritmos especialistas tradicionais. Confira a seguir a aplicação de uma rede neural em um caso de classificação binária para diferenciar unidades de dezenas.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
[[0], [1], [0], [1], [0], [1], [0], [1]]
  </code>
</pre>
<br>
<p align="justify">
Observe no código abaixo que agora estamos aplicando o mesmo algoritmo de rede neural artificial em um caso de regressão, mostrando que as redes neurais são algoritmos generalistas que podem se adaptar a qualquer conjunto de dados.
</p>
<br>
<pre>
  <code>
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
  </code>
</pre>
<br>
Resultado:
<pre>
  <code>
epoch...............................: 1 - loss: 0.80000000
epoch...............................: 2 - loss: 0.60000000
epoch...............................: 3 - loss: 0.40000000
epoch...............................: 4 - loss: 0.20000000
epoch...............................: 5 - loss: 0.00000000
[[5], [9], [13], [17], [21], [25], [29], [33], [37], [41]]
  </code>
</pre>
<br>
<p align="justify">
Dessa forma você poderá construir algoritmos que aprendam com qualquer tipo de padrão disponível. Algoritmos classificativos para casos de classificação, algoritmos regressivos para casos de regressão e redes neurais para os dois. O nível de assertividade desses algoritmos é relativo, mesmo com uma assertividade de 5% por exemplo o seu algoritmo poderá ser útil se os humanos designados para a mesma tarefa acertam menos de 5% dos resultados, em contrapartida um algoritmo que acerte acima de 90% poderá não ser útil se os humanos designados para essa tarefa acertarem mais de 90% dos resultados. Você deverá sempre avaliar a situação na qual pretende aplicar o seu modelo de IA para comparar com os resultados já conseguidos da empresa ou instituição e concluir se valerá ou não a pena implantá-lo.
</p>
<p align="justify">
Também é importante ressaltar que algoritmos especialistas tendem a ser mais rápidos do que os de redes neurais, então se o seu projeto exigir performance poderá ser mais interessante usar um algoritmo focado somente em classificação ou regressão. Além disso podem haver situações em que o problema do cliente é otimizar as tarefas do trabalho para torná-las o mais rápido possível, neste caso poderá valer a pena implantar um projeto com taxas de erro maiores contanto que executem o mesmo trabalho que os humanos executariam porém em menos tempo.
</p>
