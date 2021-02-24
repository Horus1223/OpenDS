import pandas as pd
from sklearn.tree import DecisionTreeClassifier #Импортируем классификатор, который будем использовать 
from sklearn.model_selection import train_test_split #Для разделения данных
from sklearn.tree import export_graphviz
from IPython.display import Image
from io import StringIO
import pydotplus


#Для вычисления точности предсказаний

pima = pd.read_excel('CardiologyCategorical.xlsx', sheet_name='CardiologyCategorical')
pima.head()

X=pima.loc[:, 'age':'thal'] #На чем обучаемся
y=pima['class'] #Целевой столбец

#разделяем датасет на выборку train\test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #30% test, 70% train

#Создаем классификатор
tree=DecisionTreeClassifier(criterion='entropy', max_depth=304)

#Обучаем
tree=tree.fit(X_train, y_train)

#Предсказываем
y_pred=tree.predict(X_test)

#Доля правильных ответов
print("Accuracy on training set: {:.3f}".format(tree.score(X_train,y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_train,y_train)))

#оценка точности, матрица путаницы и отчет о классификации
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

#визуализация дерева решений
dot_data = StringIO()
export_graphviz(tree, out_file= dot_data,
                feature_names=X.columns,
                filled=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('cardiology.png')
Image(graph.create_png())


