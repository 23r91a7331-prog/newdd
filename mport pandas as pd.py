mport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree # Import DecisionTreeClassifier and plot_tree

# create dataset
data={
    'age':[22,25,47,52,46,56,55,60],
    'income':[25000,30000,50000,60000,45000,65000,70000,8000],
    'student':[1,1,0,0,0,1,0,1],
    'buys_computer':[0,0,1,1,1,1,1,0]
}
df=pd.DataFrame(data) # Corrected to DataFrame

# split features and target
x=df.drop('buys_computer',axis=1)
y=df['buys_computer']

# train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42)

# train decision tree model
model=DecisionTreeClassifier(criterion='entropy') # Corrected class name and criterion
model.fit(x_train,y_train) # Added missing parenthesis

# make predictions
y_pred=model.predict(x_test)

# calculate accuracy
accuracy=accuracy_score(y_test,y_pred)
print('accuracy of decision tree:',accuracy)

# plot the decision tree
plt.figure(figsize=(10,6))
plot_tree( # Corrected function name
    model,
    feature_names=x.columns,
    class_names=['no','yes'],
    filled=True
)
plt.title('decision tree')
plt.show()