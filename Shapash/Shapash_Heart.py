from shapash.explainer.smart_explainer import SmartExplainer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

url = "C:\\Users\\deep\\PycharmProjects\\DiseasePrediction\\res\\heart_data.csv"
heart = pd.read_csv(url)

# Ordinal feature encoding
df = heart.copy()
encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    del dummy

# Separating X and y
X = df.drop('HeartDisease', axis=1)
Y = df['HeartDisease']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=200).fit(X_train, y_train)

xpl = SmartExplainer(model=regressor)
xpl.compile(
    x=X_test,

)
app = xpl.run_app(title_story='Tips Dataset')
