import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import shap
import lime
import lime.lime_tabular
from PIL import Image

data = pd.read_csv('healthcare-dataset-stroke-data.csv')

data = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
data['bmi'].fillna(data['bmi'].mean(), inplace=True)

X = data.drop(columns=['id', 'stroke'])
y = data['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, output_dict=True)

classification_df = pd.DataFrame(classification_rep).transpose()

feature_importance = clf.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importance)[::-1]
top_n = 10

fig_feature_importance = go.Figure(data=[go.Bar(x=feature_names[sorted_idx[:top_n]], y=feature_importance[sorted_idx[:top_n]])])
fig_feature_importance.update_layout(title="Feature Importance (Random Forest)", xaxis_title="Feature", yaxis_title="Importance")

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot
# fig_shap_summary = shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
shap_summary = Image.open('static/images/shap.summary.png')

fig_shap_dependence = shap.dependence_plot("age", shap_values[1], X_test, feature_names=X.columns, interaction_index=None, show=False)

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    mode="classification",
    training_labels=y_train,
    feature_names=list(X.columns),
    discretize_continuous=True
)

sample_data = X_test[0]
explanation = explainer.explain_instance(
    data_row=sample_data,
    predict_fn=clf.predict_proba,
    num_features=len(X.columns)
)

explanation = Image.open('static/images/explanation.png')
summary_plot = Image.open('static/images/summary_plot.png')

def main():

    st. set_page_config(page_title="Stroke Prediction",page_icon="",layout="wide")
    st.title("Healthcare Stroke Prediction App")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.subheader("Dataset")
    st.dataframe(data)
    
    
    st.subheader("Data")
    st.write("X:")
    st.dataframe(X)
    st.write("y:")
    st.dataframe(y)
 
    st.subheader("Classification Report")
    st.table(classification_df)

    st.subheader("Model Performance")
    st.write("Accuracy:", accuracy)

    st.subheader("Feature Importance (Random Forest)")
    st.plotly_chart(fig_feature_importance, use_container_width=True)

    col1, col2 = st.columns(2)


    with col1:
        st.subheader("SHAP Summary Plot")
        st.image(shap_summary)

    with col2:
        st.subheader("SHAP Dependence Plot")
        st.pyplot(fig_shap_dependence)
        
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Explanation with feature")
        st.image(explanation)

    with col2:
        st.subheader("Summary plots")
        st.image(summary_plot)
     
if __name__ == "__main__":
    main()
