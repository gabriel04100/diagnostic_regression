import streamlit as st
#import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import joblib

st.set_page_config(
    page_title="Diagnostic regression",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_data():
    x_test = st.file_uploader("Sélectionnez le fichier X_test", type="csv")
    y_test = st.file_uploader("Sélectionnez le fichier y_test", type="csv")
    
    if x_test is not None and y_test is not None:
        x_test_df = pd.read_csv(x_test)
        y_test_df = pd.read_csv(y_test)
        return x_test_df, y_test_df
    
    return None, None
def download_dataframe(df):
    csv = df.to_csv(index=False)
    return csv

def load_model():
    models = st.file_uploader("Sélectionnez le(s) fichier(s) modèle(s)", accept_multiple_files=True)
    loaded_models = []
    
    if models is not None:
        for model in models:
            loaded_model = joblib.load(model)
            loaded_models.append(loaded_model)
    
    return loaded_models

def evaluate_predictions(y_test, predictions):
    mae = np.round(mean_absolute_error(y_test, predictions),3)
    rmse=np.round(np.sqrt(mean_squared_error(y_test,predictions)),3)
    r2 = np.round(r2_score(y_test, predictions),3)
    mx=np.max(np.abs(y_test['transfertRealValue']-predictions))
    mn=np.min(np.abs(y_test['transfertRealValue']-predictions))
    
    return mae,rmse, r2,mx,mn

def main():
    
    x_test, y_test = load_data()
    
    models = load_model()
    st.write("Description du/des modèle(s)")
    st.write(models)
    if x_test is not None and y_test is not None and len(models) > 0:
        st.subheader("Données chargées avec succès !")
        # Sélectionneur de valeur minimale pour y_test
        st.write(f"valeur minimale donnée test : {y_test.min()}")
        st.write(f"valeur max donnée test : {y_test.max()}")

        # Sélectionner les colonnes ayant moins de dix valeurs différentes
        categorical_columns = [col for col in x_test.columns if x_test[col].nunique() < 10]
        
        # Sélectionneur de colonne catégorique pour comparer les résultats
        category_column = st.selectbox("Filtrer la colonne catégorique pour comparer les résultats ", categorical_columns + ["Toutes les colonnes"], index=len(categorical_columns))
        
        if category_column == "Toutes les colonnes":
            filtered_x_test = x_test.copy()
            filtered_y_test = y_test.copy()
        else:
            unique_categories = x_test[category_column].unique()
            selected_category = st.selectbox("Sélectionnez une catégorie", unique_categories, index=0)
            
            filtered_x_test = x_test[x_test[category_column] == selected_category]
            filtered_y_test = y_test.loc[filtered_x_test.index]

        # Filtrer les valeurs de test supérieures au seuil
        threshold_min = st.number_input("Sélectionnez une valeur minimale pour y_test", value=0.0)
        threshold_max = st.number_input("Sélectionnez une valeur maximale pour y_test", value=200.0)

        filtered_indices = filtered_y_test[(filtered_y_test['transfertRealValue'] > threshold_min) & (filtered_y_test['transfertRealValue'] <threshold_max)].index
        filtered_x_test = filtered_x_test.loc[filtered_indices]
        filtered_y_test = filtered_y_test.loc[filtered_indices]
        st.write("shape :",filtered_x_test.shape)
        
        for i, model in enumerate(models):
            st.write(f"Modèle {i+1}")
            
            # Effectuer les prédictions sur l'ensemble x_test
            predictions = model.predict(filtered_x_test)
            predictions=np.clip(predictions, 0.2, None)
                
            # Évaluation des prédictions
            mae,rmse, r2,mx,mn = evaluate_predictions(filtered_y_test, predictions)
            
            st.write("MAE :", mae)
            st.write("RMSE :", rmse)
            st.write("R2: ",r2)
            st.write("erreur maximum : ",mx)
            st.write("erreur minimum ",mn)
            errors=filtered_y_test['transfertRealValue']-predictions
            complet = filtered_x_test[['Attaquant', 'Gardien', 'Milieu','Defense']].copy()
            roles = filtered_x_test[['Attaquant', 'Milieu', 'Defense', 'Gardien']].idxmax(axis=1)
            # Enlevez le préfixe des noms de colonnes pour obtenir les rôles
            roles = roles.str.split('_', expand=True)[0]
            complet['role']=roles
            complet=complet.drop(['Attaquant', 'Gardien', 'Milieu','Defense'],axis=1)

            complet["vrai_prix"]=filtered_y_test['transfertRealValue']
            complet["prediction"]=predictions
            complet["erreur"]=np.abs(errors)
            st.write(complet)
            csv = download_dataframe(complet)
            st.download_button(label="Télécharger le DataFrame", data=csv, file_name="dataframe.csv", mime="text/csv")

            #st.write(pd.DataFrame({"vrai_prix":filtered_y_test['transfertRealValue'],"prediction":predictions,"erreur":np.abs(errors)}))
            
            # Visualisation des prédictions vs les vraies valeurs
            # Tracer le graphique des prédictions par rapport aux valeurs réelles
            
            fig = px.scatter(data_frame=complet,
                             title="Comparaison vrai prix et prédictions par rôle",
                x="vrai_prix",
                y="prediction",
                color="role",
                size_max=60,
            )
            st.plotly_chart(fig)
            feature_importances = model[1].feature_importances_
            # Obtenir les indices triés des caractéristiques les plus importantes
            top_indices = feature_importances.argsort()[::-1][:10]  # Les 10 indices des caractéristiques les plus importantes
            s=model['selector']
            result=[s.get_feature_names_out()]
            result= [str(x).split('__')[1] for x in result[0]]
            feature_names=result
            # Obtenir les noms des caractéristiques correspondantes
            top_features = [feature_names[i] for i in top_indices]  # Supposons que vous ayez les noms des caractéristiques dans une liste feature_names

            # Créer un DataFrame avec les noms des caractéristiques et leurs importances
            df_importances = pd.DataFrame({'Features': top_features, 'Importance': feature_importances[top_indices]})

            # Créer un graphique à barres pour représenter l'importance des caractéristiques
            fig_imp = px.bar(df_importances, x='Features', y='Importance', color='Importance',
                        labels={'Features': 'Caractéristiques', 'Importance': 'Importance'},
                        title='Importance des caractéristiques - Random Forest',
                        color_continuous_scale='Blues')
            st.plotly_chart(fig_imp)
   
    else:
        st.write("Veuillez charger les fichiers X_test, y_test et les fichiers modèle.")
        
if __name__ == "__main__":
    main()
