# Remplacez ce bloc de code dans votre fichier streamlit.py

# 3. Filtrage des données négatives
st.markdown('<div class="sub-header">2. Analyse des retours négatifs</div>', unsafe_allow_html=True)

# Permettre à l'utilisateur de sélectionner les valeurs négatives
all_humeurs = df['Humeur'].unique()

# Correction : vérifiez que chaque valeur est une chaîne avant d'appliquer lower()
default_negatives = []
for h in all_humeurs:
    # Vérifier si h est une chaîne (str) et non pas un nombre (float/int)
    if isinstance(h, str) and ('insatisfait' in h.lower() or 'négativ' in h.lower()):
        default_negatives.append(h)

negative_humeurs = st.multiselect(
    "Sélectionnez les humeurs à considérer comme négatives",
    all_humeurs,
    default=default_negatives
)

df_negatif = df[df['Humeur'].isin(negative_humeurs)]

st.markdown(f"""
<div class="highlight">
    <p>Nombre d'entrées négatives sélectionnées : {df_negatif.shape[0]}</p>
    <p>Pourcentage du total : {(df_negatif.shape[0] / df.shape[0] * 100):.1f}%</p>
</div>
""", unsafe_allow_html=True)
