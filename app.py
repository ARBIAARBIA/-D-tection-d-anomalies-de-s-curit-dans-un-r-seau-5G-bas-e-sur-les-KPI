import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time

from preprocess import load_and_preprocess_data
from model import train_isolation_forest, predict_anomalies

# ======================================================
# CONFIGURATION PAGE
# ======================================================
st.set_page_config(
    page_title="5G Security Anomaly Detection",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# STYLE GLOBAL - DESIGN MODERNE ET SIMPLE
# ======================================================
st.markdown("""
<style>
    /* Variables de couleurs */
    :root {
        --primary: #2563eb;
        --primary-light: #3b82f6;
        --secondary: #7c3aed;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --dark: #0f172a;
        --dark-light: #1e293b;
        --gray: #64748b;
        --light: #f8fafc;
    }
    
    /* Reset et fond */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Typographie */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    
    /* Cartes modernes */
    .modern-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .modern-card:hover {
        border-color: var(--primary-light);
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
    }
    
    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 20px;
        border-left: 4px solid;
        height: 100%;
    }
    
    .kpi-icon {
        font-size: 28px;
        margin-bottom: 12px;
        display: inline-block;
        padding: 10px;
        border-radius: 12px;
        background: rgba(37, 99, 235, 0.1);
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Boutons et éléments interactifs */
    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
    }
    
    /* Sélecteurs et inputs */
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 12px;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 12px;
        background: var(--dark-light);
    }
    
    /* Séparateurs */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.2) 50%, transparent 100%);
        margin: 32px 0;
    }
    
    /* Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
    }
    
    .status-badge.warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
    }
    
    .status-badge.danger {
        background: rgba(239, 68, 68, 0.1);
        color: var(--danger);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--gray);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# HEADER MODERNE
# ======================================================
st.markdown("""
<div class="fade-in" style="
    margin-bottom: 32px;
    padding: 32px 40px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%);
    border-radius: 24px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    backdrop-filter: blur(10px);
">
    <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 16px;">
        <div style="
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            width: 60px;
            height: 60px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
        ">
            📡
        </div>
        <div>
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                5G Security Anomaly Detection
            </h1>
            <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 1.1rem;">
                Détection d'anomalies réseau avec analyse IA en temps réel
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# CHARGEMENT DONNÉES & MODÈLE
# ======================================================
@st.cache_data(show_spinner=False)
def load_data():
    return load_and_preprocess_data("kpi_5g.csv")

@st.cache_resource(show_spinner=False)
def load_model(X):
    return train_isolation_forest(X)

# Animation de chargement
with st.spinner(" **Analyse des KPI 5G en cours...**"):
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)
    
    df, df_numeric, X_scaled = load_data()
    model = load_model(X_scaled)
    predictions, scores = predict_anomalies(model, X_scaled)

df["anomaly"] = predictions
df["anomaly_score"] = scores

anomalies = df[df["anomaly"] == -1].sort_values("anomaly_score")
normal = df[df["anomaly"] == 1]

# ======================================================
# SIDEBAR MODERNE
# ======================================================
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <h3 style="color: white; margin: 0;">📊 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation avec icônes
    page_options = {
        "🏠 Dashboard": "Vue globale du réseau",
        "📈 Analyse KPI": "Analyse détaillée des indicateurs",
        "🚨 Anomalies": "Détections critiques",
        "⚙️ Paramètres": "Configuration du système"
    }
    
    page = st.radio(
        "Sélectionner une page",
        list(page_options.keys()),
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # État du système
    st.markdown("### 📊 État du système")
    
    rate = len(anomalies) / len(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Échantillons", f"{len(df):,}")
    with col2:
        st.metric("Anomalies", len(anomalies), delta=f"{rate*100:.1f}%")
    
    # Indicateur d'état
    if rate < 0.05:
        st.markdown('<div class="status-badge">✅ Système stable</div>', unsafe_allow_html=True)
        st.progress(rate, text="Risque faible")
    elif rate < 0.15:
        st.markdown('<div class="status-badge warning">⚠️ Surveillance active</div>', unsafe_allow_html=True)
        st.progress(rate, text="Risque modéré")
    else:
        st.markdown('<div class="status-badge danger">🚨 Intervention requise</div>', unsafe_allow_html=True)
        st.progress(rate, text="Risque élevé")
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Dernière mise à jour
    st.markdown("""
    <div style="color: #64748b; font-size: 0.9rem;">
        <div>🕒 Dernière analyse</div>
        <div style="color: white; font-weight: 500;">{}</div>
    </div>
    """.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")), unsafe_allow_html=True)

# ======================================================
# FONCTION KPI CARD MODERNE
# ======================================================
def modern_kpi_card(title, value, icon, color, description=""):
    card = f"""
    <div class="modern-card fade-in">
        <div class="kpi-icon" style="color: {color};">
            {icon}
        </div>
        <h3 style="color: white; margin: 0 0 8px 0; font-size: 2rem;">
            {value}
        </h3>
        <div style="color: #94a3b8; margin-bottom: 8px; font-weight: 500;">
            {title}
        </div>
        <div style="color: #64748b; font-size: 0.9rem;">
            {description}
        </div>
    </div>
    """
    return card

# ======================================================
# PAGE 1 – DASHBOARD
# ======================================================
if page == "🏠 Dashboard":
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h2 style="color: white; margin: 0;">Vue d'ensemble du réseau 5G</h2>
        <p style="color: #94a3b8; margin: 8px 0 0 0;">
            Surveillance temps réel des performances et détection d'anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(modern_kpi_card(
            "Échantillons totaux", 
            f"{len(df):,}", 
            "📊", 
            "#3b82f6",
            "Données analysées"
        ), unsafe_allow_html=True)
    with col2:
        st.markdown(modern_kpi_card(
            "Anomalies détectées", 
            len(anomalies), 
            "🚨", 
            "#ef4444",
            f"{rate*100:.1f}% du trafic"
        ), unsafe_allow_html=True)
    with col3:
        st.markdown(modern_kpi_card(
            "Score moyen", 
            f"{df['anomaly_score'].mean():.2f}", 
            "📈", 
            "#10b981",
            "Score de confiance IA"
        ), unsafe_allow_html=True)
    with col4:
        st.markdown(modern_kpi_card(
            "KPI monitorés", 
            len(df_numeric.columns), 
            "🔍", 
            "#8b5cf6",
            "Indicateurs clés"
        ), unsafe_allow_html=True)
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Graphique principal
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h3 style="color: white; margin: 0;">Analyse temps réel des KPI</h3>
        <p style="color: #94a3b8; margin: 8px 0 0 0;">
            Visualisation des données normales et détection des anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_kpi = st.selectbox(
            "Sélectionner un KPI",
            df_numeric.columns,
            key="kpi_select",
            label_visibility="collapsed"
        )
    
    # Graphique interactif
    fig = go.Figure()
    
    # Ligne normale
    fig.add_trace(go.Scatter(
        x=normal.index,
        y=normal[selected_kpi],
        mode="lines",
        name="Trafic normal",
        line=dict(color="#10b981", width=2),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.1)',
        hovertemplate="<b>Valeur</b>: %{y:.2f}<br><b>Index</b>: %{x}<extra></extra>"
    ))
    
    # Points d'anomalies
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies[selected_kpi],
            mode="markers",
            name="Anomalies",
            marker=dict(
                color="#ef4444",
                size=8,
                symbol="diamond",
                line=dict(width=1, color="white")
            ),
            hovertemplate="<b>ANOMALIE</b><br>Valeur: %{y:.2f}<br>Score: %{customdata}<extra></extra>",
            customdata=anomalies["anomaly_score"].round(3)
        ))
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(148, 163, 184, 0.2)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.1)',
            title=dict(text="Index", font=dict(color="#94a3b8"))
        ),
        yaxis=dict(
            gridcolor='rgba(148, 163, 184, 0.1)',
            title=dict(text="Valeur KPI", font=dict(color="#94a3b8"))
        ),
        margin=dict(l=50, r=30, t=30, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})

# ======================================================
# PAGE 2 – ANALYSE KPI
# ======================================================
elif page == "📈 Analyse KPI":
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h2 style="color: white; margin: 0;">Analyse statistique avancée</h2>
        <p style="color: #94a3b8; margin: 8px 0 0 0;">
        Distribution comparative des KPI normaux vs anormaux
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_kpi = st.selectbox(
            "Choisir un KPI à analyser",
            df_numeric.columns,
            key="analysis_kpi"
        )
    
    with col2:
        st.markdown("""
        <div class="modern-card" style="height: auto;">
            <h4 style="color: white; margin: 0 0 12px 0;">📊 Statistiques</h4>
            <div style="color: #94a3b8;">
                Sélectionnez un KPI pour visualiser sa distribution statistique et comparer les valeurs normales avec les anomalies détectées.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique boxplot
    fig_box = go.Figure()
    
    fig_box.add_trace(go.Box(
        y=normal[selected_kpi],
        name="Normal",
        marker_color="#10b981",
        boxmean='sd'
    ))
    
    fig_box.add_trace(go.Box(
        y=anomalies[selected_kpi],
        name="Anomalies",
        marker_color="#ef4444",
        boxmean='sd'
    ))
    
    fig_box.update_layout(
        title=f"Distribution de {selected_kpi}",
        template="plotly_dark",
        height=400,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistiques détaillées
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: white; margin: 0 0 16px 0;">📈 Statistiques descriptives</h4>
        """, unsafe_allow_html=True)
        
        stats_normal = normal[selected_kpi].describe()
        stats_anomalies = anomalies[selected_kpi].describe()
        
        for stat in ['mean', 'std', 'min', '50%', 'max']:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                st.caption(stat.upper())
            with col_stat2:
                st.metric("Normal", f"{stats_normal[stat]:.2f}", label_visibility="collapsed")
            with col_stat3:
                st.metric("Anomalies", f"{stats_anomalies[stat]:.2f}", label_visibility="collapsed")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: white; margin: 0 0 16px 0;">📋 Vue des données</h4>
            <div style="max-height: 300px; overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        # Aperçu des données
        preview_data = pd.DataFrame({
            'Type': ['Normal'] * min(5, len(normal)) + ['Anomalie'] * min(5, len(anomalies)),
            'Valeur': list(normal[selected_kpi].head(5).values) + list(anomalies[selected_kpi].head(5).values),
            'Score': [None] * min(5, len(normal)) + list(anomalies['anomaly_score'].head(5).values)
        })
        
        st.dataframe(
            preview_data,
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# ======================================================
# PAGE 3 – ANOMALIES
# ======================================================
elif page == "🚨 Anomalies":
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h2 style="color: white; margin: 0;">Anomalies détectées</h2>
        <p style="color: #94a3b8; margin: 8px 0 0 0;">
        Liste des détections classées par niveau de criticité (score IA)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.slider(
            "Score minimum",
            min_value=float(df['anomaly_score'].min()),
            max_value=float(df['anomaly_score'].max()),
            value=float(df['anomaly_score'].quantile(0.8)),
            step=0.01
        )
    
    with col2:
        show_count = st.select_slider(
            "Nombre d'anomalies à afficher",
            options=[10, 20, 50, 100, "Toutes"],
            value=20
        )
    
    with col3:
        severity = st.multiselect(
            "Niveau de sévérité",
            ["Faible", "Moyen", "Élevé"],
            default=["Moyen", "Élevé"]
        )
    
    # Filtrage des anomalies
    filtered_anomalies = anomalies[anomalies['anomaly_score'] >= min_score]
    
    if show_count != "Toutes":
        filtered_anomalies = filtered_anomalies.head(show_count)
    
    # Tableau des anomalies
    st.markdown("""
    <div class="modern-card" style="overflow: hidden;">
    """, unsafe_allow_html=True)
    
    # En-tête avec compteur
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown(f"### {len(filtered_anomalies)} anomalies critiques")
    with col_header2:
        if len(filtered_anomalies) > 0:
            avg_score = filtered_anomalies['anomaly_score'].mean()
            st.metric("Score moyen", f"{avg_score:.3f}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tableau interactif
    if len(filtered_anomalies) > 0:
        display_cols = ["anomaly_score"] + df_numeric.columns.tolist()[:6]
        
        # Style conditionnel pour le tableau
        styled_df = filtered_anomalies[display_cols].copy()
        
        # Formater les scores
        styled_df['anomaly_score'] = styled_df['anomaly_score'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400,
            column_config={
                "anomaly_score": st.column_config.ProgressColumn(
                    "Score d'anomalie",
                    help="Score IA de détection d'anomalie",
                    format="%.3f",
                    min_value=float(df['anomaly_score'].min()),
                    max_value=float(df['anomaly_score'].max())
                )
            }
        )
        
        # Téléchargement des anomalies
        csv = filtered_anomalies.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Exporter les anomalies",
            data=csv,
            file_name=f"anomalies_5g_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px;">
            <div style="font-size: 48px; margin-bottom: 20px;">✅</div>
            <h3 style="color: white;">Aucune anomalie critique détectée</h3>
            <p style="color: #94a3b8;">Le système fonctionne normalement</p>
        </div>
        """, unsafe_allow_html=True)

# ======================================================
# PAGE 4 – PARAMÈTRES
# ======================================================
elif page == "⚙️ Paramètres":
    st.markdown("""
    <div style="margin-bottom: 32px;">
        <h2 style="color: white; margin: 0;">Configuration du système</h2>
        <p style="color: #94a3b8; margin: 8px 0 0 0;">
        Personnalisez les paramètres de détection d'anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: white; margin: 0 0 20px 0;">⚙️ Paramètres du modèle</h4>
        """, unsafe_allow_html=True)
        
        contamination = st.slider(
            "Taux de contamination estimé",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Proportion attendue d'anomalies dans les données"
        )
        
        n_estimators = st.selectbox(
            "Nombre d'arbres",
            [50, 100, 200, 500],
            index=1,
            help="Nombre d'arbres dans la forêt d'isolation"
        )
        
        max_features = st.slider(
            "Nombre maximum de features",
            min_value=1,
            max_value=len(df_numeric.columns),
            value=min(10, len(df_numeric.columns)),
            step=1
        )
        
        st.button("🔄 Réentraîner le modèle", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="modern-card">
            <h4 style="color: white; margin: 0 0 20px 0;">📊 Préférences d'affichage</h4>
        """, unsafe_allow_html=True)
        
        refresh_rate = st.selectbox(
            "Fréquence de rafraîchissement",
            ["Temps réel", "30 secondes", "1 minute", "5 minutes", "Manuel"],
            index=0
        )
        
        theme = st.selectbox(
            "Thème de l'interface",
            ["Sombre (par défaut)", "Clair", "Auto"],
            index=0
        )
        
        notifications = st.multiselect(
            "Notifications",
            ["Anomalies critiques", "Dépassements seuils", "Rapports quotidiens", "Alertes système"],
            default=["Anomalies critiques"]
        )
        
        st.button("💾 Enregistrer les préférences", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# FOOTER MODERNE
# ======================================================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div style="
    padding: 24px 0;
    text-align: center;
    color: #64748b;
    font-size: 0.9rem;
">
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 12px;
    ">
        <div style="
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        ">
            🔒
        </div>
        <div style="text-align: left;">
            <div style="color: white; font-weight: 500; font-size: 1rem;">
                Système de Sécurité 5G
            </div>
            <div>
                 Détection d'anomalies réseau
            </div>
        </div>
    </div>
    <div style="margin-top: 16px; color: #475569;">
        IA utilisée : <b>Isolation Forest</b> • Interface SOC 5G • v2.1.0
    </div>
</div>
""", unsafe_allow_html=True)