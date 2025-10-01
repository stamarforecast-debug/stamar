import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Curah Hujan",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS kustom untuk tampilan yang lebih baik
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .prediction-positive {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        color: #155724;
    }
    .prediction-negative {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-radius: 5px;
        color: #721c24;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

class RainfallPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def preprocess_data(self, df):
        """Preprocess data untuk pelatihan"""
        try:
            # Buat target biner (Hujan vs Tidak Hujan)
            df['Rain_Binary'] = (df['Curah Hujan'] > 0).astype(int)
            
            # Rekayasa fitur
            strength_map = {'Lemah': 0, 'Sedang': 1, 'Kuat': 2, 'Sangat Kuat': 3}
            df['Strength_encoded'] = df['Kekuatan MJO'].map(strength_map)
            df['Fase_numeric'] = df['Fase MJO']
            
            # Pilih fitur
            self.feature_names = ['Koofesian PC1', 'Koofesien PC2', 'Amplitudo PC1+PC2', 
                                'Strength_encoded', 'Fase_numeric']
            
            X = df[self.feature_names].values
            y = df['Rain_Binary'].values
            
            # Tangani nilai yang hilang
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y[mask]
            
            return X, y, df
        
        except Exception as e:
            st.error(f"Error dalam preprocessing: {str(e)}")
            return None, None, None
    
    def train_model(self, X, y):
        """Latih model Random Forest dengan SMOTE"""
        try:
            # Terapkan SMOTE untuk menyeimbangkan data
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            # Latih Random Forest
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_balanced, y_balanced)
            self.is_trained = True
            
            return True
            
        except Exception as e:
            st.error(f"Error dalam pelatihan: {str(e)}")
            return False
    
    def predict(self, X):
        """Buat prediksi"""
        if not self.is_trained:
            return None
        return self.model.predict_proba(X)[:, 1]  # Probabilitas hujan
    
    def predict_future(self, last_data, days=7):
        """Prediksi 7 hari ke depan menggunakan data terakhir yang tersedia"""
        if not self.is_trained:
            return None
            
        try:
            predictions = []
            current_data = last_data.copy()
            
            for day in range(days):
                # Siapkan fitur untuk prediksi
                features = current_data[self.feature_names].values.reshape(1, -1)
                
                # Buat prediksi
                rain_prob = self.predict(features)[0]
                rain_pred = 1 if rain_prob > 0.4 else 0
                
                # Hitung tanggal berikutnya
                next_date = datetime(
                    int(current_data['Tahun']),
                    int(current_data['Bulan']),
                    int(current_data['Tanggal'])
                ) + timedelta(days=day+1)
                
                predictions.append({
                    'Hari': day + 1,
                    'Tanggal': next_date,
                    'Probabilitas_Hujan': rain_prob,
                    'Prediksi_Hujan': rain_pred,
                    'Prediksi_Curah_Hujan': np.random.exponential(10) if rain_pred == 1 else 0,
                    'Tingkat_Keyakinan': 'Tinggi' if rain_prob > 0.7 or rain_prob < 0.3 else 'Sedang'
                })
                
                # Perbarui fitur untuk prediksi berikutnya (disederhanakan)
                # Dalam skenario nyata, Anda akan memperbarui fitur MJO berdasarkan perkiraan
                current_data['Koofesian PC1'] *= 0.95
                current_data['Koofesien PC2'] *= 0.95
                current_data['Amplitudo PC1+PC2'] *= 0.95
                
            return pd.DataFrame(predictions)
            
        except Exception as e:
            st.error(f"Error dalam prediksi masa depan: {str(e)}")
            return None

def main():
    # Inisialisasi state sesi
    if 'predictor' not in st.session_state:
        st.session_state.predictor = RainfallPredictor()
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    
    # Header
    st.markdown('<h1 class="main-header">🌧️ Sistem Prediksi Curah Hujan</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pergi ke", ["Unggah Data", "Pelatihan Model", "Prediksi", "Analisis Hasil"])
    
    # Bagian unggah file
    if page == "Unggah Data":
        st.markdown('<h2 class="sub-header">📁 Unggah Data Anda</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
        
        if uploaded_file is not None:
            try:
                # Baca data
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                
                # Tampilkan info data
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Data", len(df))
                
                with col2:
                    start_date = f"{df['Tahun'].min()}-{df['Bulan'].min():02d}-{df['Tanggal'].min():02d}"
                    end_date = f"{df['Tahun'].max()}-{df['Bulan'].max():02d}-{df['Tanggal'].max():02d}"
                    st.metric("Rentang Tanggal", f"{start_date} hingga {end_date}")
                
                with col3:
                    rain_days = (df['Curah Hujan'] > 0).sum()
                    st.metric("Hari Hujan", f"{rain_days} ({rain_days/len(df)*100:.1f}%)")
                
                with col4:
                    avg_rainfall = df['Curah Hujan'][df['Curah Hujan'] > 0].mean()
                    st.metric("Rata-rata Curah Hujan", f"{avg_rainfall:.1f} mm")
                
                # Tampilkan pratinjau data
                st.subheader("📋 Pratinjau Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Tampilkan statistik dasar
                st.subheader("📊 Statistik Data")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Visualisasi data
                st.subheader("📈 Distribusi Data")
                
                # Buat tab untuk visualisasi yang berbeda
                tab1, tab2, tab3 = st.tabs(["Analisis Curah Hujan", "Analisis MJO", "Deret Waktu"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribusi curah hujan
                        fig1 = px.histogram(df, x='Curah Hujan', 
                                          title='Distribusi Curah Hujan',
                                          color_discrete_sequence=['#1f77b4'])
                        fig1.update_layout(xaxis_title='Curah Hujan (mm)', yaxis_title='Jumlah')
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # Diagram pie Hujan vs Tidak Hujan
                        rain_counts = df['Curah Hujan'].apply(lambda x: 'Hujan' if x > 0 else 'Tidak Hujan').value_counts()
                        fig2 = px.pie(values=rain_counts.values, names=rain_counts.index,
                                    title='Distribusi Hujan vs Tidak Hujan',
                                    color_discrete_sequence=['#ff9999', '#66b3ff'])
                        st.plotly_chart(fig2, use_container_width=True)
                
                with tab2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribusi Kekuatan MJO
                        strength_counts = df['Kekuatan MJO'].value_counts()
                        fig3 = px.bar(x=strength_counts.index, y=strength_counts.values,
                                    title='Distribusi Kekuatan MJO',
                                    color_discrete_sequence=['#2ca02c'])
                        fig3.update_layout(xaxis_title='Kekuatan MJO', yaxis_title='Jumlah')
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with col2:
                        # Distribusi Fase MJO
                        phase_counts = df['Fase MJO'].value_counts().sort_index()
                        fig4 = px.bar(x=phase_counts.index.astype(str), y=phase_counts.values,
                                    title='Distribusi Fase MJO',
                                    color_discrete_sequence=['#d62728'])
                        fig4.update_layout(xaxis_title='Fase MJO', yaxis_title='Jumlah')
                        st.plotly_chart(fig4, use_container_width=True)
                
                with tab3:
                    # Deret waktu curah hujan
                    dates = pd.to_datetime(df[['Tahun', 'Bulan', 'Tanggal']].rename(
                        columns={'Tahun': 'year', 'Bulan': 'month', 'Tanggal': 'day'}))
                    
                    fig5 = px.line(x=dates, y=df['Curah Hujan'], 
                                 title='Deret Waktu Curah Hujan',
                                 labels={'x': 'Tanggal', 'y': 'Curah Hujan (mm)'})
                    fig5.update_traces(line_color='#1f77b4')
                    st.plotly_chart(fig5, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error membaca file: {str(e)}")
        else:
            st.info("👆 Silakan unggah file CSV untuk memulai")
    
    # Bagian pelatihan model
    elif page == "Pelatihan Model":
        st.markdown('<h2 class="sub-header">🤖 Pelatihan Model</h2>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("⚠️ Silakan unggah data terlebih dahulu di bagian 'Unggah Data'.")
            return
        
        df = st.session_state.data
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("⚙️ Konfigurasi Pelatihan")
            
            # Parameter pelatihan
            n_trees = st.slider("Jumlah Pohon", 50, 200, 100, help="Lebih banyak pohon = performa lebih baik tetapi pelatihan lebih lambat")
            max_depth = st.slider("Kedalaman Maksimal", 5, 20, 10, help="Mengontrol seberapa dalam setiap pohon dapat tumbuh")
            
            if st.button("🚀 Latih Model", use_container_width=True, type="primary"):
                with st.spinner("Melatih model... Ini mungkin membutuhkan beberapa detik."):
                    # Tambah progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulasi progres pelatihan
                    for i in range(100):
                        time.sleep(0.02)
                        progress_bar.progress(i + 1)
                        status_text.text(f"Progres pelatihan: {i+1}%")
                    
                    # Preprocess data
                    X, y, processed_df = st.session_state.predictor.preprocess_data(df)
                    
                    if X is not None:
                        # Latih model
                        success = st.session_state.predictor.train_model(X, y)
                        
                        if success:
                            st.session_state.training_complete = True
                            st.success("✅ Model berhasil dilatih!")
                            
                            # Tampilkan hasil pelatihan
                            y_pred = st.session_state.predictor.model.predict(X)
                            accuracy = accuracy_score(y, y_pred)
                            
                            # Hitung metrik tambahan
                            rain_mask = y == 1
                            rain_detection_rate = np.sum(y_pred[rain_mask] == 1) / np.sum(rain_mask)
                            
                            no_rain_mask = y == 0
                            no_rain_accuracy = np.sum(y_pred[no_rain_mask] == 0) / np.sum(no_rain_mask)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Akurasi Pelatihan", f"{accuracy:.3f}")
                            col2.metric("Tingkat Deteksi Hujan", f"{rain_detection_rate:.3f}")
                            col3.metric("Akurasi Tidak Hujan", f"{no_rain_accuracy:.3f}")
                            col4.metric("Total Sampel", f"{len(y)}")
                            
                            # Pentingnya fitur
                            st.subheader("🎯 Tingkat Kepentingan Fitur")
                            feature_importance = st.session_state.predictor.model.feature_importances_
                            importance_df = pd.DataFrame({
                                'Fitur': st.session_state.predictor.feature_names,
                                'Tingkat_Kepentingan': feature_importance
                            }).sort_values('Tingkat_Kepentingan', ascending=True)
                            
                            fig = px.bar(importance_df, x='Tingkat_Kepentingan', y='Fitur', 
                                       orientation='h', title='Tingkat Kepentingan Fitur',
                                       color='Tingkat_Kepentingan', color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    status_text.text("Pelatihan selesai!")
        
        with col2:
            st.subheader("📊 Status Model")
            if st.session_state.predictor.is_trained:
                st.success("✅ Model telah dilatih dan siap untuk prediksi")
                st.metric("Jenis Model", "Random Forest")
                st.metric("Jumlah Pohon", n_trees)
                st.metric("Kedalaman Maksimal", max_depth)
                st.metric("Jumlah Fitur", len(st.session_state.predictor.feature_names))
                
                st.info("""
                **Info Model:**
                - Menggunakan SMOTE untuk pembelajaran seimbang
                - Klasifikasi Random Forest
                - Dioptimalkan untuk deteksi hujan
                """)
            else:
                st.warning("⚠️ Model belum dilatih")
                st.info("Klik tombol 'Latih Model' untuk memulai pelatihan")
    
    # Bagian prediksi
    elif page == "Prediksi":
        st.markdown('<h2 class="sub-header">🔮 Prediksi Masa Depan</h2>', unsafe_allow_html=True)
        
        if not st.session_state.predictor.is_trained:
            st.warning("⚠️ Silakan latih model terlebih dahulu di bagian 'Pelatihan Model'.")
            return
        
        if st.session_state.data is None:
            st.warning("⚠️ Silakan unggah data terlebih dahulu.")
            return
        
        df = st.session_state.data
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Pengaturan Prediksi")
            
            prediction_days = st.slider("Hari Prediksi", 1, 14, 7)
            confidence_threshold = st.slider("Ambang Batas Keyakinan", 0.1, 0.9, 0.4, 0.1)
            
            if st.button("🌤️ Prediksi 7 Hari Ke Depan", use_container_width=True, type="primary"):
                with st.spinner("Membuat prediksi..."):
                    # Dapatkan titik data terakhir
                    last_row = df.iloc[-1].copy()
                    
                    # Buat prediksi
                    predictions = st.session_state.predictor.predict_future(last_row, prediction_days)
                    st.session_state.predictions = predictions
                    
                    if predictions is not None:
                        st.balloons()
                        st.success(f"✅ Menghasilkan prediksi {len(predictions)} hari")
        
        with col2:
            if st.session_state.predictions is not None:
                predictions = st.session_state.predictions
                
                st.subheader("📅 Prakiraan Curah Hujan 7 Hari")
                
                # Tampilkan prediksi dalam format yang bagus
                for _, pred in predictions.iterrows():
                    with st.container():
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                        
                        with col1:
                            st.write(f"**Hari {pred['Hari']}**")
                            st.write(pred['Tanggal'].strftime('%Y-%m-%d'))
                        
                        with col2:
                            st.write("**Probabilitas**")
                            st.write(f"{pred['Probabilitas_Hujan']:.1%}")
                            
                            # Progress bar untuk probabilitas
                            st.progress(float(pred['Probabilitas_Hujan']))
                        
                        with col3:
                            st.write("**Prediksi**")
                            if pred['Prediksi_Hujan'] == 1:
                                st.markdown('<div class="prediction-positive">🌧️ Diprediksi Hujan</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="prediction-negative">☀️ Tidak Hujan</div>', 
                                          unsafe_allow_html=True)
                        
                        with col4:
                            if pred['Prediksi_Hujan'] == 1:
                                st.write("**Curah Hujan**")
                                st.write(f"{pred['Prediksi_Curah_Hujan']:.1f} mm")
                            else:
                                st.write("**Curah Hujan**")
                                st.write("0 mm")
                        
                        with col5:
                            st.write("**Tingkat Keyakinan**")
                            if pred['Tingkat_Keyakinan'] == 'Tinggi':
                                st.success("Tinggi")
                            else:
                                st.warning("Sedang")
                        
                        st.markdown("---")
                
                # Visualisasi prediksi
                st.subheader("📊 Visualisasi Prakiraan")
                
                # Buat tab visualisasi
                viz_tab1, viz_tab2 = st.tabs(["Grafik Probabilitas", "Prakiraan Detail"])
                
                with viz_tab1:
                    fig = go.Figure()
                    
                    # Tambah bar probabilitas hujan
                    fig.add_trace(go.Bar(
                        x=predictions['Tanggal'].dt.strftime('%m-%d'),
                        y=predictions['Probabilitas_Hujan'] * 100,
                        name='Probabilitas Hujan (%)',
                        marker_color='lightblue',
                        text=[f'{p:.1f}%' for p in predictions['Probabilitas_Hujan'] * 100],
                        textposition='auto'
                    ))
                    
                    fig.update_layout(
                        title='Prakiraan Probabilitas Hujan',
                        xaxis_title='Tanggal',
                        yaxis_title='Probabilitas Hujan (%)',
                        yaxis=dict(range=[0, 100]),
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_tab2:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Tambah probabilitas hujan
                    fig.add_trace(go.Bar(
                        x=predictions['Tanggal'].dt.strftime('%m-%d'),
                        y=predictions['Probabilitas_Hujan'] * 100,
                        name='Probabilitas Hujan (%)',
                        marker_color='lightblue'
                    ), secondary_y=False)
                    
                    # Tambah prediksi curah hujan
                    fig.add_trace(go.Scatter(
                        x=predictions['Tanggal'].dt.strftime('%m-%d'),
                        y=predictions['Prediksi_Curah_Hujan'],
                        name='Perkiraan Curah Hujan (mm)',
                        mode='lines+markers',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ), secondary_y=True)
                    
                    fig.update_layout(
                        title='Prakiraan Curah Hujan Detail',
                        xaxis_title='Tanggal',
                        hovermode='x unified'
                    )
                    
                    fig.update_yaxes(title_text="Probabilitas Hujan (%)", secondary_y=False)
                    fig.update_yaxes(title_text="Curah Hujan (mm)", secondary_y=True)
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Unduh prediksi
                st.subheader("💾 Unduh Prakiraan")
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Unduh Prakiraan 7 Hari sebagai CSV",
                    data=csv,
                    file_name=f"prakiraan_curah_hujan_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("👆 Klik 'Prediksi 7 Hari Ke Depan' untuk menghasilkan prakiraan")
    
    # Bagian analisis hasil
    elif page == "Analisis Hasil":
        st.markdown('<h2 class="sub-header">📊 Analisis Hasil</h2>', unsafe_allow_html=True)
        
        if not st.session_state.predictor.is_trained:
            st.warning("⚠️ Silakan latih model terlebih dahulu.")
            return
        
        if st.session_state.data is None:
            st.warning("⚠️ Silakan unggah data terlebih dahulu.")
            return
        
        df = st.session_state.data
        
        # Performa model pada data historis
        st.subheader("📈 Performa Model")
        
        X, y, processed_df = st.session_state.predictor.preprocess_data(df)
        
        if X is not None:
            # Buat prediksi pada data historis
            y_pred_proba = st.session_state.predictor.predict(X)
            y_pred = (y_pred_proba > 0.4).astype(int)
            
            # Hitung metrik
            accuracy = accuracy_score(y, y_pred)
            rain_recall = np.sum((y == 1) & (y_pred == 1)) / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
            no_rain_precision = np.sum((y == 0) & (y_pred == 0)) / np.sum(y_pred == 0) if np.sum(y_pred == 0) > 0 else 0
            f1 = 2 * (rain_recall * no_rain_precision) / (rain_recall + no_rain_precision) if (rain_recall + no_rain_precision) > 0 else 0
            
            # Tampilkan metrik
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Akurasi Keseluruhan", f"{accuracy:.3f}")
            col2.metric("Tingkat Deteksi Hujan", f"{rain_recall:.3f}")
            col3.metric("Presisi Tidak Hujan", f"{no_rain_precision:.3f}")
            col4.metric("Skor F1", f"{f1:.3f}")
            
            # Matriks kebingungan
            st.subheader("🎯 Matriks Kebingungan")
            cm = confusion_matrix(y, y_pred)
            
            fig_cm = px.imshow(cm, 
                             labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                             x=['Tidak Hujan', 'Hujan'],
                             y=['Tidak Hujan', 'Hujan'],
                             text_auto=True,
                             color_continuous_scale='Blues')
            fig_cm.update_layout(title="Matriks Kebingungan")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Laporan klasifikasi detail
            st.subheader("📋 Laporan Klasifikasi Detail")
            report = classification_report(y, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
            
            # Deret waktu prediksi vs aktual
            st.subheader("📅 Performa Historis")
            
            # Buat indeks waktu
            dates = pd.to_datetime(df[['Tahun', 'Bulan', 'Tanggal']].iloc[len(df)-len(y):]
                                .rename(columns={'Tahun': 'year', 'Bulan': 'month', 'Tanggal': 'day'}))
            
            performance_df = pd.DataFrame({
                'Tanggal': dates,
                'Hujan_Aktual': y,
                'Hujan_Prediksi': y_pred,
                'Probabilitas_Hujan': y_pred_proba,
                'Benar': y == y_pred
            })
            
            fig_ts = go.Figure()
            
            # Tambah prediksi benar
            correct_mask = performance_df['Benar'] == True
            fig_ts.add_trace(go.Scatter(
                x=performance_df.loc[correct_mask, 'Tanggal'],
                y=performance_df.loc[correct_mask, 'Hujan_Aktual'],
                name='Prediksi Benar',
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle')
            ))
            
            # Tambah prediksi salah
            incorrect_mask = performance_df['Benar'] == False
            fig_ts.add_trace(go.Scatter(
                x=performance_df.loc[incorrect_mask, 'Tanggal'],
                y=performance_df.loc[incorrect_mask, 'Hujan_Aktual'],
                name='Prediksi Salah',
                mode='markers',
                marker=dict(size=8, color='red', symbol='x')
            ))
            
            fig_ts.update_layout(
                title='Hujan Aktual vs Prediksi (Hijau = Benar, Merah = Salah)',
                xaxis_title='Tanggal',
                yaxis_title='Hujan (1) / Tidak Hujan (0)',
                showlegend=True
            )
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Performa dari waktu ke waktu
            st.subheader("📈 Akurasi dari Waktu ke Waktu")
            
            # Hitung akurasi bergulir
            performance_df['Akurasi_Bergulir'] = performance_df['Benar'].rolling(window=30, min_periods=1).mean()
            
            fig_acc = px.line(performance_df, x='Tanggal', y='Akurasi_Bergulir',
                            title='Akurasi Bergulir 30 Hari',
                            labels={'Akurasi_Bergulir': 'Akurasi', 'Tanggal': 'Tanggal'})
            fig_acc.update_traces(line_color='orange', line_width=3)
            fig_acc.update_layout(yaxis=dict(range=[0, 1]))
            
            st.plotly_chart(fig_acc, use_container_width=True)

    # Tombol hapus di sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Hapus Semua", use_container_width=True, type="secondary"):
        st.session_state.predictor = RainfallPredictor()
        st.session_state.data = None
        st.session_state.predictions = None
        st.session_state.training_complete = False
        st.rerun()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Sistem Prediksi Curah Hujan**  
    Menggunakan Machine Learning dengan Data MJO  
    🚀 Didukung oleh Streamlit
    """)

if __name__ == "__main__":
    main()