import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style untuk plotting
plt.style.use('default')
sns.set_palette("viridis")

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Clustering Pola Hujan dengan Fuzzy C-Means",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk menentukan kategori curah hujan berdasarkan nilai
def get_rainfall_category(value):
    """Menentukan kategori curah hujan berdasarkan nilai"""
    if value == 0:
        return "Tidak Hujan"
    elif 0.5 <= value <= 20:
        return "Hujan Ringan"
    elif 20 < value <= 50:
        return "Hujan Sedang"
    elif 50 < value <= 100:
        return "Hujan Lebat"
    elif value > 100:
        return "Hujan Sangat Lebat"
    else:
        return "Tidak Diketahui"

# Fungsi untuk menentukan kategori cluster berdasarkan rentang nilai
def get_cluster_category(min_val, max_val):
    """Menentukan kategori cluster berdasarkan rentang nilai min dan max"""
    min_cat = get_rainfall_category(min_val)
    max_cat = get_rainfall_category(max_val)
    
    if min_cat == max_cat:
        return min_cat
    else:
        # Jika rentang mencakup beberapa kategori, gabungkan
        categories = []
        if min_val == 0:
            categories.append("Tidak Hujan")
        if min_val <= 20 and max_val >= 0.5:
            categories.append("Hujan Ringan")
        if min_val <= 50 and max_val > 20:
            categories.append("Hujan Sedang")
        if min_val <= 100 and max_val > 50:
            categories.append("Hujan Lebat")
        if max_val > 100:
            categories.append("Hujan Sangat Lebat")
        
        # Hilangkan duplikat dan urutkan
        unique_categories = list(dict.fromkeys(categories))
        
        if len(unique_categories) == 1:
            return unique_categories[0]
        else:
            return " + ".join(unique_categories)

# Cek ketersediaan library FCM
try:
    import skfuzzy as fuzz
    fcm_available = True
    st.success("Library scikit-fuzzy tersedia.")
except ImportError:
    fcm_available = False
    st.warning("Library scikit-fuzzy tidak tersedia. Menggunakan KMeans sebagai alternatif.")
    from sklearn.cluster import KMeans

# Fungsi untuk membuat animasi proses
def create_process_animation():
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Inisialisasi plot
    x = np.linspace(0, 10, 100)
    line, = ax.plot(x, np.sin(x), lw=2)
    
    # Set judul dan label
    ax.set_title("Proses Clustering Berlangsung...", fontsize=14)
    ax.set_xlabel("Iterasi")
    ax.set_ylabel("Nilai Fungsi Objektif")
    
    # Fungsi animasi
    def update(frame):
        line.set_ydata(np.sin(x + frame / 10.0))
        ax.set_title(f"Proses Clustering Berlangsung... Iterasi: {frame}")
        return line,
    
    # Buat animasi
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    
    return fig

# Fungsi utama aplikasi
def main():
    # Header
    st.title("🌧️ Clustering Pola Hujan dengan Fuzzy C-Means")
    st.markdown("""
    Aplikasi ini melakukan analisis clustering pola curah hujan bulanan menggunakan algoritma Fuzzy C-Means.
    Upload file data curah hujan harian dalam format CSV untuk memulai analisis.
    
    **Format CSV yang diharapkan:**
    - Kolom: Tahun, Bulan, Tanggal, Curah Hujan, Koofesian PC1, Koofesien PC2, Amplitudo PC1+PC2
    - Tanggal: hari dalam bulan (1-31)
    
    **Kategori Curah Hujan:**
    - Tidak Hujan: 0 mm
    - Hujan Ringan: 0,5 - 20 mm
    - Hujan Sedang: 20 - 50 mm
    - Hujan Lebat: 50 - 100 mm
    - Hujan Sangat Lebat: > 100 mm
    """)
    
    # Sidebar
    st.sidebar.header("Pengaturan")
    
    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload File CSV", type=["csv"])
    
    # Parameter clustering
    st.sidebar.subheader("Parameter Clustering")
    
    # Jumlah cluster
    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=4)
    
    # Kekuatan fuzzy (fuzziness parameter)
    m = st.sidebar.slider("Kekuatan Fuzzy (m)", min_value=1.1, max_value=5.0, value=2.0, step=0.1)
    
    # Error tolerance
    error = st.sidebar.slider("Error Tolerance", min_value=0.001, max_value=0.1, value=0.005, step=0.001)
    
    # Maksimum iterasi
    maxiter = st.sidebar.slider("Maksimum Iterasi", min_value=100, max_value=2000, value=1000, step=100)
    
    # Tombol
    train_button = st.sidebar.button("Train Model", key="train")
    test_button = st.sidebar.button("Test Model", key="test")
    clear_button = st.sidebar.button("Clear Results", key="clear")
    
    # Inisialisasi session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'cluster_counts' not in st.session_state:
        st.session_state.cluster_counts = None
    if 'cluster_counts_monthly' not in st.session_state:
        st.session_state.cluster_counts_monthly = None
    if 'metrics' not in st.session_state:
        st.session_state.metrics = None
    if 'centroids' not in st.session_state:
        st.session_state.centroids = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Clear results
    if clear_button:
        st.session_state.df = None
        st.session_state.results_df = None
        st.session_state.cluster_counts = None
        st.session_state.cluster_counts_monthly = None
        st.session_state.metrics = None
        st.session_state.centroids = None
        st.session_state.model_trained = False
        st.experimental_rerun()
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Tampilkan data
            st.subheader("Data Curah Hujan")
            st.write(f"Jumlah baris: {df.shape[0]}")
            st.write(f"Jumlah kolom: {df.shape[1]}")
            st.dataframe(df.head())
            
            # Periksa kolom yang diperlukan
            required_columns = ['Tahun', 'Bulan', 'Tanggal', 'Curah Hujan']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Kolom berikut tidak ditemukan dalam file CSV: {', '.join(missing_columns)}")
                st.error("Pastikan file CSV memiliki semua kolom yang diperlukan.")
            else:
                st.success("Data berhasil dimuat!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Train model
    if train_button and st.session_state.df is not None:
        with st.spinner("Memproses data..."):
            # Tampilkan animasi
            anim_fig = create_process_animation()
            st.pyplot(anim_fig)
            
            # Ambil data dari session state
            df = st.session_state.df.copy()
            
            # Pastikan kolom tanggal memiliki tipe numerik
            df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
            df['Bulan'] = pd.to_numeric(df['Bulan'], errors='coerce')
            df['Tanggal'] = pd.to_numeric(df['Tanggal'], errors='coerce')
            
            # Drop baris jika ada nilai NaN di kolom tanggal
            df.dropna(subset=['Tahun', 'Bulan', 'Tanggal'], inplace=True)
            
            # Ambil data curah hujan
            rainfall_data = df[['Curah Hujan']].values
            
            # Create 'Bulan_Tahun' string column
            df['Bulan_Tahun'] = df['Tahun'].astype(int).astype(str) + '-' + df['Bulan'].astype(int).astype(str).str.zfill(2)
            
            if fcm_available:
                # Fuzzy C-Means Clustering
                # Transpose data for skfuzzy
                data_for_fcm = rainfall_data.T
                
                cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                    data_for_fcm, n_clusters, m, error=error, maxiter=maxiter, init=None
                )
                
                # Ambil cluster dengan membership tertinggi
                cluster_labels = np.argmax(u, axis=0)
                
                # Calculate MSE and RMSE
                distances_squared = np.min(d, axis=0) ** 2
                mse = np.mean(distances_squared)
                rmse = np.sqrt(mse)
                
                # Centroids from FCM
                centroids = cntr.flatten()
                
                # Simpan hasil
                st.session_state.metrics = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'FPC': fpc
                }
                st.session_state.centroids = centroids
                
                # Create results dataframe
                results_df = df[['Tahun', 'Bulan', 'Tanggal', 'Curah Hujan']].copy()
                results_df['Cluster'] = cluster_labels
                results_df['Bulan_Tahun'] = df['Bulan_Tahun']
                
                # Tambahkan membership score
                results_df['Membership_Score'] = np.max(u, axis=0)
                results_df['Uncertainty'] = 1 - results_df['Membership_Score']
                
                st.session_state.results_df = results_df
                
                # Group by month and count clusters
                cluster_counts = results_df.groupby(['Bulan_Tahun', 'Cluster']).size().unstack(fill_value=0)
                # Keep integer columns for easier processing
                st.session_state.cluster_counts = cluster_counts
                
                # Group by month (across all years) and count clusters
                cluster_counts_monthly = results_df.groupby(['Bulan', 'Cluster']).size().unstack(fill_value=0)
                # Keep integer columns for easier processing
                st.session_state.cluster_counts_monthly = cluster_counts_monthly
                st.session_state.model_trained = True
                
                st.success(f"Model berhasil dilatih dengan FPC: {fpc:.4f}")
            else:
                # Fallback using KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(rainfall_data)
                centroids = kmeans.cluster_centers_.flatten()
                
                # Calculate distances for MSE/RMSE
                distances_squared = np.array([np.linalg.norm(rainfall_data[i] - centroids[cluster_labels[i]])**2 for i in range(len(rainfall_data))])
                mse = np.mean(distances_squared)
                rmse = np.sqrt(mse)
                
                # Simpan hasil
                st.session_state.metrics = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'FPC': None  # KMeans tidak memiliki FPC
                }
                st.session_state.centroids = centroids
                
                # Create results dataframe
                results_df = df[['Tahun', 'Bulan', 'Tanggal', 'Curah Hujan']].copy()
                results_df['Cluster'] = cluster_labels
                results_df['Bulan_Tahun'] = df['Bulan_Tahun']
                
                # Untuk KMeans, kita buat membership score dummy
                results_df['Membership_Score'] = 1.0  # KMeans adalah hard clustering
                results_df['Uncertainty'] = 0.0
                
                st.session_state.results_df = results_df
                
                # Group by month and count clusters
                cluster_counts = results_df.groupby(['Bulan_Tahun', 'Cluster']).size().unstack(fill_value=0)
                st.session_state.cluster_counts = cluster_counts
                
                # Group by month (across all years) and count clusters
                cluster_counts_monthly = results_df.groupby(['Bulan', 'Cluster']).size().unstack(fill_value=0)
                st.session_state.cluster_counts_monthly = cluster_counts_monthly
                st.session_state.model_trained = True
                
                st.success("Model berhasil dilatih dengan KMeans (fallback)")
    
    # Test model
    if test_button:
        if not st.session_state.model_trained:
            st.warning("Silakan latih model terlebih dahulu dengan menekan tombol 'Train Model'!")
        else:
            st.subheader("Hasil Evaluasi Model")
            
            # Tampilkan metrik
            metrics = st.session_state.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("MSE", f"{metrics['MSE']:.4f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            if metrics['FPC'] is not None:
                col3.metric("FPC", f"{metrics['FPC']:.4f}")
            else:
                col3.metric("FPC", "N/A (KMeans)")
            
            # Tampilkan centroids
            st.write(f"**Centroids:** {st.session_state.centroids}")
            
            # Analisis karakteristik cluster berdasarkan min dan max
            st.subheader("Karakteristik Cluster (Berdasarkan Rentang Nilai)")
            
            cluster_info = []
            for cluster in sorted(st.session_state.results_df['Cluster'].unique()):
                cluster_data = st.session_state.results_df[st.session_state.results_df['Cluster'] == cluster]
                min_val = cluster_data['Curah Hujan'].min()
                max_val = cluster_data['Curah Hujan'].max()
                count = len(cluster_data)
                
                # Tentukan kategori berdasarkan rentang
                category = get_cluster_category(min_val, max_val)
                
                cluster_info.append({
                    'Cluster': cluster,
                    'Kategori': category,
                    'Min (mm)': min_val,
                    'Max (mm)': max_val,
                    'Jumlah Data': count
                })
            
            # Tampilkan tabel karakteristik cluster
            cluster_info_df = pd.DataFrame(cluster_info)
            st.dataframe(cluster_info_df)
            
            # Simpan nama cluster untuk visualisasi
            cluster_names = {info['Cluster']: info['Kategori'] for info in cluster_info}
            
            # Tampilkan distribusi kategori dalam setiap cluster
            st.subheader("Distribusi Kategori dalam Setiap Cluster")
            
            # Tambahkan kategori untuk setiap data point
            st.session_state.results_df['Kategori'] = st.session_state.results_df['Curah Hujan'].apply(get_rainfall_category)
            
            # Hitung distribusi kategori per cluster
            category_dist = pd.crosstab(
                st.session_state.results_df['Cluster'], 
                st.session_state.results_df['Kategori'],
                normalize='index'
            ) * 100
            
            # Format persentase
            category_dist = category_dist.round(2)
            category_dist = category_dist.astype(str) + '%'
            
            st.dataframe(category_dist)
            
            # Visualisasi
            st.subheader("Visualisasi Hasil Clustering")
            
            # Setup figure
            fig = plt.figure(figsize=(20, 20))  # Diperbesar untuk menambah subplot baru
            
            # 1. Scatter plot clustering
            ax1 = plt.subplot(4, 3, 1)
            scatter = ax1.scatter(range(len(st.session_state.results_df)), 
                               st.session_state.results_df['Curah Hujan'], 
                               c=st.session_state.results_df['Cluster'], 
                               cmap='viridis', 
                               alpha=0.7,
                               s=st.session_state.results_df['Membership_Score']*100,
                               edgecolor='black', linewidth=0.5)
            ax1.set_xlabel('Data Point')
            ax1.set_ylabel('Curah Hujan (mm)')
            ax1.set_title('Scatter Plot Clustering\n(Ukuran = Tingkat Keanggotaan)', fontsize=12, fontweight='bold')
            plt.colorbar(scatter, ax=ax1, label='Cluster')
            ax1.grid(True, alpha=0.3)
            
            # 2. Distribusi cluster per bulan-tahun
            ax2 = plt.subplot(4, 3, 2)
            # Buat salinan dengan nama kolom string untuk display
            cluster_counts_display = st.session_state.cluster_counts.copy()
            cluster_counts_display.columns = [f'Cluster {i}' for i in cluster_counts_display.columns]
            cluster_counts_display.plot(kind='bar', stacked=True, ax=ax2, colormap='Set3')
            ax2.set_title('Distribusi Cluster per Bulan-Tahun', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Bulan-Tahun')
            ax2.set_ylabel('Jumlah Kejadian')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend(title='Cluster')
            ax2.grid(True, alpha=0.3)
            
            # 3. Boxplot curah hujan per cluster
            ax3 = plt.subplot(4, 3, 3)
            box_data = [st.session_state.results_df[st.session_state.results_df['Cluster'] == cluster]['Curah Hujan'] 
                       for cluster in sorted(st.session_state.results_df['Cluster'].unique())]
            
            box_plot = ax3.boxplot(box_data, labels=[f'{cluster_names[cluster]}' for cluster in sorted(st.session_state.results_df['Cluster'].unique())])
            ax3.set_ylabel('Curah Hujan (mm)')
            ax3.set_title('Distribusi Curah Hujan per Cluster', fontsize=12, fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # 4. Distribusi kategori per cluster
            ax4 = plt.subplot(4, 3, 4)
            category_counts = pd.crosstab(
                st.session_state.results_df['Cluster'], 
                st.session_state.results_df['Kategori']
            )
            category_counts.plot(kind='bar', stacked=True, ax=ax4, colormap='viridis')
            ax4.set_title('Distribusi Kategori per Cluster', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Cluster')
            ax4.set_ylabel('Jumlah Data')
            ax4.legend(title='Kategori', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # 5. Time series curah hujan dengan cluster
            ax5 = plt.subplot(4, 3, 5)
            # Buat tanggal
            st.session_state.results_df['Date'] = pd.to_datetime(
                st.session_state.results_df['Tahun'].astype(str) + '-' + 
                st.session_state.results_df['Bulan'].astype(str) + '-' + 
                st.session_state.results_df['Tanggal'].astype(str)
            )
            
            for cluster in sorted(st.session_state.results_df['Cluster'].unique()):
                cluster_data = st.session_state.results_df[st.session_state.results_df['Cluster'] == cluster]
                ax5.scatter(cluster_data['Date'], cluster_data['Curah Hujan'], 
                          label=f'{cluster_names[cluster]}', 
                          alpha=0.7, s=30)
            
            ax5.set_xlabel('Tanggal')
            ax5.set_ylabel('Curah Hujan (mm)')
            ax5.set_title('Time Series Curah Hujan per Cluster', fontsize=12, fontweight='bold')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
            
            # 6. Pola musiman rata-rata curah hujan per cluster
            ax6 = plt.subplot(4, 3, 6)
            monthly_avg = st.session_state.results_df.groupby(['Bulan', 'Cluster'])['Curah Hujan'].mean().unstack()
            
            for cluster in sorted(st.session_state.results_df['Cluster'].unique()):
                ax6.plot(monthly_avg.index, monthly_avg[cluster], 
                        marker='o', label=f'{cluster_names[cluster]}', 
                        linewidth=2, markersize=6)
            
            ax6.set_xlabel('Bulan')
            ax6.set_ylabel('Rata-rata Curah Hujan (mm)')
            ax6.set_title('Pola Musiman per Cluster', fontsize=12, fontweight='bold')
            ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax6.set_xticks(range(1, 13))
            ax6.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax6.grid(True, alpha=0.3)
            
            # 7. Rentang nilai per cluster
            ax7 = plt.subplot(4, 3, 7)
            cluster_ranges = []
            cluster_labels_range = []
            
            for cluster in sorted(st.session_state.results_df['Cluster'].unique()):
                cluster_data = st.session_state.results_df[st.session_state.results_df['Cluster'] == cluster]
                min_val = cluster_data['Curah Hujan'].min()
                max_val = cluster_data['Curah Hujan'].max()
                cluster_ranges.append([min_val, max_val])
                cluster_labels_range.append(f'{cluster_names[cluster]}\n({min_val:.1f} - {max_val:.1f} mm)')
            
            for i, (min_val, max_val) in enumerate(cluster_ranges):
                ax7.barh(i, max_val - min_val, left=min_val, height=0.6, alpha=0.7)
                ax7.text(min_val + (max_val - min_val)/2, i, f'{min_val:.1f} - {max_val:.1f}', 
                        ha='center', va='center', fontweight='bold')
            
            ax7.set_yticks(range(len(cluster_ranges)))
            ax7.set_yticklabels(cluster_labels_range)
            ax7.set_xlabel('Curah Hujan (mm)')
            ax7.set_title('Rentang Nilai per Cluster', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3)
            
            # 8. Heatmap pola hujan per bulan dan cluster
            ax8 = plt.subplot(4, 3, 8)
            heatmap_data = st.session_state.results_df.pivot_table(
                values='Curah Hujan', 
                index='Bulan', 
                columns='Cluster', 
                aggfunc='mean'
            )
            
            sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f', 
                       cbar_kws={'label': 'Rata-rata Curah Hujan (mm)'}, ax=ax8)
            ax8.set_title('Heatmap Rata-rata Curah Hujan\nper Bulan dan Cluster', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Cluster')
            ax8.set_ylabel('Bulan')
            
            # 9. Jumlah cluster per bulan (semua tahun)
            ax9 = plt.subplot(4, 3, 9)
            # Buat salinan dengan nama kolom string untuk display
            cluster_counts_monthly_display = st.session_state.cluster_counts_monthly.copy()
            cluster_counts_monthly_display.columns = [f'Cluster {i}' for i in cluster_counts_monthly_display.columns]
            cluster_counts_monthly_display.plot(kind='bar', stacked=True, ax=ax9, colormap='viridis')
            ax9.set_title('Jumlah Cluster per Bulan\n(Semua Tahun)', fontsize=12, fontweight='bold')
            ax9.set_xlabel('Bulan')
            ax9.set_ylabel('Jumlah Kejadian')
            ax9.set_xticks(range(len(cluster_counts_monthly_display)))
            ax9.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax9.legend(title='Cluster')
            ax9.grid(True, alpha=0.3)
            
            # 10. Persentase distribusi cluster per bulan (semua tahun)
            ax10 = plt.subplot(4, 3, 10)
            # Hitung persentase
            cluster_percent_monthly = st.session_state.cluster_counts_monthly.div(st.session_state.cluster_counts_monthly.sum(axis=1), axis=0) * 100
            cluster_percent_monthly_display = cluster_percent_monthly.copy()
            cluster_percent_monthly_display.columns = [f'Cluster {i}' for i in cluster_percent_monthly_display.columns]
            cluster_percent_monthly_display.plot(kind='bar', stacked=True, ax=ax10, colormap='plasma')
            ax10.set_title('Persentase Distribusi Cluster per Bulan\n(Semua Tahun)', fontsize=12, fontweight='bold')
            ax10.set_xlabel('Bulan')
            ax10.set_ylabel('Persentase (%)')
            ax10.set_xticks(range(len(cluster_percent_monthly_display)))
            ax10.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax10.legend(title='Cluster')
            ax10.grid(True, alpha=0.3)
            
            # 11. Pola dominan per bulan - PERBAIKAN DI SINI
            ax11 = plt.subplot(4, 3, 11)
            # Tentukan cluster dominan untuk setiap bulan
            dominant_cluster = st.session_state.cluster_counts_monthly.idxmax(axis=1)
            dominant_counts = st.session_state.cluster_counts_monthly.max(axis=1)
            
            # Generate colors for each cluster
            n_clusters_actual = len(st.session_state.cluster_counts_monthly.columns)
            colors = plt.cm.viridis(np.linspace(0, 1, n_clusters_actual))
            
            # Create list of colors for each bar based on dominant cluster
            bar_colors = []
            for cluster_idx in dominant_cluster:
                bar_colors.append(colors[cluster_idx])
            
            bars = ax11.bar(range(len(dominant_cluster)), dominant_counts, color=bar_colors)
            
            ax11.set_xlabel('Bulan')
            ax11.set_ylabel('Jumlah Kejadian')
            ax11.set_title('Cluster Dominan per Bulan\n(Semua Tahun)', fontsize=12, fontweight='bold')
            ax11.set_xticks(range(len(dominant_cluster)))
            ax11.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax11.grid(True, alpha=0.3)
            
            # Tambahkan label cluster dominan
            for i, (cluster_idx, count) in enumerate(zip(dominant_cluster, dominant_counts)):
                ax11.text(i, count + max(dominant_counts) * 0.01, cluster_names[cluster_idx], 
                         ha='center', va='bottom', fontsize=8, rotation=45)
            
            # 12. Ringkasan statistik per bulan
            ax12 = plt.subplot(4, 3, 12)
            # Hitung statistik per bulan
            monthly_stats = st.session_state.results_df.groupby('Bulan')['Curah Hujan'].agg(['count', 'mean', 'std'])
            
            # Plot rata-rata dengan error bar
            ax12.errorbar(monthly_stats.index, monthly_stats['mean'], yerr=monthly_stats['std'], 
                         fmt='-o', capsize=5, capthick=2, color='skyblue', ecolor='darkblue')
            ax12.set_xlabel('Bulan')
            ax12.set_ylabel('Curah Hujan (mm)')
            ax12.set_title('Rata-rata Curah Hujan per Bulan\n(dengan Standar Deviasi)', fontsize=12, fontweight='bold')
            ax12.set_xticks(range(1, 13))
            ax12.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax12.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tampilkan count cluster per bulan-tahun
            st.subheader("Jumlah Cluster per Bulan-Tahun")
            # Tampilkan dengan nama kolom string
            cluster_counts_display = st.session_state.cluster_counts.copy()
            cluster_counts_display.columns = [f'Cluster {i}' for i in cluster_counts_display.columns]
            st.dataframe(cluster_counts_display)
            
            # Tampilkan count cluster per bulan (semua tahun)
            st.subheader("Jumlah Cluster per Bulan (Semua Tahun)")
            # Tampilkan dengan nama kolom string
            cluster_counts_monthly_display = st.session_state.cluster_counts_monthly.copy()
            cluster_counts_monthly_display.columns = [f'Cluster {i}' for i in cluster_counts_monthly_display.columns]
            st.dataframe(cluster_counts_monthly_display)
            
            # Tampilkan ringkasan statistik per bulan
            st.subheader("Ringkasan Statistik Curah Hujan per Bulan")
            st.dataframe(monthly_stats.round(2))
            
            # Download hasil
            st.subheader("Download Hasil")
            
            # Download data hasil clustering
            csv = st.session_state.results_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_clustering.csv">Download Data Hasil Clustering (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
            
            # Download cluster counts per bulan-tahun
            csv_counts = cluster_counts_display.to_csv()
            b64_counts = base64.b64encode(csv_counts.encode()).decode()
            href_counts = f'<a href="data:file/csv;base64,{b64_counts}" download="cluster_counts_by_month_year.csv">Download Cluster Counts by Month-Year (CSV)</a>'
            st.markdown(href_counts, unsafe_allow_html=True)
            
            # Download cluster counts per bulan (semua tahun)
            csv_counts_monthly = cluster_counts_monthly_display.to_csv()
            b64_counts_monthly = base64.b64encode(csv_counts_monthly.encode()).decode()
            href_counts_monthly = f'<a href="data:file/csv;base64,{b64_counts_monthly}" download="cluster_counts_by_month_all_years.csv">Download Cluster Counts by Month (All Years) (CSV)</a>'
            st.markdown(href_counts_monthly, unsafe_allow_html=True)
            
            # Download gambar
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            href_img = f'<a href="data:image/png;base64,{img_str}" download="visualisasi_clustering.png">Download Visualisasi (PNG)</a>'
            st.markdown(href_img, unsafe_allow_html=True)

if __name__ == "__main__":
    main()